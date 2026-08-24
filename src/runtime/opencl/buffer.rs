//! Sealed logical-buffer identity and generation-backed physical allocations.
use super::{OpenClError, RawBuffer, resource::ContextInner};
use crate::DType;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LogicalBufferDesc {
    pub bytes: usize,
    pub dtype: Option<DType>,
}

pub(super) struct PhysicalBuffer {
    context: Rc<ContextInner>,
    raw: Option<RawBuffer>,
}

impl PhysicalBuffer {
    pub fn allocate(context: Rc<ContextInner>, bytes: usize) -> Result<Rc<Self>, OpenClError> {
        context.live()?;
        let raw = if bytes == 0 {
            None
        } else {
            Some(
                context
                    .dispatch
                    .buffer_create(context.raw, bytes, context.owner)?,
            )
        };
        Ok(Rc::new(Self { context, raw }))
    }

    pub fn raw(&self) -> Option<RawBuffer> {
        self.raw
    }
}

impl Drop for PhysicalBuffer {
    fn drop(&mut self) {
        if let Some(raw) = self.raw {
            let _ = self
                .context
                .dispatch
                .buffer_release(raw, self.context.owner);
        }
    }
}

struct VisibleGeneration {
    generation: u64,
    physical: Rc<PhysicalBuffer>,
}

struct LogicalBuffer {
    context: Rc<ContextInner>,
    desc: LogicalBufferDesc,
    visible: RefCell<VisibleGeneration>,
    closed: Cell<bool>,
}

/// Stable logical OpenCL buffer identity whose visible physical generation may
/// change only after a staged transaction has completed successfully.
pub struct OpenClBuffer {
    inner: Rc<LogicalBuffer>,
}

#[derive(Clone)]
pub(super) struct BufferSnapshot {
    logical: Rc<LogicalBuffer>,
    generation: u64,
    physical: Rc<PhysicalBuffer>,
}

impl BufferSnapshot {
    pub fn raw(&self) -> Option<RawBuffer> {
        self.physical.raw()
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn physical(&self) -> Rc<PhysicalBuffer> {
        self.physical.clone()
    }

    pub fn validate_current(&self) -> Result<(), OpenClError> {
        let actual = self.logical.visible.borrow().generation;
        if actual == self.generation {
            Ok(())
        } else {
            Err(OpenClError::StaleGeneration {
                expected: self.generation,
                actual,
            })
        }
    }
}

impl OpenClBuffer {
    pub(super) fn allocate(
        context: Rc<ContextInner>,
        bytes: usize,
        dtype: Option<DType>,
    ) -> Result<Self, OpenClError> {
        let physical = PhysicalBuffer::allocate(context.clone(), bytes)?;
        Ok(Self {
            inner: Rc::new(LogicalBuffer {
                context,
                desc: LogicalBufferDesc { bytes, dtype },
                visible: RefCell::new(VisibleGeneration {
                    generation: 1,
                    physical,
                }),
                closed: Cell::new(false),
            }),
        })
    }

    pub fn len(&self) -> usize {
        self.inner.desc.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn owner_id(&self) -> u64 {
        self.inner.context.owner
    }
    pub fn generation(&self) -> u64 {
        self.inner.visible.borrow().generation
    }
    pub fn dtype(&self) -> Option<DType> {
        self.inner.desc.dtype
    }

    pub(super) fn snapshot(
        &self,
        context: &Rc<ContextInner>,
        offset: usize,
        bytes: usize,
        dtype: Option<DType>,
    ) -> Result<BufferSnapshot, OpenClError> {
        self.inner.context.live()?;
        if self.inner.closed.get() {
            return Err(OpenClError::Closed("buffer"));
        }
        if !Rc::ptr_eq(&self.inner.context, context) {
            return Err(OpenClError::OwnerMismatch);
        }
        let end = offset.checked_add(bytes).ok_or(OpenClError::Overflow)?;
        if end > self.inner.desc.bytes {
            return Err(OpenClError::Bounds);
        }
        if let (Some(actual), Some(expected)) = (self.inner.desc.dtype, dtype)
            && actual != expected
        {
            return Err(OpenClError::InvalidBinding(
                "logical buffer dtype mismatch".into(),
            ));
        }
        let visible = self.inner.visible.borrow();
        Ok(BufferSnapshot {
            logical: self.inner.clone(),
            generation: visible.generation,
            physical: visible.physical.clone(),
        })
    }

    pub(super) fn candidate(&self) -> Result<Rc<PhysicalBuffer>, OpenClError> {
        PhysicalBuffer::allocate(self.inner.context.clone(), self.inner.desc.bytes)
    }

    pub(super) fn commit_candidate(
        &self,
        expected: u64,
        candidate: Rc<PhysicalBuffer>,
    ) -> Result<u64, OpenClError> {
        self.inner.context.live()?;
        if self.inner.closed.get() {
            return Err(OpenClError::Closed("buffer"));
        }
        let mut visible = self.inner.visible.borrow_mut();
        if visible.generation != expected {
            return Err(OpenClError::StaleGeneration {
                expected,
                actual: visible.generation,
            });
        }
        let next = expected.checked_add(1).ok_or(OpenClError::Overflow)?;
        *visible = VisibleGeneration {
            generation: next,
            physical: candidate,
        };
        Ok(next)
    }
}

impl Drop for OpenClBuffer {
    fn drop(&mut self) {
        self.inner.closed.set(true);
    }
}
