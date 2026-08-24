//! Deterministic no-storage planning runtime; it never fabricates values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NullBuffer {
    pub bytes: usize,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NullTrace {
    Allocate(usize),
    Copy { bytes: usize },
    Launch { name: String },
}
#[derive(Default)]
pub struct NullRuntime {
    trace: Vec<NullTrace>,
}
impl NullRuntime {
    pub fn allocate(&mut self, bytes: usize) -> NullBuffer {
        self.trace.push(NullTrace::Allocate(bytes));
        NullBuffer { bytes }
    }
    pub fn copy(
        &mut self,
        dst: &NullBuffer,
        src: &NullBuffer,
        bytes: usize,
    ) -> Result<(), &'static str> {
        if bytes > dst.bytes || bytes > src.bytes {
            return Err("null copy bounds");
        }
        self.trace.push(NullTrace::Copy { bytes });
        Ok(())
    }
    pub fn launch(&mut self, name: &str) {
        self.trace.push(NullTrace::Launch { name: name.into() })
    }
    pub fn trace(&self) -> &[NullTrace] {
        &self.trace
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trace_and_bounds_are_deterministic() {
        let mut r = NullRuntime::default();
        let a = r.allocate(4);
        let b = r.allocate(8);
        r.copy(&b, &a, 4).unwrap();
        assert!(r.copy(&a, &b, 5).is_err());
        r.launch("checked");
        assert_eq!(
            r.trace(),
            &[
                NullTrace::Allocate(4),
                NullTrace::Allocate(8),
                NullTrace::Copy { bytes: 4 },
                NullTrace::Launch {
                    name: "checked".into()
                }
            ]
        );
    }
}
