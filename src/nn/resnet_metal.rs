//! Typed persistent Metal deployment for static Eval/F32 ResNet inference.

use super::{Mode, ResNet};
use crate::runtime::metal::{
    MetalDevice, MetalDeviceInfo, MetalDeviceRun, MetalDeviceRunReport, MetalDeviceSession,
    MetalDeviceSessionSummary, MetalError, MetalInferencePlan, MetalPlanOptions, RenderedMetal,
};
use crate::{
    BufferDesc, CapturedInference, CapturedInferenceError, CapturedSchedule, DType, Error,
    ExecutionPlanSummary, Graph, NodeId, ReplayInput, Shape, TensorData,
};
use std::{collections::BTreeMap, error, fmt};

const IMAGE_INPUT: &str = "image";

/// Preparation-free, inspectable deployment of one static Eval/F32 ResNet graph.
///
/// Construction snapshots every capture-admitted module value and binds the
/// plan to the exact selected [`MetalDevice`] owner. The source model may be
/// mutated or dropped after this value is returned.
pub struct ResNetMetalPlan {
    inner: MetalInferencePlan,
    graph: Graph,
    image: ReplayInput,
    logits: NodeId,
    logits_desc: BufferDesc,
    selected_device: MetalDevice,
}

/// Persistent, thread-confined Metal resources for one ResNet deployment.
pub struct ResNetMetalSession {
    inner: MetalDeviceSession,
    image_shape: Shape,
    logits_shape: Shape,
}

/// Detached logits and measurements committed by one successful invocation.
pub struct ResNetMetalRun {
    inner: MetalDeviceRun,
}

/// Typed graph, capture, device, or invocation failure for the ResNet facade.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResNetMetalError {
    Graph(Error),
    Capture(CapturedInferenceError),
    Metal(MetalError),
    ClassifierRequired,
    InvalidImage {
        expected_shape: Shape,
        actual_shape: Shape,
        actual_dtype: DType,
    },
    Contract(&'static str),
}

impl fmt::Display for ResNetMetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "ResNet graph: {error}"),
            Self::Capture(error) => write!(f, "ResNet capture: {error}"),
            Self::Metal(error) => write!(f, "ResNet Metal: {error}"),
            Self::ClassifierRequired => write!(f, "ResNet Metal inference requires a classifier"),
            Self::InvalidImage {
                expected_shape,
                actual_shape,
                actual_dtype,
            } => write!(
                f,
                "ResNet Metal image must be F32 {expected_shape}, got {actual_dtype:?} {actual_shape}"
            ),
            Self::Contract(reason) => write!(f, "invalid ResNet Metal contract: {reason}"),
        }
    }
}

impl error::Error for ResNetMetalError {}

impl From<Error> for ResNetMetalError {
    fn from(value: Error) -> Self {
        Self::Graph(value)
    }
}

impl From<CapturedInferenceError> for ResNetMetalError {
    fn from(value: CapturedInferenceError) -> Self {
        Self::Capture(value)
    }
}

impl From<MetalError> for ResNetMetalError {
    fn from(value: MetalError) -> Self {
        Self::Metal(value)
    }
}

impl ResNetMetalPlan {
    /// Builds, captures, and renders one complete static Eval/F32 ResNet graph.
    /// No queue, pipeline, or buffer is created during this operation.
    pub fn eval_f32(
        model: &ResNet,
        device: &MetalDevice,
        input_shape: [usize; 4],
        options: MetalPlanOptions,
    ) -> Result<Self, ResNetMetalError> {
        let mut graph = Graph::new();
        let image_node = graph.input_dtype(IMAGE_INPUT, input_shape, DType::F32);
        let logits = {
            let forward = model.forward_mode(&mut graph, image_node, Mode::Eval)?;
            if !forward.pending.is_empty() {
                return Err(ResNetMetalError::Contract(
                    "Eval forward retained mutable effects",
                ));
            }
            forward
                .output
                .logits()
                .ok_or(ResNetMetalError::ClassifierRequired)?
        };
        let expected_logits = Shape::new([
            input_shape[0],
            model
                .config()
                .num_classes
                .ok_or(ResNetMetalError::ClassifierRequired)?,
        ]);
        if graph.shape(logits)? != &expected_logits || graph.dtype(logits)? != DType::F32 {
            return Err(ResNetMetalError::Contract(
                "Eval classifier output descriptor is inconsistent",
            ));
        }

        let inference = CapturedInference::from_module_graph(model, &graph, &[logits])?;
        let renderer = device.renderer(options.local_size)?;
        let inner = MetalInferencePlan::new(inference, renderer)?;
        let [image] = inner.transient_inputs() else {
            return Err(ResNetMetalError::Contract(
                "ResNet capture must expose exactly one transient image",
            ));
        };
        if image.name != IMAGE_INPUT
            || image.node != image_node
            || image.desc.id != image_node.index() as u64
            || image.desc.shape != Shape::new(input_shape)
            || image.desc.dtype != DType::F32
            || image.desc.view.is_some()
            || !image.desc.read_only
        {
            return Err(ResNetMetalError::Contract(
                "ResNet image capture descriptor is inconsistent",
            ));
        }
        let [logits_desc] = inner.execution_plan().requested_outputs.as_slice() else {
            return Err(ResNetMetalError::Contract(
                "ResNet capture must retain exactly one logits output",
            ));
        };
        if logits_desc.shape != expected_logits || logits_desc.dtype != DType::F32 {
            return Err(ResNetMetalError::Contract(
                "ResNet captured logits descriptor is inconsistent",
            ));
        }
        if logits_desc.id != logits.index() as u64 {
            return Err(ResNetMetalError::Contract(
                "ResNet captured logits owner is inconsistent",
            ));
        }
        if inner.summary().fallback_count != 0 {
            return Err(ResNetMetalError::Contract(
                "strict Metal plan admitted a fallback",
            ));
        }

        Ok(Self {
            graph,
            image: image.clone(),
            logits,
            logits_desc: logits_desc.clone(),
            selected_device: device.clone(),
            inner,
        })
    }

    /// Returns the exact graph whose immutable module leaves were captured.
    pub const fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the graph node retained as the classifier output.
    pub const fn logits_node(&self) -> NodeId {
        self.logits
    }

    /// Returns the only typed transient input accepted by the session.
    pub const fn image_input(&self) -> &ReplayInput {
        &self.image
    }

    /// Returns the exact requested logits descriptor.
    pub const fn logits_output(&self) -> &BufferDesc {
        &self.logits_desc
    }

    /// Returns handle-free identity and capabilities for the selected device.
    pub fn selected_device_info(&self) -> &MetalDeviceInfo {
        self.selected_device.info()
    }

    /// Returns the stable owner identity authenticated again at preparation.
    pub fn selected_device_owner_id(&self) -> u64 {
        self.selected_device.owner_id()
    }

    /// Returns the complete authenticated concrete capture.
    pub fn capture(&self) -> &CapturedSchedule {
        self.inner.capture()
    }

    /// Returns backend-neutral schedule and memory-plan facts.
    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.inner.execution_plan()
    }

    /// Returns immutable parameter leaves uploaded once during preparation.
    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.inner.resident_inputs()
    }

    /// Returns the per-invocation input schema (exactly `image`).
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.inner.transient_inputs()
    }

    /// Returns deterministic planned resource, kernel, and transfer facts.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    /// Returns every scheduled item rendered to MSL, including zero-work items.
    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    /// Exposes the generic plan for existing scoreboard and inspection tools.
    pub const fn metal_plan(&self) -> &MetalInferencePlan {
        &self.inner
    }

    /// Creates persistent resources on the exact device selected at planning.
    pub fn prepare(self) -> Result<ResNetMetalSession, ResNetMetalError> {
        let image_shape = self.image.desc.shape.clone();
        let logits_shape = self.logits_desc.shape.clone();
        let inner = self.inner.prepare(self.selected_device)?;
        Ok(ResNetMetalSession {
            inner,
            image_shape,
            logits_shape,
        })
    }
}

impl ResNetMetalSession {
    /// Exposes the underlying strict session for detailed inspection tooling.
    pub const fn metal_session(&self) -> &MetalDeviceSession {
        &self.inner
    }

    /// Executes one exact F32 NCHW image and returns detached logits plus metrics.
    pub fn run(&mut self, image: TensorData) -> Result<ResNetMetalRun, ResNetMetalError> {
        if image.shape() != &self.image_shape || image.dtype() != DType::F32 {
            return Err(ResNetMetalError::InvalidImage {
                expected_shape: self.image_shape.clone(),
                actual_shape: image.shape().clone(),
                actual_dtype: image.dtype(),
            });
        }
        let run = self
            .inner
            .run(&BTreeMap::from([(IMAGE_INPUT.to_owned(), image)]))?;
        debug_assert_eq!(run.outputs().len(), 1);
        debug_assert_eq!(run.outputs()[0].shape(), &self.logits_shape);
        debug_assert_eq!(run.outputs()[0].dtype(), DType::F32);
        Ok(ResNetMetalRun { inner: run })
    }
}

impl ResNetMetalRun {
    /// Returns the detached `[batch, classes]` F32 logits.
    pub fn logits(&self) -> &TensorData {
        &self.inner.outputs()[0]
    }

    /// Returns metrics committed by this successful synchronous invocation.
    pub fn report(&self) -> &MetalDeviceRunReport {
        self.inner.report()
    }

    /// Exposes the exact committed run to existing scoreboard tooling.
    pub fn metal_run(&self) -> &MetalDeviceRun {
        &self.inner
    }

    /// Splits the run into owned logits and its committed report.
    pub fn into_parts(self) -> (TensorData, MetalDeviceRunReport) {
        let (outputs, report) = self.inner.into_parts();
        let [logits]: [TensorData; 1] = outputs
            .try_into()
            .expect("authenticated ResNet capture has one requested output");
        (logits, report)
    }
}
