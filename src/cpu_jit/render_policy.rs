use super::{JitError, RENDERER_VERSION, THREEFRY_RENDERER_VERSION, native_cache_key};
use crate::{MatmulValue, MovementValue, Operation, UOp, VectorPlan};

#[cfg(test)]
thread_local! {
    static POLICY_LINEAR_DERIVATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) struct ElementwiseTopology {
    nodes: Vec<UOp>,
    request_vector: bool,
    projected: bool,
}

impl ElementwiseTopology {
    pub(super) fn new(root: &UOp, request_vector: bool) -> Result<Self, JitError> {
        let nodes = root
            .topological()
            .map_err(|error| JitError::Unsupported(error.to_string()))?;
        let projected = nodes
            .iter()
            .any(crate::projected_index::ProjectedIndexPlan::is_projected);
        Ok(Self {
            nodes,
            request_vector,
            projected,
        })
    }

    pub(super) fn nodes(&self) -> &[UOp] {
        &self.nodes
    }

    pub(super) fn into_render_policy(
        self,
        root: &UOp,
    ) -> Result<ElementwiseRenderPolicy, JitError> {
        ElementwiseRenderPolicy::from_topology(root, self)
    }

    #[cfg(test)]
    fn cache_policy_without_linear_validation(
        &self,
        root: &UOp,
    ) -> Result<NativeCachePolicy, JitError> {
        if !self.request_vector || self.projected {
            return Ok(NativeCachePolicy::Ordinary(String::new()));
        }
        let linear = linear_kernel(root)?;
        let vector_program = vector_program(&linear)?;
        Ok(NativeCachePolicy::Ordinary(
            if vector_program.b1_eligibility().is_ok() {
                format!("b1-{}", vector_program.cache_key)
            } else {
                String::new()
            },
        ))
    }
}

pub(super) struct ElementwiseRenderPolicy {
    reported_vector: VectorPlan,
    rendered_vector: VectorPlan,
    linear_key: Option<(u64, usize, usize, usize, u64)>,
    b1_program: Option<crate::VectorProgram>,
}

impl ElementwiseRenderPolicy {
    pub(super) fn new(root: &UOp, request_vector: bool) -> Result<Self, JitError> {
        ElementwiseTopology::new(root, request_vector)?.into_render_policy(root)
    }

    fn from_topology(root: &UOp, topology: ElementwiseTopology) -> Result<Self, JitError> {
        if !topology.request_vector || topology.projected {
            return Ok(Self {
                reported_vector: VectorPlan {
                    lanes: 1,
                    enabled: false,
                    reason: if topology.projected && topology.request_vector {
                        "projected indices use the checked scalar address dialect"
                    } else {
                        "disabled"
                    }
                    .into(),
                },
                rendered_vector: VectorPlan {
                    lanes: 1,
                    enabled: false,
                    reason: "disabled".into(),
                },
                linear_key: None,
                b1_program: None,
            });
        }

        let (linear, vector) = elementwise_vector_plan(root)?;
        let vector_program = vector_program(&linear)?;
        let linear_key = Some((
            linear.cache_key,
            linear.program.instructions.len(),
            linear.program.peak_scalar,
            linear.program.peak_vector,
            vector_program.cache_key,
        ));
        let b1_program = vector_program
            .b1_eligibility()
            .is_ok()
            .then_some(vector_program);
        Ok(Self {
            reported_vector: vector.clone(),
            rendered_vector: vector,
            linear_key,
            b1_program,
        })
    }

    pub(super) fn reported_vector(&self) -> &VectorPlan {
        &self.reported_vector
    }

    pub(super) fn rendered_vector(&self) -> &VectorPlan {
        &self.rendered_vector
    }

    pub(super) fn linear_key(&self) -> Option<(u64, usize, usize, usize, u64)> {
        self.linear_key
    }

    pub(super) fn b1_program(&self) -> Option<&crate::VectorProgram> {
        self.b1_program.as_ref()
    }

    fn cache_discriminator(&self) -> String {
        self.b1_program
            .as_ref()
            .map(|program| format!("b1-{}", program.cache_key))
            .unwrap_or_default()
    }
}

enum NativeCachePolicy {
    Ordinary(String),
    Conv2d(u64),
}

impl NativeCachePolicy {
    fn key(&self, source: &str) -> String {
        match self {
            Self::Ordinary(discriminator) => native_cache_key(discriminator, source),
            Self::Conv2d(plan) => super::key(
                &(RENDERER_VERSION.to_owned()
                    + std::env::consts::ARCH
                    + std::env::consts::OS
                    + &plan.to_string()
                    + source),
            ),
        }
    }
}

pub(crate) struct NativeRenderAuthentication {
    vector: VectorPlan,
    cache: NativeCachePolicy,
}

impl NativeRenderAuthentication {
    pub(crate) fn new(root: &UOp, request_vector: bool) -> Result<Self, JitError> {
        let (vector, cache) = if let Some(cache) = dedicated_cache_policy(root) {
            (dedicated_vector(root), cache)
        } else {
            let policy = ElementwiseRenderPolicy::new(root, request_vector)?;
            (
                policy.reported_vector().clone(),
                NativeCachePolicy::Ordinary(policy.cache_discriminator()),
            )
        };
        Ok(Self { vector, cache })
    }

    pub(crate) fn vector(&self) -> &VectorPlan {
        &self.vector
    }

    pub(crate) fn cache_key(&self, source: &str) -> String {
        self.cache.key(source)
    }
}

fn dedicated_vector(root: &UOp) -> VectorPlan {
    VectorPlan {
        lanes: 1,
        enabled: false,
        reason: if matches!(root.operation(), Operation::PrefixScan(_)) {
            "prefix scan uses a serial axis loop"
        } else if matches!(root.operation(), Operation::Threefry(_)) {
            "live Threefry uses a dedicated scalar kernel"
        } else {
            "static contraction uses scalar lanes"
        }
        .into(),
    }
}

pub(super) fn vector_plan(root: &UOp) -> Result<VectorPlan, JitError> {
    if matches!(
        root.operation(),
        Operation::Matmul(_)
            | Operation::Conv2d(_)
            | Operation::Movement(_)
            | Operation::Random(_)
            | Operation::PrefixScan(_)
            | Operation::Threefry(_)
    ) {
        return Ok(dedicated_vector(root));
    }
    let topology = ElementwiseTopology::new(root, true)?;
    if topology.projected {
        return Ok(VectorPlan {
            lanes: 1,
            enabled: false,
            reason: "projected indices use the checked scalar address dialect".into(),
        });
    }
    elementwise_vector_plan(root).map(|(_, vector)| vector)
}

fn elementwise_vector_plan(root: &UOp) -> Result<(crate::LinearKernel, VectorPlan), JitError> {
    let linear = linear_kernel(root)?;
    linear
        .validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    let vector = VectorPlan {
        lanes: linear.lanes,
        enabled: linear.enabled,
        reason: linear.reason.clone(),
    };
    Ok((linear, vector))
}

fn linear_kernel(root: &UOp) -> Result<crate::LinearKernel, JitError> {
    #[cfg(test)]
    POLICY_LINEAR_DERIVATIONS.with(|count| count.set(count.get().saturating_add(1)));
    crate::LinearKernel::from_uop(root).map_err(|error| JitError::Unsupported(error.to_string()))
}

fn vector_program(linear: &crate::LinearKernel) -> Result<crate::VectorProgram, JitError> {
    let memory_spaces = crate::MemorySpacePlan::from_linear(linear)
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    crate::VectorProgram::from_linear(linear, &memory_spaces)
        .map_err(|error| JitError::Unsupported(error.to_string()))
}

#[cfg(test)]
fn native_rendered_cache_key(
    root: &UOp,
    request_vector: bool,
    source: &str,
) -> Result<String, JitError> {
    let cache = match dedicated_cache_policy(root) {
        Some(cache) => cache,
        None => ElementwiseTopology::new(root, request_vector)?
            .cache_policy_without_linear_validation(root)?,
    };
    Ok(cache.key(source))
}

fn dedicated_cache_policy(root: &UOp) -> Option<NativeCachePolicy> {
    Some(match root.operation() {
        Operation::PrefixScan(_) => NativeCachePolicy::Ordinary("prefix-scan".into()),
        Operation::Random(_) => NativeCachePolicy::Ordinary("random".into()),
        Operation::Threefry(_) => NativeCachePolicy::Ordinary(THREEFRY_RENDERER_VERSION.into()),
        Operation::Matmul(MatmulValue::Quantized(plan)) => {
            NativeCachePolicy::Ordinary(plan.cache_key.to_string())
        }
        Operation::Matmul(MatmulValue::Serial(plan)) => {
            NativeCachePolicy::Ordinary(plan.cache_key.to_string())
        }
        Operation::Matmul(MatmulValue::Tiled(payload)) => {
            NativeCachePolicy::Ordinary(payload.matmul.cache_key.to_string())
        }
        Operation::Matmul(MatmulValue::TensorCore(payload)) => {
            NativeCachePolicy::Ordinary(payload.matmul.cache_key.to_string())
        }
        Operation::Movement(MovementValue::QuantizedRowGather(plan)) => {
            NativeCachePolicy::Ordinary(plan.cache_key.to_string())
        }
        Operation::Movement(MovementValue::Plan(plan)) => {
            NativeCachePolicy::Ordinary(format!("movement-{}", plan.cache_key))
        }
        Operation::Conv2d(plan) => NativeCachePolicy::Conv2d(plan.cache_key),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Graph, Shape};

    fn legacy_vector_plan(root: &UOp) -> Result<VectorPlan, JitError> {
        if matches!(
            root.operation(),
            Operation::Matmul(_)
                | Operation::Conv2d(_)
                | Operation::Movement(_)
                | Operation::Random(_)
                | Operation::PrefixScan(_)
                | Operation::Threefry(_)
        ) {
            return Ok(VectorPlan {
                lanes: 1,
                enabled: false,
                reason: if matches!(root.operation(), Operation::PrefixScan(_)) {
                    "prefix scan uses a serial axis loop"
                } else if matches!(root.operation(), Operation::Threefry(_)) {
                    "live Threefry uses a dedicated scalar kernel"
                } else {
                    "static contraction uses scalar lanes"
                }
                .into(),
            });
        }
        if root
            .topological()
            .map_err(|error| JitError::Unsupported(error.to_string()))?
            .iter()
            .any(crate::projected_index::ProjectedIndexPlan::is_projected)
        {
            return Ok(VectorPlan {
                lanes: 1,
                enabled: false,
                reason: "projected indices use the checked scalar address dialect".into(),
            });
        }
        let linear = crate::LinearKernel::from_uop(root)
            .map_err(|error| JitError::Unsupported(error.to_string()))?;
        linear
            .validate()
            .map_err(|error| JitError::Unsupported(error.to_string()))?;
        Ok(VectorPlan {
            lanes: linear.lanes,
            enabled: linear.enabled,
            reason: linear.reason,
        })
    }

    fn legacy_cache_key(
        root: &UOp,
        request_vector: bool,
        source: &str,
    ) -> Result<String, JitError> {
        let discriminator = match root.operation() {
            Operation::PrefixScan(_) => "prefix-scan".to_owned(),
            Operation::Random(_) => "random".to_owned(),
            Operation::Threefry(_) => THREEFRY_RENDERER_VERSION.to_owned(),
            Operation::Matmul(MatmulValue::Quantized(plan)) => plan.cache_key.to_string(),
            Operation::Matmul(MatmulValue::Serial(plan)) => plan.cache_key.to_string(),
            Operation::Matmul(MatmulValue::Tiled(payload)) => payload.matmul.cache_key.to_string(),
            Operation::Matmul(MatmulValue::TensorCore(payload)) => {
                payload.matmul.cache_key.to_string()
            }
            Operation::Movement(MovementValue::QuantizedRowGather(plan)) => {
                plan.cache_key.to_string()
            }
            Operation::Movement(MovementValue::Plan(plan)) => {
                format!("movement-{}", plan.cache_key)
            }
            Operation::Conv2d(plan) => {
                return Ok(super::super::key(
                    &(RENDERER_VERSION.to_owned()
                        + std::env::consts::ARCH
                        + std::env::consts::OS
                        + &plan.cache_key.to_string()
                        + source),
                ));
            }
            _ => {
                let nodes = root
                    .topological()
                    .map_err(|error| JitError::Unsupported(error.to_string()))?;
                let request_vector = request_vector
                    && !nodes
                        .iter()
                        .any(crate::projected_index::ProjectedIndexPlan::is_projected);
                if request_vector {
                    let linear = crate::LinearKernel::from_uop(root)
                        .map_err(|error| JitError::Unsupported(error.to_string()))?;
                    let memory_spaces = crate::MemorySpacePlan::from_linear(&linear)
                        .map_err(|error| JitError::Unsupported(error.to_string()))?;
                    let vector = crate::VectorProgram::from_linear(&linear, &memory_spaces)
                        .map_err(|error| JitError::Unsupported(error.to_string()))?;
                    if vector.b1_eligibility().is_ok() {
                        format!("b1-{}", vector.cache_key)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            }
        };
        Ok(native_cache_key(&discriminator, source))
    }

    fn scheduled_kernel(graph: &Graph, output: crate::NodeId) -> UOp {
        crate::schedule(graph, output)
            .unwrap()
            .items
            .into_iter()
            .find(|item| item.node == output)
            .unwrap()
            .kernel
    }

    #[test]
    fn shared_policy_matches_independent_legacy_vector_and_cache_oracles() {
        let mut f32_graph = Graph::new();
        let f32_input = f32_graph.input_dtype("f32", Shape::from([8]), DType::F32);
        let f32_output = f32_graph.neg(f32_input).unwrap();
        let f32 = crate::lower_graph_elementwise(&f32_graph, f32_output).unwrap();

        let mut narrow_graph = Graph::new();
        let f16_input = narrow_graph.input_dtype("f16", Shape::from([5]), DType::F16);
        let bf16_input = narrow_graph.input_dtype("bf16", Shape::from([5]), DType::BF16);
        let f16_output = narrow_graph.sin(f16_input).unwrap();
        let bf16_output = narrow_graph.sin(bf16_input).unwrap();
        let f16 = crate::lower_graph_elementwise(&narrow_graph, f16_output).unwrap();
        let bf16 = crate::lower_graph_elementwise(&narrow_graph, bf16_output).unwrap();

        let mut projected_graph = Graph::new();
        let projected_input = projected_graph.input_dtype("projected", [2, 3, 4], DType::F32);
        let projected_owner = projected_graph.square(projected_input).unwrap();
        let projected_view = projected_graph.permute(projected_owner, [0, 2, 1]).unwrap();
        let projected_view = projected_graph.reshape(projected_view, [2, 3, 4]).unwrap();
        let projected_output = projected_graph.relu(projected_view).unwrap();
        let projected = scheduled_kernel(&projected_graph, projected_output);

        let mut matmul_graph = Graph::new();
        let lhs = matmul_graph.input_dtype("lhs", [2, 3], DType::F32);
        let rhs = matmul_graph.input_dtype("rhs", [3, 4], DType::F32);
        let matmul_output = matmul_graph.matmul(lhs, rhs).unwrap();
        let matmul = scheduled_kernel(&matmul_graph, matmul_output);

        let mut movement_graph = Graph::new();
        let movement_input = movement_graph.input_dtype("movement", [2, 3], DType::F32);
        let movement_view = movement_graph.permute(movement_input, [1, 0]).unwrap();
        let movement_output = movement_graph.contiguous(movement_view).unwrap();
        let movement = scheduled_kernel(&movement_graph, movement_output);

        let b1_eligibility = |root: &UOp| {
            let linear = crate::LinearKernel::from_uop(root).unwrap();
            let memory_spaces = crate::MemorySpacePlan::from_linear(&linear).unwrap();
            crate::VectorProgram::from_linear(&linear, &memory_spaces)
                .unwrap()
                .b1_eligibility()
        };
        assert!(b1_eligibility(&f32).is_ok(), "F32 fixture must exercise B1");
        assert!(
            b1_eligibility(&f16).is_err(),
            "F16 fixture must exercise the narrow scalar fallback"
        );
        assert!(
            b1_eligibility(&bf16).is_err(),
            "BF16 fixture must exercise the narrow scalar fallback"
        );
        assert!(
            projected
                .topological()
                .unwrap()
                .iter()
                .any(crate::projected_index::ProjectedIndexPlan::is_projected),
            "projected fixture must retain a projected-index node"
        );
        assert!(matches!(matmul.operation(), Operation::Matmul(_)));
        assert!(matches!(movement.operation(), Operation::Movement(_)));

        for (case, root) in [
            ("f32-b1", f32.clone()),
            ("f16-narrow", f16),
            ("bf16-narrow", bf16),
            ("projected", projected),
            ("matmul", matmul),
            ("movement", movement),
        ] {
            let rendered = crate::CpuJit::render_vectorized(&root).unwrap();
            let policy = NativeRenderAuthentication::new(&root, true).unwrap();
            assert_eq!(
                policy.vector(),
                &legacy_vector_plan(&root).unwrap(),
                "{case}"
            );
            assert_eq!(
                policy.cache_key(&rendered.source),
                legacy_cache_key(&root, true, &rendered.source).unwrap(),
                "{case}"
            );
            assert_eq!(
                policy.cache_key(&rendered.source),
                rendered.cache_key,
                "{case}"
            );
        }

        let scalar = crate::CpuJit::render(&f32).unwrap();
        assert_eq!(
            native_rendered_cache_key(&f32, false, &scalar.source).unwrap(),
            legacy_cache_key(&f32, false, &scalar.source).unwrap()
        );
    }

    #[test]
    fn combined_capsule_policy_derives_one_linear_and_preserves_error_order() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [8], DType::F32);
        let output = graph.neg(input).unwrap();
        let root = crate::lower_graph_elementwise(&graph, output).unwrap();
        POLICY_LINEAR_DERIVATIONS.with(|count| count.set(0));
        NativeRenderAuthentication::new(&root, true).unwrap();
        assert_eq!(POLICY_LINEAR_DERIVATIONS.with(|count| count.get()), 1);

        let malformed = UOp::sink(vec![]);
        assert_eq!(
            native_rendered_cache_key(&malformed, false, "malformed-source").unwrap(),
            legacy_cache_key(&malformed, false, "malformed-source").unwrap(),
            "cache-only scalar identity remains independent of renderer admission"
        );
        assert_eq!(
            crate::CpuJit::render_vectorized(&malformed).unwrap_err(),
            JitError::Unsupported("Sink without Store".into())
        );

        let store = root
            .sources()
            .iter()
            .find(|node| matches!(node.operation(), Operation::Store))
            .unwrap()
            .clone();
        let load = root
            .topological()
            .unwrap()
            .into_iter()
            .find(|node| matches!(node.operation(), Operation::Load))
            .unwrap();
        let invalid_load = UOp::from_operation(Operation::Load, load.ty(), Vec::new());
        let invalid_abi_and_policy = UOp::sink(vec![store, invalid_load]);
        assert!(ElementwiseRenderPolicy::new(&invalid_abi_and_policy, true).is_err());
        assert_eq!(
            crate::CpuJit::render_vectorized(&invalid_abi_and_policy).unwrap_err(),
            JitError::Unsupported("load without index".into()),
            "ABI admission must retain precedence over later vector-policy analysis"
        );
    }
}
