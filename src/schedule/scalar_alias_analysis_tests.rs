use super::*;
use crate::nn::StateKind;
use crate::rangeify::{ProjectionSource, RangeifyError};
use crate::{
    Backend, CompiledAdamWConfig, CompiledAdamWGraph, CompiledAdamWIgnoreIndexContext,
    CompiledDropoutConfig, CompiledDropoutKey, CompiledInputBatch, CompiledInputSpec,
    CompiledModuleAdamWPlan, CompiledMultiStepLr, DType, LossOptions, Module, Operation, Parameter,
    Reduction, Scalar, TensorData, TrainingDropoutProvider, UType,
    sparse_categorical_cross_entropy,
};

const BATCH: usize = 4;
const TIME: usize = 8;
const VOCAB: usize = 16;
const EMBEDDING: usize = 8;
const HEADS: usize = 2;
const FEED_FORWARD: usize = 32;
const ACCUMULATION_STEPS: u64 = 2;
const IGNORE_INDEX: i32 = -100;
const TARGETS: &str = "targets";
const TOKENS: &str = "tokens";
const FROZEN_POSITION_WEIGHT: &str = "positions.weight";
const ATTENTION_KEEP_MASK_SHAPE: [usize; 4] = [BATCH, 1, 1, TIME];
const SCALE_SEED: u64 = 0x5ca1_e000;

#[path = "../../examples/support/causal_transformer.rs"]
mod scale_model;
use scale_model::CausalTransformer as ScaleTransformer;

#[derive(Clone)]
struct SharedScaleTransformer(std::rc::Rc<ScaleTransformer>);

impl Module for SharedScaleTransformer {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.0.visit(prefix, visitor);
    }
}

struct ScaleBatch {
    tokens: TensorData,
    targets: TensorData,
}

impl ScaleBatch {
    const SCHEMA: [CompiledInputSpec; 2] = [
        CompiledInputSpec::new(TARGETS, &[BATCH, TIME], DType::I32),
        CompiledInputSpec::host_token(TOKENS, &[BATCH, TIME]),
    ];
}

impl CompiledInputBatch for ScaleBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &Self::SCHEMA
    }

    fn into_compiled_inputs(self) -> crate::Result<BTreeMap<String, TensorData>> {
        Ok(BTreeMap::from([
            (TARGETS.into(), self.targets),
            (TOKENS.into(), self.tokens),
        ]))
    }
}

fn scale_losses(graph: &mut Graph, logits: NodeId, targets: NodeId) -> crate::Result<NodeId> {
    sparse_categorical_cross_entropy(
        graph,
        logits,
        targets,
        LossOptions {
            reduction: Reduction::None,
            class_axis: 2,
            ignore_index: Some(i64::from(IGNORE_INDEX)),
            label_smoothing: 0.0,
        },
    )
}

fn build_scale_training(
    model: &SharedScaleTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
    dropout: &mut dyn TrainingDropoutProvider,
) -> crate::Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model
        .0
        .forward_training(graph, inputs[TOKENS], attention_keep_mask, dropout)?;
    Ok(CompiledAdamWGraph::token_mean(
        scale_losses(graph, logits, inputs[TARGETS])?,
        BTreeMap::new(),
    ))
}

fn build_scale_evaluation(
    model: &SharedScaleTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
) -> crate::Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model
        .0
        .forward_evaluation(graph, inputs[TOKENS], attention_keep_mask)?;
    Ok(CompiledAdamWGraph::token_mean(
        scale_losses(graph, logits, inputs[TARGETS])?,
        BTreeMap::new(),
    ))
}

fn scale_config() -> crate::Result<CompiledAdamWConfig> {
    Ok(CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_loss_scale(128.0)?
        .with_gradient_accumulation(ACCUMULATION_STEPS)?
        .with_max_gradient_norm(0.25)?
        .with_input_batch::<ScaleBatch>()?
        .with_token_weighted_ignore_index(TARGETS, IGNORE_INDEX)?
        .with_frozen_parameters([FROZEN_POSITION_WEIGHT])?
        .with_captured_multi_step_lr(CompiledMultiStepLr::new(2e-3, 0.5, [1])?)
        .with_clip_report()
        .with_window_loss_report())
}

fn affine(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let shared = graph.permute(producer, [1, 0]).unwrap();
    let first = graph.reshape(shared, [1, 3, 2]).unwrap();
    let second = graph.reshape(shared, [3, 2, 1]).unwrap();
    let second = graph.permute(second, [2, 0, 1]).unwrap();
    graph.add(first, second).unwrap()
}

fn projected(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph.permute(producer, [0, 2, 1, 3]).unwrap();
    let view = graph.reshape(view, [1, 2, 4]).unwrap();
    let lhs = graph.relu(view).unwrap();
    let rhs = graph.square(view).unwrap();
    graph.add(lhs, rhs).unwrap()
}

fn padded(graph: &mut Graph, input: NodeId) -> NodeId {
    let pad = graph.pad(input, [(1, 2)], Scalar::F(0.0)).unwrap();
    let view = graph
        .reshape(pad, [1, graph.shape(pad).unwrap().dims()[0]])
        .unwrap();
    graph.square(view).unwrap()
}

fn reversed(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph
        .stride(
            producer,
            [
                crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
            ],
        )
        .unwrap();
    let lhs = graph.relu(view).unwrap();
    let rhs = graph.square(view).unwrap();
    graph.add(lhs, rhs).unwrap()
}

fn reduced(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph.permute(producer, [1, 0]).unwrap();
    graph.sum(view, 1).unwrap()
}

fn empty(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph.permute(producer, [1, 0]).unwrap();
    graph.square(view).unwrap()
}

fn compare_schedules(graph: &Graph, requested: &[NodeId], actual: &Schedule, legacy: &Schedule) {
    actual.validate().unwrap();
    legacy.validate().unwrap();
    assert_eq!(actual.items.len(), legacy.items.len());
    assert_eq!(actual.requested_passthroughs, legacy.requested_passthroughs);
    for (actual, legacy) in actual.items.iter().zip(&legacy.items) {
        assert_eq!(actual.node, legacy.node);
        assert_eq!(actual.kernel, legacy.kernel);
        assert_eq!(
            actual.kernel.topological().unwrap(),
            legacy.kernel.topological().unwrap()
        );
        assert_eq!(actual.cache_key, legacy.cache_key);
        assert_eq!(actual.input_bindings, legacy.input_bindings);
        assert_eq!(actual.inputs, legacy.inputs);
        assert_eq!(actual.dependencies, legacy.dependencies);
        assert_eq!(actual.outputs, legacy.outputs);
        assert_eq!(actual.boundary, legacy.boundary);
    }
    let capture = |schedule| crate::CapturedSchedule::capture(graph, schedule, requested).unwrap();
    assert_eq!(
        capture(actual).to_bytes().unwrap(),
        capture(legacy).to_bytes().unwrap()
    );
}

#[test]
fn fixed_point_reuse_matches_previous_discovery_and_cpu() {
    type Builder = fn(&mut Graph, NodeId) -> NodeId;
    let cases: &[(&str, &[usize], Builder)] = &[
        ("affine aliases", &[2, 3], affine),
        ("projected aliases", &[1, 2, 2, 2], projected),
        ("predicated pad", &[2], padded),
        ("empty source pad", &[0], padded),
        ("signed stride", &[2, 3], reversed),
        ("reduction iteration", &[2, 3], reduced),
        ("empty output", &[0, 3], empty),
    ];
    let mut exercised_candidate_reuse = false;
    let mut cached_forest = crate::rangeify::ProjectedForestCounts::default();
    let mut released_forest = crate::rangeify::ProjectedForestCounts::default();
    for &(name, shape, build) in cases {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::new(shape.to_vec()));
        let output = build(&mut graph, input);
        let ((cached, counts), cached_counts) =
            crate::rangeify::with_projected_forest_mode(true, || {
                with_mode(true, || schedule(&graph, output))
            });
        let ((legacy, previous), released_counts) =
            crate::rangeify::with_projected_forest_mode(false, || {
                with_mode(false, || schedule(&graph, output))
            });
        cached_forest.queries += cached_counts.queries;
        cached_forest.source_queries += cached_counts.source_queries;
        cached_forest.hits += cached_counts.hits;
        cached_forest.extractions += cached_counts.extractions;
        cached_forest.emissions += cached_counts.emissions;
        cached_forest.released_builds += cached_counts.released_builds;
        cached_forest.retention_bypasses += cached_counts.retention_bypasses;
        released_forest.queries += released_counts.queries;
        released_forest.source_queries += released_counts.source_queries;
        released_forest.hits += released_counts.hits;
        released_forest.extractions += released_counts.extractions;
        released_forest.emissions += released_counts.emissions;
        released_forest.released_builds += released_counts.released_builds;
        released_forest.retention_bypasses += released_counts.retention_bypasses;
        let cached = cached.unwrap();
        let legacy = legacy.unwrap();
        assert_eq!(
            counts.fixed_point_passes, previous.fixed_point_passes,
            "{name}"
        );
        assert!(
            counts.candidate_collections <= previous.candidate_collections,
            "{name}"
        );
        assert_eq!(previous.candidate_hits, 0, "{name}");
        if counts.candidate_hits > 0 {
            assert!(counts.fixed_point_passes >= 2, "{name}");
            assert!(
                counts.candidate_collections < previous.candidate_collections,
                "{name}"
            );
            exercised_candidate_reuse = true;
        }
        compare_schedules(&graph, &[output], &cached, &legacy);
        let data = TensorData::new(
            Shape::new(shape.to_vec()),
            (0..graph.shape(input).unwrap().numel().unwrap())
                .map(|i| (i as f32 - 2.0) * 0.5)
                .collect(),
        )
        .unwrap();
        let oracle = crate::CpuBackend
            .execute(
                &graph,
                output,
                &std::collections::HashMap::from([("input".into(), data.clone())]),
            )
            .unwrap();
        let inputs = BTreeMap::from([("input".into(), data)]);
        for schedule in [&cached, &legacy] {
            let capture = crate::CapturedSchedule::capture(&graph, schedule, &[output]).unwrap();
            let result = capture.replay(&inputs).unwrap();
            assert_eq!(result[0].shape(), oracle.shape(), "{name}");
            assert_eq!(result[0].storage(), oracle.storage(), "{name}");
        }
    }
    assert!(
        exercised_candidate_reuse,
        "no repeated candidate discovery was exercised"
    );
    assert_eq!(cached_forest.queries, released_forest.queries);
    // The released run also repeats candidate discovery at each fixed point.
    // Concrete lowering trials are unchanged, but cached discovery may avoid
    // its source-classification calls before forest lookup is even reached.
    assert!(cached_forest.source_queries <= released_forest.source_queries);
    assert!(cached_forest.hits > released_forest.hits);
    assert!(cached_forest.extractions > released_forest.extractions);
    assert!(cached_forest.emissions > released_forest.emissions);
    assert!(cached_forest.released_builds < released_forest.released_builds);
}

#[test]
fn projected_forest_schedule_capture_and_cpu_match_uncached_lowering() {
    let mut graph = Graph::new();
    let input = graph.input("input", [1, 2, 2, 2]);
    let output = projected(&mut graph, input);
    let (cached, cached_counts) =
        crate::rangeify::with_projected_forest_mode(true, || schedule(&graph, output));
    let (uncached, uncached_counts) =
        crate::rangeify::with_projected_forest_mode(false, || schedule(&graph, output));
    let cached = cached.unwrap();
    let uncached = uncached.unwrap();
    compare_schedules(&graph, &[output], &cached, &uncached);
    assert!(cached_counts.hits > uncached_counts.hits);
    assert!(cached_counts.extractions > uncached_counts.extractions);
    assert!(cached_counts.emissions > uncached_counts.emissions);
    assert!(cached_counts.released_builds < uncached_counts.released_builds);

    let data = TensorData::new(
        Shape::from([1, 2, 2, 2]),
        (0..8).map(|index| index as f32 - 3.0).collect(),
    )
    .unwrap();
    let oracle = crate::CpuBackend
        .execute(
            &graph,
            output,
            &std::collections::HashMap::from([("input".into(), data.clone())]),
        )
        .unwrap();
    let inputs = BTreeMap::from([("input".into(), data)]);
    for schedule in [&cached, &uncached] {
        let capture = crate::CapturedSchedule::capture(&graph, schedule, &[output]).unwrap();
        let replayed = capture.replay(&inputs).unwrap();
        assert_eq!(replayed[0].shape(), oracle.shape());
        assert_eq!(replayed[0].storage(), oracle.storage());
    }
}

#[test]
fn maintained_transformer_scale_compile_preserves_full_artifact_across_cache_modes() {
    let model = std::rc::Rc::new(ScaleTransformer::new(SCALE_SEED).unwrap());
    let compile = || {
        CompiledModuleAdamWPlan::compile_graph_with_dropout_and_ignore_index(
            scale_config().unwrap(),
            CompiledDropoutConfig::new(CompiledDropoutKey([0x5ca1_e001, 0x5ca1_e002])),
            SharedScaleTransformer(std::rc::Rc::clone(&model)),
            build_scale_training,
        )
        .map_err(|error| error.into_parts().1)
        .unwrap()
        .with_evaluation_graph_and_ignore_index(build_scale_evaluation)
        .map_err(|error| error.into_parts().1)
        .unwrap()
    };

    let (cached, cached_counts) = crate::rangeify::with_projected_forest_mode(true, &compile);
    let (uncached, uncached_counts) = crate::rangeify::with_projected_forest_mode(false, &compile);

    let inspect = |plan: &CompiledModuleAdamWPlan<SharedScaleTransformer>| {
        let inspection = plan.inspection().unwrap();
        let phases = inspection
            .compile_phases()
            .expect("fresh scale compile must retain phase observations");
        assert_eq!(phases.compile_count(), 1);
        assert!(phases.accumulation_capture().is_some());
        assert!(phases.partial_flush().is_some());
        assert!(phases.zero_grad().is_some());
        assert!(phases.evaluation().is_some());
        assert_eq!(inspection.main().0, plan.capture_identity());
        assert!(inspection.accumulation().is_some());
        assert!(inspection.partial_flush().is_some());
        assert!(inspection.zero_grad().is_some());
        assert!(inspection.evaluation().is_some());
        assert_eq!(inspection.recurrent_state_count(), 145);
        assert_eq!(inspection.recurrent_state_bytes(), 30_244);
        assert_eq!(inspection.main().1.schedule_item_count, 785);
        let roles = [
            Some(inspection.main()),
            inspection.accumulation(),
            inspection.partial_flush(),
            inspection.zero_grad(),
            inspection.evaluation(),
        ];
        assert_eq!(roles.into_iter().flatten().count(), 5);
        (
            plan.capture_identity(),
            plan.evaluation_capture_identity()
                .expect("scale evaluation capture must be attached"),
        )
    };
    let cached_identities = inspect(&cached);
    let uncached_identities = inspect(&uncached);
    assert_eq!(cached_identities, uncached_identities);
    assert_eq!(
        cached_identities,
        (12_721_483_959_162_265_181, 11_955_252_298_382_724_919)
    );
    assert_ne!(cached_identities.0, cached_identities.1);
    let cached_artifact = cached.program_artifact().unwrap();
    let uncached_artifact = uncached.program_artifact().unwrap();
    assert_eq!(cached_artifact.as_bytes(), uncached_artifact.as_bytes());
    assert!(cached_counts.hits > uncached_counts.hits);
    assert!(cached_counts.emissions > uncached_counts.emissions);

    let report = |mode: &str,
                  counts: crate::rangeify::ProjectedForestCounts,
                  identities: (u64, u64),
                  artifact_bytes: usize| {
        println!(
            "{{\"mode\":\"{mode}\",\"queries\":{},\"source_queries\":{},\"hits\":{},\"extractions\":{},\"emissions\":{},\"released_builds\":{},\"retention_bypasses\":{},\"capture_roles\":5,\"main_capture_identity\":{},\"evaluation_capture_identity\":{},\"artifact_bytes\":{},\"artifact_bytes_equal\":true}}",
            counts.queries,
            counts.source_queries,
            counts.hits,
            counts.extractions,
            counts.emissions,
            counts.released_builds,
            counts.retention_bypasses,
            identities.0,
            identities.1,
            artifact_bytes,
        );
    };
    report(
        "cached",
        cached_counts,
        cached_identities,
        cached_artifact.as_bytes().len(),
    );
    report(
        "uncached",
        uncached_counts,
        uncached_identities,
        uncached_artifact.as_bytes().len(),
    );
}

#[test]
fn queries_preserve_guard_classification_full_shapes_and_exact_errors() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let alias = graph.reshape(input, [1, 2, 3]).unwrap();
    with_mode(true, || {
        let mut queries = crate::rangeify::ProjectedForestCache::new(&graph);
        let shape = Shape::from([1, 2, 3]);
        // A guarded-query miss is not a failed projection. Its later
        // unconditional lookup must find the source, not reuse that miss.
        assert_eq!(
            queries.predicated_source(&graph, alias, &shape),
            Err(RangeifyError::Unsupported(alias))
        );
        assert_eq!(queries.projected_source(&graph, alias, &shape), Ok(input));
        assert_eq!(
            queries.projected_source(&graph, alias, &Shape::from([2, 3, 1])),
            Err(RangeifyError::Invalid)
        );
        assert_eq!(
            queries.projected_source(&graph, alias, &Shape::from([2, 3, 1])),
            Err(RangeifyError::Invalid)
        );
        for bad in [
            NodeId::from_index(graph.node_count()),
            NodeId::from_index(usize::MAX),
        ] {
            assert_eq!(
                queries.projected_source(&graph, bad, &shape),
                Err(RangeifyError::Invalid)
            );
            assert_eq!(
                queries.predicated_source(&graph, bad, &shape),
                Err(RangeifyError::Invalid)
            );
        }
    });

    for length in [0, 2] {
        let mut graph = Graph::new();
        let input = graph.input("input", [length]);
        let pad = graph.pad(input, [(1, 2)], Scalar::F(0.0)).unwrap();
        let shape = graph.shape(pad).unwrap();
        let mut queries = crate::rangeify::ProjectedForestCache::new(&graph);
        assert_eq!(queries.predicated_source(&graph, pad, shape), Ok(input));
        assert_eq!(queries.projected_source(&graph, pad, shape), Ok(input));
        assert_eq!(
            queries.classification(&graph, pad, shape),
            Ok(ProjectionSource::Predicated(input))
        );
    }
}

#[test]
fn candidate_discovery_is_scoped_to_its_borrowed_frontier() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let producer = graph.square(input).unwrap();
    let alias = graph.permute(producer, [1, 0]).unwrap();
    let output = graph.square(alias).unwrap();
    let roots = BTreeSet::from([producer.index(), alias.index(), output.index()]);
    let none = BTreeSet::new();
    let external = BTreeSet::from([alias.index()]);
    let requested = BTreeSet::from([alias.index(), output.index()]);
    let mut projections = crate::rangeify::ProjectedForestCache::new(&graph);
    let mut internal = ScalarAliasAnalysis::new(&graph, &roots, &none, &none);
    assert!(
        internal
            .candidates(output, &mut projections)
            .unwrap()
            .unwrap()
            .affine_maps
            .contains_key(&producer.index())
    );
    for (external, requested) in [(&external, &none), (&none, &requested)] {
        let mut observed = ScalarAliasAnalysis::new(&graph, &roots, external, requested);
        let candidates = observed
            .candidates(output, &mut projections)
            .unwrap()
            .unwrap();
        assert!(candidates.affine_maps.is_empty());
        assert!(candidates.projected_view_roots.is_empty());
        assert_eq!(candidates.direct_roots, BTreeSet::from([alias.index()]));
    }
    assert!(
        internal
            .candidates(output, &mut projections)
            .unwrap()
            .unwrap()
            .affine_maps
            .contains_key(&producer.index())
    );
    let unknown = NodeId::from_index(graph.node_count());
    assert!(
        matches!(internal.candidates(unknown, &mut projections), Err(ScheduleError::Graph(crate::Error::UnknownNode(n))) if n == unknown)
    );
    assert!(!internal.candidates.contains_key(&unknown));
}

#[test]
fn observed_frontiers_match_previous_discovery() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let producer = graph.square(input).unwrap();
    let alias = graph.permute(producer, [1, 0]).unwrap();
    let lhs = graph.square(alias).unwrap();
    let rhs = graph.neg(alias).unwrap();
    let output = graph.add(lhs, rhs).unwrap();
    for (requested, external) in [
        (vec![alias, output], vec![]),
        (vec![output], vec![producer]),
        (vec![output], vec![alias]),
    ] {
        let (cached, _) = with_mode(true, || {
            schedule_with_external_materializations(&graph, &requested, &external)
        });
        let (legacy, _) = with_mode(false, || {
            schedule_with_external_materializations(&graph, &requested, &external)
        });
        compare_schedules(&graph, &requested, &cached.unwrap(), &legacy.unwrap());
    }
}

#[test]
fn source_discovery_does_not_admit_a_concrete_range_or_address() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let alias = graph.permute(input, [1, 0]).unwrap();
    let shape = Shape::from([3, 2]);
    let mut queries = crate::rangeify::ProjectedForestCache::new(&graph);
    assert_eq!(queries.projected_source(&graph, alias, &shape), Ok(input));
    let integer = UType::scalar(DType::I64);
    assert_eq!(
        crate::rangeify::projected_view(&graph, alias, &shape, &UOp::constant(6, integer)),
        Err(RangeifyError::Invalid)
    );

    let range = UOp::from_operation(
        Operation::Range(0),
        Some(integer),
        vec![UOp::constant(6, integer)],
    );
    let projected = crate::rangeify::projected_view(&graph, alias, &shape, &range).unwrap();
    let value = UType::scalar(DType::F32);
    let address = UOp::from_operation(
        Operation::DefineGlobal(crate::uop::AddressValue {
            space: crate::AddressSpace::Global,
            name: "source".into(),
            element: value,
        }),
        Some(value),
        vec![],
    );
    let invalid = UOp::from_operation(
        Operation::Index(crate::IndexValue::Buffer {
            buffer: input.index() as u64,
            elements: 6,
            input_shape: Shape::from([2, 3]),
            output_shape: shape.clone(),
            addressing: crate::IndexAddressing::Projected,
        }),
        Some(value),
        vec![
            address,
            UOp::binary(
                crate::uop::Binary::Add,
                projected.expression,
                UOp::constant(6, integer),
            ),
        ],
    );
    assert!(matches!(
        crate::projected_index::ProjectedIndexPlan::from_index(&invalid),
        Err(UOpError::InvalidIndex)
    ));
    assert_eq!(queries.projected_source(&graph, alias, &shape), Ok(input));
}

#[test]
fn scheduling_failures_keep_the_previous_error_contract() {
    let graph = Graph::new();
    let unknown = NodeId::from_index(0);
    let (cached, _) = with_mode(true, || schedule(&graph, unknown));
    let (legacy, _) = with_mode(false, || schedule(&graph, unknown));
    let expected = ScheduleError::Graph(crate::Error::UnknownNode(unknown));
    assert_eq!(cached.unwrap_err(), expected);
    assert_eq!(legacy.unwrap_err(), expected);

    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [usize::MAX, 2], DType::F32);
    // An external input alone deliberately produces no scheduled item and
    // does not enter kernel extent validation. Preserve that contract too.
    for reuse in [true, false] {
        let (passthrough, counts) = with_mode(reuse, || schedule(&graph, input));
        assert!(passthrough.unwrap().items.is_empty());
        assert_eq!(counts.candidate_collections, 0);
    }
    // Inject a malformed producer directly: the public graph constructor
    // would reject its overflowing descriptor before scheduling can observe it.
    let output = graph.push(
        Op::Unary {
            op: crate::UnaryOp::Neg,
            input,
        },
        Shape::from([usize::MAX, 2]),
        DType::F32,
    );
    for reuse in [true, false] {
        let (result, counts) = with_mode(reuse, || schedule(&graph, output));
        assert_eq!(
            result.unwrap_err(),
            ScheduleError::UOp(UOpError::InvalidArgument)
        );
        assert!(counts.candidate_collections > 0);
    }
}
