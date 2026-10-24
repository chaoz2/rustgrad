use super::ordering_tests::ReferenceKey;
use super::*;
use std::cell::Cell;

thread_local! {
    static LEGACY: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn legacy_enabled() -> bool {
    LEGACY.get()
}

fn with_legacy<T>(legacy: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            LEGACY.set(self.0);
        }
    }
    let _reset = Reset(LEGACY.replace(legacy));
    run()
}

// Independent released driver: ReferenceKey recursively compares every node
// field and never uses the production shared-allocation ordering shortcut.
pub(super) fn recursive_driver<C, R: RewriteRuleSpec<C>>(
    root: &UOp,
    rules: &mut [R],
    walk: Walk,
    context: &mut C,
) -> Result<(UOp, RewriteTrace), UOpError> {
    const STEP_LIMIT: usize = 128;

    rules.sort_by_key(|rule| rule.priority());
    let rules = rules
        .iter()
        .map(|rule| PreparedRewriteRule {
            direct_source_requirements: rule.pattern().direct_source_requirements(),
            rule,
        })
        .collect::<Vec<_>>();
    let mut trace = RewriteTrace { rules: vec![] };
    let mut memo = BTreeMap::new();
    let mut active = BTreeSet::new();
    fn go<C, R: RewriteRuleSpec<C>>(
        n: &UOp,
        r: &[PreparedRewriteRule<'_, R>],
        w: Walk,
        m: &mut BTreeMap<ReferenceKey, UOp>,
        active: &mut BTreeSet<ReferenceKey>,
        t: &mut RewriteTrace,
        context: &mut C,
    ) -> Result<UOp, UOpError> {
        let key = ReferenceKey(n.clone());
        if let Some(x) = m.get(&key) {
            return Ok(x.clone());
        }
        if !active.insert(key.clone()) {
            return Err(UOpError::RewriteCycle);
        }

        let mut x = n.clone();
        let mut seen = BTreeSet::new();
        for _ in 0..STEP_LIMIT {
            if !seen.insert(ReferenceKey(x.clone())) {
                active.remove(&key);
                return Err(UOpError::RewriteCycle);
            }
            let before = x.clone();
            if w == Walk::BottomUp {
                let sources = x
                    .sources()
                    .iter()
                    .map(|source| go(source, r, w, m, active, t, context))
                    .collect::<Result<Vec<_>, _>>()?;
                if sources.as_slice() != x.sources() {
                    x = x.replace_sources(sources);
                }
            }

            for rule in r {
                if !rule.admits(&x) {
                    continue;
                }
                let mut next = None;
                rule.rule
                    .pattern()
                    .visit_matches(&x, &Captures::default(), &mut |captures| match rule
                        .rule
                        .apply(context, &captures, &x)
                    {
                        Some(candidate) => {
                            if candidate != x {
                                next = Some(candidate);
                            }
                            true
                        }
                        None => false,
                    });
                if let Some(next) = next {
                    if !x.is_pure() || !next.is_pure() {
                        active.remove(&key);
                        return Err(UOpError::EffectRewrite);
                    }
                    t.rules.push(rule.rule.name());
                    x = next;
                    break;
                }
            }

            if w == Walk::TopDown {
                let sources = x
                    .sources()
                    .iter()
                    .map(|source| go(source, r, w, m, active, t, context))
                    .collect::<Result<Vec<_>, _>>()?;
                if sources.as_slice() != x.sources() {
                    x = x.replace_sources(sources);
                }
            }

            if x == before {
                active.remove(&key);
                m.insert(key, x.clone());
                return Ok(x);
            }
        }
        active.remove(&key);
        Err(UOpError::RewriteStepLimit)
    }
    let x = go(
        root,
        &rules,
        walk,
        &mut memo,
        &mut active,
        &mut trace,
        context,
    )?;
    Ok((x, trace))
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Context {
    calls: Vec<String>,
    replace: bool,
}

fn special(name: &str) -> UOp {
    UOp::from_operation(Operation::Special(name.into()), None, vec![])
}

fn literal(value: i64) -> UOp {
    UOp::constant(value, UType::scalar(crate::DType::I32))
}

fn integer(operation: &Operation) -> bool {
    matches!(operation, Operation::Const(LiteralValue::Int(_)))
}

fn integer_value(node: &UOp) -> i64 {
    let Operation::Const(LiteralValue::Int(value)) = node.operation() else {
        panic!("expected integer literal")
    };
    *value
}

fn rule(
    name: &'static str,
    priority: i32,
    pattern: UPat,
    apply: ContextRewriteFn<Context>,
) -> ContextRewriteRule<Context> {
    ContextRewriteRule {
        name,
        priority,
        pattern,
        apply,
    }
}

fn compare_drivers(
    root: &UOp,
    rules: &[ContextRewriteRule<Context>],
    walk: Walk,
    context: Context,
) -> (Result<UOp, UOpError>, Vec<&'static str>, Context) {
    let run = |legacy| {
        let mut rules = rules.to_vec();
        let mut context = context.clone();
        let result = with_legacy(legacy, || {
            rewrite_with_context(root, &mut rules, walk, &mut context)
        });
        match result {
            Ok((node, trace)) => (Ok(node), trace.rules, context),
            Err(error) => (Err(error), vec![], context),
        }
    };
    let actual = run(false);
    let expected = run(true);
    assert_eq!(actual, expected, "walk={walk:?}");
    if let (Ok(actual), Ok(expected)) = (&actual.0, &expected.0) {
        assert_eq!(actual.validate(), expected.validate());
        assert_eq!(artifact::encode(actual), artifact::encode(expected));
        assert_eq!(actual.topological(), expected.topological());
        // Equality alone cannot observe allocation sharing. Compare the full
        // source-occurrence sharing matrix and retained input allocations.
        fn occurrences(node: &UOp, out: &mut Vec<UOp>) {
            out.push(node.clone());
            for source in node.sources() {
                occurrences(source, out);
            }
        }
        let mut actual_nodes = vec![];
        let mut expected_nodes = vec![];
        let mut original_nodes = vec![];
        occurrences(actual, &mut actual_nodes);
        occurrences(expected, &mut expected_nodes);
        occurrences(root, &mut original_nodes);
        assert_eq!(actual_nodes.len(), expected_nodes.len());
        for (a, e) in actual_nodes.iter().zip(&expected_nodes) {
            for (other_a, other_e) in actual_nodes.iter().zip(&expected_nodes) {
                assert_eq!(a.shares_node_with(other_a), e.shares_node_with(other_e));
            }
            for original in &original_nodes {
                assert_eq!(a.shares_node_with(original), e.shares_node_with(original));
            }
        }
    }
    actual
}

fn bind_or_equal(context: &mut Context, _: &Captures, node: &UOp) -> Option<UOp> {
    context.calls.push("unbound".into());
    Some(if context.replace {
        special("bound")
    } else {
        UOp::from_operation(node.operation().clone(), node.ty(), node.sources().to_vec())
            .retag(node.tag().cloned())
    })
}

#[test]
fn rewrite_preserves_structural_memo_and_actual_source_allocations() {
    let rules = [rule(
        "bind",
        0,
        UPat::op(Operation::Special("unbound".into())),
        bind_or_equal,
    )];
    for walk in [Walk::BottomUp, Walk::TopDown] {
        for shared in [false, true] {
            let first = special("unbound");
            let second = if shared {
                first.clone()
            } else {
                special("unbound")
            };
            let root = UOp::sink(vec![first, second]);
            for replace in [false, true] {
                let (result, trace, context) = compare_drivers(
                    &root,
                    &rules,
                    walk,
                    Context {
                        replace,
                        ..Context::default()
                    },
                );
                let output = result.unwrap();
                assert_eq!(context.calls, ["unbound"]);
                assert_eq!(trace, if replace { vec!["bind"] } else { vec![] });
                assert_eq!(output.shares_node_with(&root), !replace);
                assert_eq!(
                    output.sources()[0].shares_node_with(&output.sources()[1]),
                    replace || shared
                );
            }
        }
    }
}

fn decline_then_select(context: &mut Context, captures: &Captures, _: &UOp) -> Option<UOp> {
    assert!(captures.get("leaked").is_none());
    if captures.get("first").is_some() {
        context.calls.push("decline".into());
        None
    } else {
        context.calls.push("select".into());
        captures.get("selected").cloned()
    }
}

fn accept_equal(context: &mut Context, captures: &Captures, node: &UOp) -> Option<UOp> {
    assert!(captures.get("later").is_none());
    context.calls.push("equal".into());
    Some(UOp::from_operation(
        node.operation().clone(),
        node.ty(),
        node.sources().to_vec(),
    ))
}

fn select_rhs(context: &mut Context, _: &Captures, node: &UOp) -> Option<UOp> {
    context.calls.push("rhs".into());
    Some(node.sources()[1].clone())
}

#[test]
fn rewrite_preserves_alternatives_permutations_and_equal_acceptance() {
    let root = UOp::binary(Binary::Add, literal(1), literal(2));
    let pattern = UPat::any_of([
        UPat::op(Operation::Binary(Binary::Add)).sources(vec![
            UPat::any().named("leaked"),
            UPat::op(Operation::Const(LiteralValue::Int(99))),
        ]),
        UPat::op(Operation::Binary(Binary::Add))
            .sources(vec![UPat::any().named("first"), UPat::any()]),
        UPat::op(Operation::Binary(Binary::Add))
            .sources(vec![UPat::any(), UPat::any().named("selected")]),
    ])
    .unwrap();
    let permuted = UPat::op(Operation::Binary(Binary::Add)).sources_permuted(vec![
        UPat::any().named("selected"),
        UPat::op(Operation::Const(LiteralValue::Int(1))),
    ]);
    let equal = UPat::any_of([
        UPat::op(Operation::Binary(Binary::Add)).named("accepted"),
        UPat::op(Operation::Binary(Binary::Add)).named("later"),
    ])
    .unwrap();
    for walk in [Walk::BottomUp, Walk::TopDown] {
        for (pattern, expected_calls) in [
            (pattern.clone(), vec!["decline", "select"]),
            (permuted.clone(), vec!["select"]),
        ] {
            let (result, trace, context) = compare_drivers(
                &root,
                &[rule("choose", 0, pattern, decline_then_select)],
                walk,
                Context::default(),
            );
            assert_eq!(result.unwrap(), literal(2));
            assert_eq!(trace, ["choose"]);
            assert_eq!(context.calls, expected_calls);
        }
        // A fresh equal result stops this rule's alternatives, not later
        // rules. Sort priority, not declaration order, selects the equal rule.
        let rules = [
            rule(
                "rhs",
                9,
                UPat::op(Operation::Binary(Binary::Add)),
                select_rhs,
            ),
            rule("equal", 0, equal.clone(), accept_equal),
        ];
        let (result, trace, context) = compare_drivers(&root, &rules, walk, Context::default());
        assert_eq!(result.unwrap(), literal(2));
        assert_eq!(trace, ["rhs"]);
        assert_eq!(context.calls, ["equal", "rhs"]);
    }
}

fn toggle(context: &mut Context, _: &Captures, node: &UOp) -> Option<UOp> {
    context.calls.push(integer_value(node).to_string());
    Some(literal(1 - integer_value(node)))
}

fn increment(context: &mut Context, _: &Captures, node: &UOp) -> Option<UOp> {
    context.calls.push(integer_value(node).to_string());
    Some(literal(integer_value(node) + 1))
}

fn effect(context: &mut Context, _: &Captures, _: &UOp) -> Option<UOp> {
    context.calls.push("effect".into());
    Some(UOp::from_operation(Operation::Barrier, None, vec![]))
}

fn ancestor(context: &mut Context, _: &Captures, node: &UOp) -> Option<UOp> {
    context.calls.push("ancestor".into());
    Some(UOp::from_operation(
        Operation::Unary(Unary::Neg),
        node.ty(),
        vec![UOp::from_operation(
            node.operation().clone(),
            node.ty(),
            vec![],
        )],
    ))
}

#[test]
fn rewrite_preserves_error_contracts_and_callback_counts() {
    for walk in [Walk::BottomUp, Walk::TopDown] {
        for (name, callback, error, count) in [
            (
                "toggle",
                toggle as ContextRewriteFn<Context>,
                UOpError::RewriteCycle,
                2,
            ),
            (
                "increment",
                increment as ContextRewriteFn<Context>,
                UOpError::RewriteStepLimit,
                128,
            ),
            (
                "effect",
                effect as ContextRewriteFn<Context>,
                UOpError::EffectRewrite,
                1,
            ),
        ] {
            let (result, trace, context) = compare_drivers(
                &literal(0),
                &[rule(name, 0, UPat::operation_predicate(integer), callback)],
                walk,
                Context::default(),
            );
            assert_eq!(result, Err(error), "{name}/{walk:?}");
            assert!(trace.is_empty());
            assert_eq!(context.calls.len(), count, "{name}/{walk:?}");
        }
        let (result, _, context) = compare_drivers(
            &special("unbound"),
            &[rule(
                "ancestor",
                0,
                UPat::op(Operation::Special("unbound".into())),
                ancestor,
            )],
            walk,
            Context::default(),
        );
        assert_eq!(result, Err(UOpError::RewriteCycle));
        assert_eq!(context.calls, ["ancestor"]);
    }
}

type ReplacementOwners = Vec<std::sync::Weak<UOpNode>>;

fn probed(owners: &mut ReplacementOwners, node: UOp) -> UOp {
    owners.push(Arc::downgrade(&node.0));
    node
}

fn probed_bind(owners: &mut ReplacementOwners, _: &Captures, _: &UOp) -> Option<UOp> {
    Some(probed(owners, special("bound")))
}

fn probed_increment(owners: &mut ReplacementOwners, _: &Captures, node: &UOp) -> Option<UOp> {
    Some(probed(owners, literal(integer_value(node) + 1)))
}

fn probed_toggle(owners: &mut ReplacementOwners, _: &Captures, node: &UOp) -> Option<UOp> {
    Some(probed(owners, literal(1 - integer_value(node))))
}

fn probed_effect(owners: &mut ReplacementOwners, _: &Captures, _: &UOp) -> Option<UOp> {
    Some(probed(
        owners,
        UOp::from_operation(Operation::Barrier, None, vec![]),
    ))
}

#[test]
fn rewrite_errors_release_completed_siblings_and_dynamic_replacements() {
    for walk in [Walk::BottomUp, Walk::TopDown] {
        for (name, callback, expected, replacements) in [
            (
                "step limit",
                probed_increment as ContextRewriteFn<ReplacementOwners>,
                UOpError::RewriteStepLimit,
                129,
            ),
            ("cycle", probed_toggle, UOpError::RewriteCycle, 3),
            ("effect", probed_effect, UOpError::EffectRewrite, 2),
        ] {
            for legacy in [false, true] {
                // The first child completes and enters the structural memo;
                // the second child fails after creating dynamic replacements.
                let root = UOp::sink(vec![special("unbound"), literal(0)]);
                let mut rules = [
                    ContextRewriteRule {
                        name: "bind",
                        priority: 0,
                        pattern: UPat::op(Operation::Special("unbound".into())),
                        apply: probed_bind,
                    },
                    ContextRewriteRule {
                        name,
                        priority: 1,
                        pattern: UPat::operation_predicate(integer),
                        apply: callback,
                    },
                ];
                let mut owners = vec![];
                let error = with_legacy(legacy, || {
                    rewrite_with_context(&root, &mut rules, walk, &mut owners)
                })
                .unwrap_err();
                assert_eq!(error, expected, "{name}/{walk:?}/{legacy}");
                assert_eq!(owners.len(), replacements, "{name}/{walk:?}/{legacy}");
                assert!(
                    owners.iter().all(|owner| owner.upgrade().is_none()),
                    "{name}/{walk:?}/{legacy}: replacement owner survived unwind"
                );
            }
        }
    }
}

fn collapse(context: &mut Context, _: &Captures, _: &UOp) -> Option<UOp> {
    context.calls.push("collapse".into());
    Some(literal(99))
}

#[test]
fn rewrite_preserves_walk_error_precedence() {
    let root = UOp::binary(Binary::Add, literal(0), literal(1));
    let rules = [
        rule(
            "collapse",
            0,
            UPat::op(Operation::Binary(Binary::Add)),
            collapse,
        ),
        rule(
            "toggle",
            1,
            UPat::ops([
                Operation::Const(LiteralValue::Int(0)),
                Operation::Const(LiteralValue::Int(1)),
            ]),
            toggle,
        ),
    ];
    let (result, _, context) = compare_drivers(&root, &rules, Walk::BottomUp, Context::default());
    assert_eq!(result, Err(UOpError::RewriteCycle));
    assert_eq!(context.calls, ["0", "1"]);
    let (result, trace, context) =
        compare_drivers(&root, &rules, Walk::TopDown, Context::default());
    assert_eq!(result.unwrap(), literal(99));
    assert_eq!(trace, ["collapse"]);
    assert_eq!(context.calls, ["collapse"]);
}

#[test]
fn rewrite_keeps_reduction_ownership_and_rebuild_behavior() {
    let reduction = || {
        let ty = UType::scalar(crate::DType::F32);
        let init = UOp::from_operation(
            Operation::ReduceInit(ReductionValue {
                input_shape: crate::Shape::from([1]),
                output_shape: crate::Shape::new([]),
                axes: vec![0],
                keepdim: false,
                kind: crate::ReduceKind::Sum,
            }),
            Some(ty),
            vec![],
        );
        let source = UOp::scalar_constant(crate::DType::F32, 1.0f32.to_bits() as u64, ty);
        let update = UOp::from_operation(Operation::ReduceAccumulate, Some(ty), vec![init, source]);
        UOp::from_operation(Operation::ReduceFinalize, Some(ty), vec![update])
    };
    let root = UOp::sink(vec![reduction(), reduction()]);
    assert_eq!(root.validate(), Err(UOpError::InvalidArgument));
    for walk in [Walk::BottomUp, Walk::TopDown] {
        let (result, trace, _) = compare_drivers(&root, &[], walk, Context::default());
        let output = result.unwrap();
        assert!(output.shares_node_with(&root));
        assert!(!output.sources()[0].shares_node_with(&output.sources()[1]));
        assert_eq!(output.validate(), Err(UOpError::InvalidArgument));
        assert!(trace.is_empty());
        let valid = UOp::sink(vec![reduction()]);
        let (result, _, _) = compare_drivers(&valid, &[], walk, Context::default());
        assert_eq!(result.unwrap().validate(), Ok(()));
        let shared = reduction();
        let duplicate_use = UOp::sink(vec![shared.clone(), shared]);
        assert_eq!(duplicate_use.validate(), Err(UOpError::InvalidArgument));
        let (result, _, _) = compare_drivers(&duplicate_use, &[], walk, Context::default());
        assert_eq!(result.unwrap().validate(), Err(UOpError::InvalidArgument));

        // A genuine sibling change rebuilds the parent. Preserve the released
        // driver's actual memo-result sharing, not an invented identity rule.
        let rebuilt = UOp::sink(vec![reduction(), reduction(), literal(0)]);
        let rules = [rule(
            "advance-zero",
            0,
            UPat::op(Operation::Const(LiteralValue::Int(0))),
            |context, _, _| {
                context.calls.push("zero".into());
                Some(literal(1))
            },
        )];
        let (result, trace, context) = compare_drivers(&rebuilt, &rules, walk, Context::default());
        let result = result.unwrap();
        assert!(!result.shares_node_with(&rebuilt));
        assert!(result.sources()[0].shares_node_with(&result.sources()[1]));
        assert_eq!(result.validate(), Err(UOpError::InvalidArgument));
        assert_eq!(trace, ["advance-zero"]);
        assert_eq!(context.calls, ["zero"]);
    }
}

#[test]
fn rewrite_preserves_builtin_tag_and_raw_literal_behavior() {
    let ty = UType::scalar(crate::DType::F32);
    for bits in [0, 0x8000_0000, 0x7fc0_1234, 0x7fc0_5678, 0x7f80_0000] {
        let value = UOp::scalar_constant(crate::DType::F32, bits, ty);
        let root = UOp::binary(Binary::Add, value.clone(), value.clone())
            .retag(Some(UOpTag::Text("training".into())));
        for walk in [Walk::BottomUp, Walk::TopDown] {
            let actual = with_legacy(false, || rewrite(&root, &mut builtin_rules(), walk)).unwrap();
            let expected =
                with_legacy(true, || rewrite(&root, &mut builtin_rules(), walk)).unwrap();
            assert_eq!(actual.0, expected.0);
            assert_eq!(actual.1.rules, expected.1.rules);
            assert_eq!(artifact::encode(&actual.0), artifact::encode(&expected.0));
            assert_eq!(actual.0.tag(), expected.0.tag());
        }
    }
}

#[test]
fn rewrite_preserves_schedule_capture_and_cpu_results() {
    use crate::{Backend, CapturedSchedule, CpuBackend, Graph, Scalar, Shape, TensorData};
    for shape in [[2, 3], [0, 3]] {
        let mut graph = Graph::new();
        let input = graph.input("input", shape);
        let square = graph.square(input).unwrap();
        let view = graph.permute(square, [1, 0]).unwrap();
        let relu = graph.relu(view).unwrap();
        let sum = graph.sum(relu, 1).unwrap();
        let padded = graph.pad(sum, [(1, 2)], Scalar::F(0.0)).unwrap();
        let output = graph.square(padded).unwrap();
        let actual = with_legacy(false, || crate::schedule::schedule(&graph, output)).unwrap();
        let expected = with_legacy(true, || crate::schedule::schedule(&graph, output)).unwrap();
        actual.validate().unwrap();
        expected.validate().unwrap();
        assert_eq!(actual.items.len(), expected.items.len());
        for (actual, expected) in actual.items.iter().zip(&expected.items) {
            assert_eq!(actual.kernel, expected.kernel);
            assert_eq!(actual.kernel.topological(), expected.kernel.topological());
            assert_eq!(actual.cache_key, expected.cache_key);
            assert_eq!(actual.input_bindings, expected.input_bindings);
            assert_eq!(actual.dependencies, expected.dependencies);
            assert_eq!(actual.outputs, expected.outputs);
            assert_eq!(actual.boundary, expected.boundary);
        }
        let actual = CapturedSchedule::capture(&graph, &actual, &[output]).unwrap();
        let expected = CapturedSchedule::capture(&graph, &expected, &[output]).unwrap();
        assert_eq!(actual.to_bytes().unwrap(), expected.to_bytes().unwrap());
        let data = TensorData::new(
            Shape::from(shape),
            (0..shape.iter().product::<usize>())
                .map(|i| (i as f32 - 2.0) * 0.5)
                .collect(),
        )
        .unwrap();
        let oracle = CpuBackend
            .execute(
                &graph,
                output,
                &std::collections::HashMap::from([("input".into(), data.clone())]),
            )
            .unwrap();
        let bindings = BTreeMap::from([("input".into(), data)]);
        for capture in [actual, expected] {
            let replay = capture.replay(&bindings).unwrap();
            assert_eq!(replay[0].shape(), oracle.shape());
            assert_eq!(replay[0].storage(), oracle.storage());
        }
    }
}

#[derive(Default)]
struct TransientContext {
    intermediate: Option<std::sync::Weak<UOpNode>>,
    observed_release: bool,
}

fn transient(context: &mut TransientContext, _: &Captures, node: &UOp) -> Option<UOp> {
    match integer_value(node) {
        0 => {
            let next = literal(1);
            context.intermediate = Some(Arc::downgrade(&next.0));
            Some(next)
        }
        1 => Some(literal(2)),
        _ => None,
    }
}

fn observe_release(context: &mut TransientContext, _: &Captures, _: &UOp) -> Option<UOp> {
    let intermediate = context.intermediate.as_ref()?;
    assert!(
        intermediate.upgrade().is_none(),
        "completed child rewrite retained its intermediate"
    );
    context.observed_release = true;
    Some(literal(99))
}

#[test]
fn rewrite_releases_child_intermediates_before_parent_callback() {
    for walk in [Walk::BottomUp, Walk::TopDown] {
        for legacy in [false, true] {
            let root = UOp::binary(Binary::Add, literal(0), literal(5));
            let mut context = TransientContext::default();
            let mut rules = [
                ContextRewriteRule {
                    name: "advance",
                    priority: 0,
                    pattern: UPat::operation_predicate(integer),
                    apply: transient,
                },
                ContextRewriteRule {
                    name: "finish",
                    priority: 1,
                    pattern: UPat::op(Operation::Binary(Binary::Add)),
                    apply: observe_release,
                },
            ];
            let (output, trace) = with_legacy(legacy, || {
                rewrite_with_context(&root, &mut rules, walk, &mut context)
            })
            .unwrap();
            assert_eq!(output, literal(99));
            assert_eq!(trace.rules, ["advance", "advance", "finish"]);
            assert!(context.observed_release);
            assert!(context.intermediate.unwrap().upgrade().is_none());
        }
    }
}
