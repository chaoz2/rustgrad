use super::*;
use crate::DType;
use crate::uop::{
    Captures, ContextRewriteRule, Operation, UOpError, UPat, UType, Walk, rewrite_with_context,
};
use std::sync::Arc;

fn special(name: &str) -> UOp {
    UOp::from_operation(Operation::Special(name.into()), None, vec![])
}

fn literal(value: i64) -> UOp {
    UOp::constant(value, UType::scalar(DType::I32))
}

fn bind(context: &mut usize, _: &Captures, _: &UOp) -> Option<UOp> {
    *context += 1;
    Some(special("bound"))
}

#[test]
fn completed_allocation_lookups_keep_structural_callbacks_and_trace() {
    for walk in [Walk::BottomUp, Walk::TopDown] {
        for shared in [true, false] {
            let input = special("unbound");
            let root = UOp::sink(
                (0..8)
                    .map(|_| {
                        if shared {
                            input.clone()
                        } else {
                            special("unbound")
                        }
                    })
                    .collect(),
            );
            let mut rules = [ContextRewriteRule {
                name: "bind",
                priority: 0,
                pattern: UPat::op(Operation::Special("unbound".into())),
                apply: bind,
            }];
            let mut context = 0;
            let (actual, counts) =
                with_lookup_counts(|| rewrite_with_context(&root, &mut rules, walk, &mut context));
            let (actual, trace) = actual.unwrap();
            let mut reference_context = 0;
            let (expected, expected_trace) = super::super::rewrite_key_tests::recursive_driver(
                &root,
                &mut rules,
                walk,
                &mut reference_context,
            )
            .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(trace.rules, expected_trace.rules);
            assert_eq!(trace.rules, ["bind"]);
            assert_eq!(context, reference_context);
            assert_eq!(context, 1);
            assert_eq!(counts.queries, 17);
            assert_eq!(counts.allocation_hits, if shared { 14 } else { 7 });
            assert_eq!(counts.structural_queries, if shared { 3 } else { 10 });
            assert!(
                actual
                    .sources()
                    .iter()
                    .all(|source| { source.shares_node_with(&actual.sources()[0]) })
            );
            assert_eq!(actual.validate(), expected.validate());
            assert_eq!(
                crate::uop::artifact::encode(&actual),
                crate::uop::artifact::encode(&expected)
            );
        }
    }
}

#[test]
fn completed_memo_keeps_alias_release_and_replacement_owners() {
    let mut memo = RewriteMemo::default();
    let input = literal(0);
    let output = literal(99);
    let input_weak = Arc::downgrade(&input.0);
    let output_weak = Arc::downgrade(&output.0);
    memo.insert(input.clone(), output.clone());
    drop(output);
    let alias = literal(0);
    let alias_weak = Arc::downgrade(&alias.0);
    let (found, counts) = with_lookup_counts(|| memo.get(&alias));
    assert!(
        found
            .unwrap()
            .shares_node_with(&output_weak.upgrade().map(UOp).unwrap())
    );
    assert_eq!(counts.allocation_hits, 0);
    drop(alias);
    assert!(
        alias_weak.upgrade().is_none(),
        "structural hits must not retain aliases"
    );

    let replacement_alias = literal(0);
    let replacement_weak = Arc::downgrade(&replacement_alias.0);
    let replacement = literal(42);
    let replacement_output_weak = Arc::downgrade(&replacement.0);
    memo.insert(replacement_alias, replacement);
    assert!(replacement_weak.upgrade().is_none());
    assert!(output_weak.upgrade().is_none());
    let (found, counts) = with_lookup_counts(|| memo.get(&input));
    assert_eq!(found.unwrap(), literal(42));
    assert_eq!(counts.allocation_hits, 1);
    drop(input);
    assert!(input_weak.upgrade().is_some() && replacement_output_weak.upgrade().is_some());
    drop(memo);
    assert!(input_weak.upgrade().is_none() && replacement_output_weak.upgrade().is_none());
}

#[test]
fn completed_memo_does_not_publish_misses_or_escape_a_rewrite_call() {
    let mut memo = RewriteMemo::default();
    let input = literal(0);
    let ((), counts) = with_lookup_counts(|| {
        assert!(memo.get(&input).is_none());
        assert!(memo.get(&input).is_none());
        memo.insert(input.clone(), literal(1));
        assert_eq!(memo.get(&input), Some(literal(1)));
    });
    assert_eq!(counts.queries, 3);
    assert_eq!(counts.structural_queries, 2);
    assert_eq!(counts.allocation_hits, 1);

    for walk in [Walk::BottomUp, Walk::TopDown] {
        for _ in 0..2 {
            let root = UOp::sink(vec![special("unbound"), literal(0)]);
            let mut rules = [
                ContextRewriteRule {
                    name: "bind",
                    priority: 0,
                    pattern: UPat::op(Operation::Special("unbound".into())),
                    apply: |owners: &mut Vec<_>, _, _| {
                        let next = special("bound");
                        owners.push(Arc::downgrade(&next.0));
                        Some(next)
                    },
                },
                ContextRewriteRule {
                    name: "increment",
                    priority: 1,
                    pattern: UPat::operation_predicate(|operation| {
                        matches!(operation, Operation::Const(_))
                    }),
                    apply: |owners: &mut Vec<_>, _, node| {
                        let Operation::Const(crate::uop::LiteralValue::Int(value)) =
                            node.operation()
                        else {
                            panic!("expected integer literal");
                        };
                        let next = literal(value + 1);
                        owners.push(Arc::downgrade(&next.0));
                        Some(next)
                    },
                },
            ];
            let mut owners = vec![];
            let error = rewrite_with_context(&root, &mut rules, walk, &mut owners).unwrap_err();
            assert_eq!(error, UOpError::RewriteStepLimit);
            assert_eq!(owners.len(), 129);
            assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
        }
    }
}
