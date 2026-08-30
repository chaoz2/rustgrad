use crate::{Graph, Shape, SymbolicError, SymbolicExpr, SymbolicShape};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

fn variable(name: &str, min: i64, max: i64) -> SymbolicExpr {
    SymbolicExpr::variable(name, min, max).unwrap()
}
#[test]
fn symbolic_bounds_and_python_floor_semantics_are_checked() {
    let x = variable("x", -10, 10);
    let pos = variable("p", 1, 3);
    let neg = variable("n", -3, -1);
    for (name, expr, expected) in [
        (
            "positive-div",
            x.clone().try_floor_div(pos).unwrap(),
            (-10, 10),
        ),
        (
            "negative-div",
            x.clone().try_floor_div(neg).unwrap(),
            (-10, 10),
        ),
        (
            "positive-mod",
            x.clone().try_modulo(SymbolicExpr::constant(3)).unwrap(),
            (0, 2),
        ),
        (
            "negative-mod",
            x.clone().try_modulo(SymbolicExpr::constant(-3)).unwrap(),
            (-2, 0),
        ),
    ] {
        assert_eq!(
            (expr.bounds().unwrap().min, expr.bounds().unwrap().max),
            expected,
            "{name}"
        );
    }
    assert_eq!(
        SymbolicExpr::constant(-7)
            .try_floor_div(SymbolicExpr::constant(3))
            .unwrap()
            .evaluate(&BTreeMap::new())
            .unwrap(),
        -3
    );
    assert_eq!(
        SymbolicExpr::constant(-7)
            .try_modulo(SymbolicExpr::constant(3))
            .unwrap()
            .evaluate(&BTreeMap::new())
            .unwrap(),
        2
    );
    assert!(x.try_floor_div(SymbolicExpr::constant(0)).is_err());
}

#[test]
fn symbolic_simplification_is_canonical_and_semantics_preserving() {
    let x = variable("x", -3, 3);
    let y = variable("y", -2, 2);
    let original = (x.clone() + SymbolicExpr::constant(0)) + (y.clone() + x.clone());
    let simplified = original.simplify().unwrap();
    assert!(!simplified.trace.is_empty());
    for xv in -3..=3 {
        for yv in -2..=2 {
            let mut b = BTreeMap::new();
            let vars = original.variables();
            let mut it = vars.into_iter();
            let vx = it.next().unwrap();
            let vy = it.next().unwrap();
            if vx.name() == "x" {
                b.insert(vx, xv);
                b.insert(vy, yv);
            } else {
                b.insert(vx, yv);
                b.insert(vy, xv);
            }
            assert_eq!(
                original.evaluate(&b),
                simplified.expression.evaluate(&b),
                "x={xv}, y={yv}"
            );
        }
    }
    let same_name_a = variable("n", 0, 1);
    let same_name_b = variable("n", 0, 1);
    assert_ne!(same_name_a, same_name_b);
    assert!(simplified.trace.contains(&"combine-like-terms"));
}

#[test]
fn symbolic_shape_binds_before_graph_use() {
    let batch = variable("batch", 1, 4);
    let batch_var = batch.variables().into_iter().next().unwrap();
    let shape = SymbolicShape::new(vec![batch.into(), 3usize.into()]);
    let mut b = BTreeMap::new();
    b.insert(batch_var, 2);
    assert_eq!(shape.bind(&b).unwrap(), Shape::from([2, 3]));
    assert_eq!(shape.numel().unwrap().evaluate(&b).unwrap(), 6);
    let mut graph = Graph::new();
    let id = graph.input_symbolic("x", &shape, &b).unwrap();
    assert_eq!(graph.shape(id).unwrap(), &Shape::from([2, 3]));
    assert!(
        shape.bind(&BTreeMap::new()).is_err(),
        "unbound shape is rejected"
    );
    let mut out_of_bounds = b.clone();
    *out_of_bounds.values_mut().next().unwrap() = 5;
    assert!(
        shape.bind(&out_of_bounds).is_err(),
        "out-of-bound shape binding is rejected"
    );
}

#[test]
fn symbolic_substitution_is_simultaneous_canonical_and_bounds_checked() {
    let x = variable("x", 0, 8);
    let y = variable("y", 0, 2);
    let z = variable("z", 0, 4);
    let x_var = x.variables().into_iter().next().unwrap();
    let y_var = y.variables().into_iter().next().unwrap();
    let z_var = z.variables().into_iter().next().unwrap();
    let original = (x.clone() + y.clone()) + x.clone();
    let original_snapshot = original.clone();

    let replacement = z.clone() * SymbolicExpr::constant(2);
    let substituted = original
        .substitute(&BTreeMap::from([(x_var.clone(), replacement)]))
        .unwrap();
    assert_eq!(
        original, original_snapshot,
        "substitution must not mutate its source"
    );
    assert_eq!(
        substituted
            .expression
            .evaluate(&BTreeMap::from([(y_var, 1), (z_var, 3)])),
        Ok(13)
    );
    assert_eq!(substituted.trace.first(), Some(&"substitute"));

    let alternate = original
        .substitute(&BTreeMap::from([(x_var, z * SymbolicExpr::constant(2))]))
        .unwrap();
    assert_eq!(substituted.expression, alternate.expression);
    let mut left_hash = DefaultHasher::new();
    substituted.expression.hash(&mut left_hash);
    let mut right_hash = DefaultHasher::new();
    alternate.expression.hash(&mut right_hash);
    assert_eq!(left_hash.finish(), right_hash.finish());

    let too_wide = variable("wide", 0, 9);
    let too_wide_var = too_wide.variables().into_iter().next().unwrap();
    let x = variable("n", 0, 8);
    let x_var = x.variables().into_iter().next().unwrap();
    assert!(matches!(
        x.substitute(&BTreeMap::from([(x_var, too_wide)])),
        Err(SymbolicError::InvalidSubstitution { .. })
    ));
    let same_name = variable("n", 0, 8);
    let same_name_var = same_name.variables().into_iter().next().unwrap();
    assert_ne!(
        too_wide_var, same_name_var,
        "variable identity is not display identity"
    );
    assert!(matches!(
        x.substitute(&BTreeMap::from([(
            same_name_var,
            SymbolicExpr::constant(1)
        )])),
        Err(SymbolicError::ExtraBinding(_))
    ));
}

#[test]
fn symbolic_shape_substitution_preserves_zero_domains_and_rejects_overflow() {
    let extent = variable("extent", 0, 4);
    let extent_var = extent.variables().into_iter().next().unwrap();
    let shape = SymbolicShape::new(vec![extent.clone().into(), 0usize.into()]);
    let specialized = shape
        .substitute(&BTreeMap::from([(extent_var, SymbolicExpr::constant(0))]))
        .unwrap();
    assert_eq!(specialized.bind(&BTreeMap::new()), Ok(Shape::from([0, 0])));
    assert_eq!(
        specialized.numel().unwrap().evaluate(&BTreeMap::new()),
        Ok(0)
    );

    let max = variable("max", i64::MAX, i64::MAX);
    let max_var = max.variables().into_iter().next().unwrap();
    let slot = variable("slot", 0, i64::MAX);
    let slot_var = slot.variables().into_iter().next().unwrap();
    let source = slot.clone() + SymbolicExpr::constant(1);
    let source_snapshot = source.clone();
    assert!(matches!(
        source.substitute(&BTreeMap::from([(
            slot_var,
            max + SymbolicExpr::constant(1)
        )])),
        Err(SymbolicError::Overflow { op: "addition" })
    ));
    assert_eq!(
        source, source_snapshot,
        "failed substitution leaves the source intact"
    );
    assert_ne!(
        extent, slot,
        "independent variables retain structural identity"
    );
    assert_ne!(max_var.id(), 0);
}

#[test]
fn symbolic_shape_extent_overflow_rejects_before_graph_input_publication() {
    let left = variable("left", i64::MAX, i64::MAX);
    let right = variable("right", i64::MAX, i64::MAX);
    let left_var = left.variables().into_iter().next().unwrap();
    let right_var = right.variables().into_iter().next().unwrap();
    let shape = SymbolicShape::new(vec![left.into(), right.into()]);
    let bindings = BTreeMap::from([(left_var, i64::MAX), (right_var, i64::MAX)]);

    assert!(shape.bind(&bindings).is_err());
    let mut graph = Graph::new();
    let before = graph.node_count();
    assert!(graph.input_symbolic("overflow", &shape, &bindings).is_err());
    assert_eq!(graph.node_count(), before);
}
