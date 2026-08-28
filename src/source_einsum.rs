//! Descriptor-only parsing for tinygrad's public `Tensor.einsum` literal.
//!
//! This is intentionally separate from [`crate::EinsumPlan`].  The latter is
//! the normalized contract consumed by the raw `Op::Einsum`; this module keeps
//! the concrete ASCII ellipsis names and movement sequence needed by a future
//! source-literal Graph lowering.

use crate::{Error, Result, Shape};
use std::collections::{BTreeMap, BTreeSet};

/// One zero-offset diagonal extraction in tinygrad's literal implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceDiagonalStep {
    pub label: char,
    pub first_axis: usize,
    pub second_axis: usize,
    pub extent: usize,
    pub permutation: Vec<usize>,
    pub input_shape: Shape,
    pub output_shape: Shape,
}

/// How an operand is permuted and expanded into tinygrad's sorted alphabet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceEinsumOperandPlan {
    pub input_labels: Vec<char>,
    pub diagonal_steps: Vec<SourceDiagonalStep>,
    pub final_labels: Vec<char>,
    pub final_shape: Shape,
    pub alignment_permutation: Vec<usize>,
    pub aligned_shape: Shape,
}

/// A source-faithful, graph-free descriptor plan for public tinygrad Einsum.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceEinsumPlan {
    pub normalized_formula: String,
    pub expanded_inputs: Vec<String>,
    pub expanded_output: String,
    pub alphabet: Vec<char>,
    pub label_extents: BTreeMap<char, usize>,
    pub operands: Vec<SourceEinsumOperandPlan>,
    pub reduction_axes: Vec<usize>,
    pub product_shape: Shape,
    pub reduced_shape: Shape,
    pub final_permutation: Vec<usize>,
    pub output_shape: Shape,
}

impl SourceEinsumPlan {
    /// Reproduces the checked-in Python parser and all static movement shapes.
    pub(crate) fn parse(formula: &str, shapes: &[Shape]) -> Result<Self> {
        let formula = formula.replace(' ', "");
        let (lhs, explicit_rhs) = split_formula(&formula)?;
        let raw_inputs = lhs.split(',').collect::<Vec<_>>();
        if raw_inputs.len() != shapes.len() {
            return Err(Error::EinsumOperandCount {
                expected: raw_inputs.len(),
                actual: shapes.len(),
            });
        }
        for shape in shapes {
            shape.numel()?;
        }

        let (expanded_inputs, expanded_output) = if formula.contains("...") {
            expand_ellipsis(&formula, &raw_inputs, explicit_rhs, shapes)?
        } else {
            (
                raw_inputs.iter().map(|s| (*s).to_owned()).collect(),
                explicit_rhs
                    .map(str::to_owned)
                    .unwrap_or_else(|| implicit_output(lhs)),
            )
        };
        validate_labels(&expanded_output, &formula)?;
        let mut operands = Vec::with_capacity(shapes.len());
        for (input, shape) in expanded_inputs.iter().zip(shapes) {
            let labels = chars(input, &formula)?;
            if labels.len() != shape.rank() {
                return invalid(&formula, "operand rank does not match its subscript");
            }
            let operand = diagonal_plan(labels, shape.clone(), &formula)?;
            operands.push(operand);
        }

        // `merge_dicts([dict(zip(s, x.shape)) ...])` accepts no ordinary
        // broadcast exception: a shared label must have precisely one extent,
        // including the observable `0` versus `1` case.
        let mut label_extents = BTreeMap::new();
        for operand in &operands {
            for (label, extent) in operand.final_labels.iter().zip(operand.final_shape.dims()) {
                match label_extents.insert(*label, *extent) {
                    Some(previous) if previous != *extent => {
                        return invalid(&formula, "labeled dimensions must be equal")
                    }
                    _ => {}
                }
            }
        }
        let alphabet = label_extents.keys().copied().collect::<Vec<_>>();
        let rhs = chars(&expanded_output, &formula)?;
        let output_seen = rhs.iter().copied().collect::<BTreeSet<_>>();
        if output_seen.len() != rhs.len() || rhs.iter().any(|label| !label_extents.contains_key(label)) {
            return invalid(&formula, "output labels must be unique and appear in an operand");
        }

        for operand in &mut operands {
            if operand.final_labels.is_empty() {
                // tinygrad leaves a scalar operand untouched rather than
                // reshaping it to all singleton alphabet axes.
                operand.aligned_shape.numel()?;
                continue;
            }
            operand.alignment_permutation = alphabet
                .iter()
                .filter_map(|label| operand.final_labels.iter().position(|x| x == label))
                .collect();
            let permuted_shape = Shape::new(
                operand
                    .alignment_permutation
                    .iter()
                    .map(|axis| operand.final_shape.dims()[*axis])
                    .collect::<Vec<_>>(),
            );
            permuted_shape.numel()?;
            operand.aligned_shape = Shape::new(
                alphabet
                    .iter()
                    .map(|label| if operand.final_labels.contains(label) { label_extents[label] } else { 1 })
                    .collect::<Vec<_>>(),
            );
            operand.aligned_shape.numel()?;
        }
        let product_shape = Shape::new(
            alphabet.iter().map(|label| label_extents[label]).collect::<Vec<_>>(),
        );
        product_shape.numel()?;
        let reduction_axes = alphabet
            .iter()
            .enumerate()
            .filter_map(|(axis, label)| (!rhs.contains(label)).then_some(axis))
            .collect::<Vec<_>>();
        let reduced_labels = alphabet
            .iter()
            .copied()
            .filter(|label| rhs.contains(label))
            .collect::<Vec<_>>();
        let reduced_shape = Shape::new(
            reduced_labels.iter().map(|label| label_extents[label]).collect::<Vec<_>>(),
        );
        reduced_shape.numel()?;
        let final_permutation = rhs
            .iter()
            .map(|label| reduced_labels.iter().position(|x| x == label).expect("validated output label"))
            .collect::<Vec<_>>();
        let output_shape = Shape::new(rhs.iter().map(|label| label_extents[label]).collect::<Vec<_>>());
        output_shape.numel()?;
        Ok(Self {
            normalized_formula: formula,
            expanded_inputs,
            expanded_output,
            alphabet,
            label_extents,
            operands,
            reduction_axes,
            product_shape,
            reduced_shape,
            final_permutation,
            output_shape,
        })
    }
}

fn diagonal_plan(mut labels: Vec<char>, mut shape: Shape, equation: &str) -> Result<SourceEinsumOperandPlan> {
    let input_labels = labels.clone();
    let mut diagonal_steps = Vec::new();
    // tinygrad currently traverses `set(s)`.  That traversal is randomized by
    // Python's string hash seed and therefore is not a stable public contract.
    // We choose first occurrence deterministically.  Extractions for distinct
    // labels commute: each is a zero-offset diagonal on only its two equal-
    // label axes, and the recorded labels/permutations carry the corresponding
    // axis renumbering.  Thus values, dtype, final shape, and compositional VJP
    // are invariant; only unobservable movement-node order differs.
    let labels_in_first_occurrence = labels.clone().into_iter().fold(Vec::new(), |mut seen, c| {
        if !seen.contains(&c) { seen.push(c); }
        seen
    });
    for label in labels_in_first_occurrence {
        while labels.iter().filter(|c| **c == label).count() > 1 {
            let first_axis = labels.iter().position(|c| *c == label).expect("counted label");
            let second_axis = labels
                .iter()
                .enumerate()
                .skip(first_axis + 1)
                .find_map(|(axis, c)| (*c == label).then_some(axis))
                .expect("counted second label");
            let extent = shape.dims()[first_axis];
            if shape.dims()[second_axis] != extent {
                return invalid(equation, "repeated label dimensions must be equal");
            }
            let permutation = (0..shape.rank())
                .filter(|axis| *axis != first_axis && *axis != second_axis)
                .chain([first_axis, second_axis])
                .collect::<Vec<_>>();
            // The rank>2 source path materializes these descriptor boundaries:
            // flatten(n,n), pad by n, unflatten(n,n+1), then select column 0.
            // Validate them even when a leading zero makes the final numel zero.
            if shape.rank() > 2 {
                let n_squared = extent.checked_mul(extent).ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                let padded = n_squared.checked_add(extent).ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                let n_plus_one = extent.checked_add(1).ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                let prefix = permutation[..permutation.len() - 2]
                    .iter().map(|axis| shape.dims()[*axis]).collect::<Vec<_>>();
                Shape::new(prefix.iter().copied().chain([n_squared]).collect::<Vec<_>>()).numel()?;
                Shape::new(prefix.iter().copied().chain([padded]).collect::<Vec<_>>()).numel()?;
                Shape::new(prefix.iter().copied().chain([extent, n_plus_one]).collect::<Vec<_>>()).numel()?;
            }
            let mut output_dims = shape.dims().to_vec();
            output_dims.remove(second_axis);
            let output_shape = Shape::new(output_dims);
            output_shape.numel()?;
            diagonal_steps.push(SourceDiagonalStep {
                label,
                first_axis,
                second_axis,
                extent,
                permutation,
                input_shape: shape.clone(),
                output_shape: output_shape.clone(),
            });
            labels.remove(second_axis);
            shape = output_shape;
        }
    }
    shape.numel()?;
    Ok(SourceEinsumOperandPlan {
        input_labels,
        diagonal_steps,
        final_labels: labels,
        final_shape: shape.clone(),
        alignment_permutation: Vec::new(),
        aligned_shape: shape,
    })
}

fn expand_ellipsis(
    formula: &str,
    raw_inputs: &[&str],
    explicit_rhs: Option<&str>,
    shapes: &[Shape],
) -> Result<(Vec<String>, String)> {
    let ell = (b'a'..=b'z').chain(b'A'..=b'Z')
        .map(char::from)
        .filter(|c| !formula.contains(*c))
        .collect::<String>();
    let widths = raw_inputs.iter().zip(shapes).map(|(subscript, shape)| {
        if subscript.contains("...") {
            let explicit = subscript.len().checked_sub(3).expect("ellipsis present");
            shape.rank().checked_sub(explicit).ok_or_else(|| Error::InvalidEinsum {
                equation: formula.into(), reason: "operand rank is smaller than its subscript",
            })
        } else { Ok(0) }
    }).collect::<Result<Vec<_>>>()?;
    let width = widths.iter().copied().max().unwrap_or(0);
    if width > ell.len() { return invalid(formula, "ellipsis expansion exhausts ASCII labels"); }
    let expanded_inputs = raw_inputs.iter().zip(widths).map(|(subscript, operand_width)| {
        if subscript.matches("...").count() > 1 { return invalid(formula, "a subscript may contain only one ellipsis"); }
        Ok(subscript.replace("...", &ell[width - operand_width..width]))
    }).collect::<Result<Vec<_>>>()?;
    let expanded_output = match explicit_rhs {
        Some(output) => {
            if output.matches("...").count() > 1 { return invalid(formula, "a subscript may contain only one ellipsis"); }
            output.replace("...", &ell[..width])
        }
        None => {
            let lhs = expanded_inputs.join(",");
            format!("{}{}", &ell[..width], implicit_output(&lhs))
        }
    };
    Ok((expanded_inputs, expanded_output))
}

fn split_formula(formula: &str) -> Result<(&str, Option<&str>)> {
    let mut pieces = formula.split("->");
    let lhs = pieces.next().unwrap_or_default();
    let rhs = pieces.next();
    if pieces.next().is_some() { return invalid(formula, "equation must contain at most one arrow"); }
    Ok((lhs, rhs))
}

fn implicit_output(text: &str) -> String {
    text.chars().filter(|c| c.is_ascii_alphabetic() && text.matches(*c).count() == 1).collect::<BTreeSet<_>>()
        .into_iter().collect()
}

fn validate_labels(text: &str, equation: &str) -> Result<()> {
    chars(text, equation).map(|_| ())
}

fn chars(text: &str, equation: &str) -> Result<Vec<char>> {
    if !text.chars().all(|c| c.is_ascii_alphabetic()) {
        return invalid(equation, "subscripts use ASCII letters and an optional ellipsis");
    }
    Ok(text.chars().collect())
}

fn invalid<T>(equation: &str, reason: &'static str) -> Result<T> {
    Err(Error::InvalidEinsum { equation: equation.into(), reason })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shapes(dims: &[&[usize]]) -> Vec<Shape> { dims.iter().map(|d| Shape::new(d.to_vec())).collect() }

    #[test]
    fn preserves_concrete_ellipsis_letters_and_sorted_source_alphabet() {
        let plan = SourceEinsumPlan::parse(" a... , ...b -> ...ab ", &shapes(&[&[2, 3, 4], &[5, 4]])).unwrap();
        // `a` and `b` are occupied, so tinygrad's ascii_letters pool starts
        // at c; unequal ellipses right-align to `cd` and `d`.
        assert_eq!(plan.expanded_inputs, vec!["acd".to_owned(), "db".to_owned()]);
        assert_eq!(plan.expanded_output, "cdab");
        assert_eq!(plan.alphabet, vec!['a', 'b', 'c', 'd']);
        assert_eq!(plan.output_shape, Shape::new([2, 5, 3, 4]));
    }

    #[test]
    fn diagonal_order_is_deterministic_and_semantically_invariant() {
        let left = SourceEinsumPlan::parse("aabb->ab", &shapes(&[&[2, 2, 3, 3]])).unwrap();
        let right = SourceEinsumPlan::parse("bbaa->ab", &shapes(&[&[3, 3, 2, 2]])).unwrap();
        assert_eq!(left.operands[0].diagonal_steps.iter().map(|s| s.label).collect::<Vec<_>>(), vec!['a', 'b']);
        assert_eq!(right.operands[0].diagonal_steps.iter().map(|s| s.label).collect::<Vec<_>>(), vec!['b', 'a']);
        assert_eq!(left.operands[0].final_labels, vec!['a', 'b']);
        assert_eq!(right.operands[0].final_labels, vec!['b', 'a']);
        // Their later source alignment and explicit output contract agree.
        assert_eq!(left.alphabet, right.alphabet);
        assert_eq!(left.output_shape, right.output_shape);
        assert_eq!(left.final_permutation, right.final_permutation);
    }

    #[test]
    fn handles_scalars_zero_extents_and_output_reordering() {
        let scalar = SourceEinsumPlan::parse("->", &shapes(&[&[]])).unwrap();
        assert_eq!(scalar.output_shape, Shape::new([]));
        let zero = SourceEinsumPlan::parse("ij->ji", &shapes(&[&[0, 3]])).unwrap();
        assert_eq!(zero.product_shape, Shape::new([0, 3]));
        assert_eq!(zero.final_permutation, vec![1, 0]);
    }

    #[test]
    fn rejects_source_incompatible_and_malformed_forms() {
        for (equation, input) in [
            ("ij->i->j", shapes(&[&[2, 3]])),
            ("ij", shapes(&[&[2]])),
            ("ii", shapes(&[&[2, 3]])),
            ("i,i", shapes(&[&[0], &[1]])),
            ("ij->ii", shapes(&[&[2, 3]])),
            ("ij->k", shapes(&[&[2, 3]])),
            ("i...j...", shapes(&[&[2, 3]])),
            ("i,j", shapes(&[&[2]])),
            ("ij", shapes(&[&[usize::MAX, 2]])),
        ] {
            assert!(SourceEinsumPlan::parse(equation, &input).is_err(), "{equation}");
        }
    }
}
