//! Parsing and normalization for static dense Einstein summation.
//!
//! The plan is deliberately independent from graph execution so a future
//! lowering can reuse exactly the same validated coordinate contract.
use crate::{Error, Result, Shape};
use std::collections::{BTreeMap, BTreeSet};

/// An axis in a normalized [`EinsumPlan`].  Ellipsis axes are numbered from
/// the left after all operands' ellipses have been right-aligned.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EinsumLabel {
    Named(char),
    Ellipsis(usize),
}

/// A reusable, fully validated static indexed-contraction plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EinsumPlan {
    pub operand_labels: Vec<Vec<EinsumLabel>>,
    pub label_extents: BTreeMap<EinsumLabel, usize>,
    pub output_labels: Vec<EinsumLabel>,
    pub contracted_labels: Vec<EinsumLabel>,
}

impl EinsumPlan {
    /// Parses `equation` and normalizes it against the supplied operand shapes.
    pub fn parse(equation: &str, shapes: &[Shape]) -> Result<Self> {
        // tinygrad accepts presentation whitespace throughout its public
        // formula and normalizes before ellipsis/output processing.
        let normalized = equation
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        let equation = normalized.as_str();
        let (inputs, output) = split_equation(equation)?;
        if inputs.len() != shapes.len() {
            return Err(Error::EinsumOperandCount {
                expected: inputs.len(),
                actual: shapes.len(),
            });
        }
        let parsed = inputs
            .iter()
            .map(|s| parse_subscript(s))
            .collect::<Result<Vec<_>>>()?;
        let mut ellipsis_width = 0usize;
        for (term, shape) in parsed.iter().zip(shapes) {
            let explicit = term.labels.len();
            if term.ellipsis {
                let width =
                    shape
                        .rank()
                        .checked_sub(explicit)
                        .ok_or_else(|| Error::InvalidEinsum {
                            equation: equation.into(),
                            reason: "operand rank is smaller than its subscript",
                        })?;
                ellipsis_width = ellipsis_width.max(width);
            } else if shape.rank() != explicit {
                return Err(Error::InvalidEinsum {
                    equation: equation.into(),
                    reason: "operand rank does not match its subscript",
                });
            }
        }
        let mut operand_labels = Vec::with_capacity(parsed.len());
        for (term, shape) in parsed.iter().zip(shapes) {
            let width = shape.rank() - term.labels.len();
            let mut labels = Vec::with_capacity(shape.rank());
            if term.ellipsis {
                labels.extend((ellipsis_width - width..ellipsis_width).map(EinsumLabel::Ellipsis));
            }
            // The parser records explicit labels before and after the marker;
            // expand them in their original positions.
            let mut explicit = term.labels.iter();
            for piece in &term.pieces {
                match piece {
                    Piece::Label(_) => {
                        labels.push(EinsumLabel::Named(*explicit.next().expect("parsed label")))
                    }
                    Piece::Ellipsis => labels.extend(
                        (ellipsis_width - width..ellipsis_width).map(EinsumLabel::Ellipsis),
                    ),
                }
            }
            // Terms without an ellipsis use the ordinary path above.  Terms
            // with one must not retain the provisional prefix created above.
            if term.ellipsis {
                labels.clear();
                for piece in &term.pieces {
                    match piece {
                        Piece::Label(c) => labels.push(EinsumLabel::Named(*c)),
                        Piece::Ellipsis => labels.extend(
                            (ellipsis_width - width..ellipsis_width).map(EinsumLabel::Ellipsis),
                        ),
                    }
                }
            }
            debug_assert_eq!(labels.len(), shape.rank());
            operand_labels.push(labels);
        }

        let mut label_extents = BTreeMap::new();
        for (operand, shape) in operand_labels.iter().zip(shapes) {
            let mut local = BTreeMap::new();
            for (label, extent) in operand.iter().zip(shape.dims()) {
                if let Some(previous) = local.insert(label.clone(), *extent)
                    && previous != *extent
                {
                    return Err(Error::InvalidEinsum {
                        equation: equation.into(),
                        reason: "repeated label dimensions must be equal",
                    });
                }
                match label_extents.get(label).copied() {
                    None => {
                        label_extents.insert(label.clone(), *extent);
                    }
                    Some(previous) if previous == *extent || previous == 1 => {
                        label_extents.insert(label.clone(), previous.max(*extent));
                    }
                    Some(1) => {
                        label_extents.insert(label.clone(), *extent);
                    }
                    Some(_) if *extent == 1 => {}
                    Some(_) => {
                        return Err(Error::InvalidEinsum {
                            equation: equation.into(),
                            reason: "labeled dimensions are not broadcast-compatible",
                        });
                    }
                }
            }
        }
        // `0` broadcasts with `1`, but max(0, 1) is not the desired extent.
        for (label, extent) in label_extents.iter_mut() {
            let has_zero = operand_labels.iter().zip(shapes).any(|(labels, shape)| {
                labels
                    .iter()
                    .zip(shape.dims())
                    .any(|(l, d)| l == label && *d == 0)
            });
            if has_zero {
                *extent = 0;
            }
        }
        let output_labels = match output {
            Some(out) => expand_output(out, ellipsis_width, &label_extents, equation)?,
            None => implicit_output(&operand_labels, ellipsis_width),
        };
        let seen = output_labels.iter().cloned().collect::<BTreeSet<_>>();
        if seen.len() != output_labels.len()
            || output_labels
                .iter()
                .any(|label| !label_extents.contains_key(label))
        {
            return Err(Error::InvalidEinsum {
                equation: equation.into(),
                reason: "output labels must be unique and appear in an operand",
            });
        }
        let contracted_labels = label_extents
            .keys()
            .filter(|label| !seen.contains(*label))
            .cloned()
            .collect::<Vec<_>>();
        Shape::new(
            output_labels
                .iter()
                .map(|label| label_extents[label])
                .collect::<Vec<_>>(),
        )
        .numel()?;
        Shape::new(
            contracted_labels
                .iter()
                .map(|label| label_extents[label])
                .collect::<Vec<_>>(),
        )
        .numel()?;
        Ok(Self {
            operand_labels,
            label_extents,
            output_labels,
            contracted_labels,
        })
    }

    pub fn output_shape(&self) -> Shape {
        Shape::new(
            self.output_labels
                .iter()
                .map(|label| self.label_extents[label])
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Clone, Debug)]
enum Piece {
    Label(char),
    Ellipsis,
}
#[derive(Clone, Debug)]
struct Subscript {
    pieces: Vec<Piece>,
    labels: Vec<char>,
    ellipsis: bool,
}

fn split_equation(equation: &str) -> Result<(Vec<&str>, Option<&str>)> {
    let mut arrows = equation.split("->");
    let input = arrows.next().unwrap_or_default();
    let output = arrows.next();
    if arrows.next().is_some() {
        return Err(Error::InvalidEinsum {
            equation: equation.into(),
            reason: "equation must contain one non-empty input side and at most one arrow",
        });
    }
    Ok((input.split(',').collect(), output))
}

fn parse_subscript(text: &str) -> Result<Subscript> {
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut pieces = Vec::new();
    let mut labels = Vec::new();
    let mut ellipsis = false;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"...") {
            if ellipsis {
                return Err(Error::InvalidEinsum {
                    equation: text.into(),
                    reason: "a subscript may contain only one ellipsis",
                });
            }
            ellipsis = true;
            pieces.push(Piece::Ellipsis);
            i += 3;
        } else {
            let c = bytes[i] as char;
            if !c.is_ascii_alphabetic() {
                return Err(Error::InvalidEinsum {
                    equation: text.into(),
                    reason: "subscripts use ASCII letters and an optional ellipsis",
                });
            }
            pieces.push(Piece::Label(c));
            labels.push(c);
            i += 1;
        }
    }
    Ok(Subscript {
        pieces,
        labels,
        ellipsis,
    })
}

fn expand_output(
    text: &str,
    ellipsis_width: usize,
    extents: &BTreeMap<EinsumLabel, usize>,
    equation: &str,
) -> Result<Vec<EinsumLabel>> {
    let parsed = parse_subscript(text)?;
    let mut output = Vec::new();
    for piece in parsed.pieces {
        match piece {
            Piece::Label(c) => output.push(EinsumLabel::Named(c)),
            Piece::Ellipsis => output.extend((0..ellipsis_width).map(EinsumLabel::Ellipsis)),
        }
    }
    if output.iter().any(|label| !extents.contains_key(label)) {
        return Err(Error::InvalidEinsum {
            equation: equation.into(),
            reason: "output contains a label absent from the inputs",
        });
    }
    Ok(output)
}

fn implicit_output(operands: &[Vec<EinsumLabel>], ellipsis_width: usize) -> Vec<EinsumLabel> {
    let mut counts = BTreeMap::<char, usize>::new();
    for labels in operands {
        for label in labels {
            if let EinsumLabel::Named(c) = label {
                *counts.entry(*c).or_default() += 1;
            }
        }
    }
    let mut output = (0..ellipsis_width)
        .map(EinsumLabel::Ellipsis)
        .collect::<Vec<_>>();
    output.extend(
        counts
            .into_iter()
            .filter_map(|(c, n)| (n == 1).then_some(EinsumLabel::Named(c))),
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shapes(dims: &[&[usize]]) -> Vec<Shape> {
        dims.iter().map(|d| Shape::new(d.to_vec())).collect()
    }
    #[test]
    fn table_driven_normalization() {
        let cases = [
            ("ij,jk->ik", shapes(&[&[2, 3], &[3, 4]]), vec![2, 4]),
            (
                "...ij,...jk->...ik",
                shapes(&[&[5, 2, 3], &[1, 3, 4]]),
                vec![5, 2, 4],
            ),
            ("ii->i", shapes(&[&[3, 3]]), vec![3]),
            ("ij,j", shapes(&[&[2, 3], &[3]]), vec![2]),
            (",->", shapes(&[&[], &[]]), vec![]),
            (
                " i j ,  j k  ->  i k ",
                shapes(&[&[2, 3], &[3, 4]]),
                vec![2, 4],
            ),
            (
                " ... i , ... i  -> ... ",
                shapes(&[&[2, 3], &[1, 3]]),
                vec![2],
            ),
        ];
        for (equation, input, output) in cases {
            assert_eq!(
                EinsumPlan::parse(equation, &input).unwrap().output_shape(),
                Shape::new(output)
            );
        }
    }
    #[test]
    fn rejects_invalid_forms() {
        for (equation, input) in [
            ("ij", shapes(&[&[2]])),
            ("ii", shapes(&[&[2, 3]])),
            ("ij->ii", shapes(&[&[2, 3]])),
            ("ij->k", shapes(&[&[2, 3]])),
            ("i...j...", shapes(&[&[2, 3]])),
            ("i$,j", shapes(&[&[2], &[2]])),
            ("ij,jk", shapes(&[&[2, 3], &[4, 5]])),
        ] {
            assert!(EinsumPlan::parse(equation, &input).is_err(), "{equation}");
        }
    }
}
