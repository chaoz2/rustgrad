use crate::{Error, Graph, NodeId, Result, Shape};
use std::collections::{BTreeMap, BTreeSet};

/// The static section specification accepted by [`Graph::split`].
///
/// `Size` produces equal-sized sections with a possible shorter final
/// section. `Sections` spells out every section and must cover the selected
/// axis exactly. This is the static subset of tinygrad's `Tensor.split`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SplitSizes {
    Size(usize),
    Sections(Vec<usize>),
}

impl From<usize> for SplitSizes {
    fn from(size: usize) -> Self {
        Self::Size(size)
    }
}

impl From<Vec<usize>> for SplitSizes {
    fn from(sections: Vec<usize>) -> Self {
        Self::Sections(sections)
    }
}

impl From<&[usize]> for SplitSizes {
    fn from(sections: &[usize]) -> Self {
        Self::Sections(sections.to_vec())
    }
}

/// A fully checked static partition of one tensor axis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StaticSplitPlan {
    axis: usize,
    sections: Vec<usize>,
}

impl StaticSplitPlan {
    fn split(input: NodeId, shape: &Shape, sizes: SplitSizes, axis: isize) -> Result<Self> {
        let axis = resolve_graph_axis(input, axis, shape.rank())?;
        let extent = shape.dims()[axis];
        let sections = match sizes {
            // tinygrad's `range(0, max(1, dim_sz), ...)` has no iterations
            // for a zero axis, even for a zero scalar split size.
            SplitSizes::Size(_) if extent == 0 => Vec::new(),
            SplitSizes::Size(0) => {
                return Err(Error::InvalidSplit {
                    reason: "split size must be positive for a non-empty axis",
                });
            }
            SplitSizes::Size(size) => {
                let count = extent / size + usize::from(extent % size != 0);
                (0..count)
                    .map(|part| {
                        let start = part
                            .checked_mul(size)
                            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                        Ok((extent - start).min(size))
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            SplitSizes::Sections(sections) => {
                let total = sections.iter().try_fold(0usize, |total, section| {
                    total
                        .checked_add(*section)
                        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                })?;
                if total != extent {
                    return Err(Error::InvalidSplit {
                        reason: "section sizes must sum exactly to the selected axis",
                    });
                }
                sections
            }
        };
        Ok(Self { axis, sections })
    }

    fn chunk(input: NodeId, shape: &Shape, chunks: usize, axis: isize) -> Result<Self> {
        if chunks == 0 {
            return Err(Error::InvalidSplit {
                reason: "chunk count must be positive",
            });
        }
        let axis = resolve_graph_axis(input, axis, shape.rank())?;
        let extent = shape.dims()[axis];
        if extent == 0 {
            return Ok(Self {
                axis,
                sections: vec![0; chunks],
            });
        }
        let size = extent / chunks + usize::from(extent % chunks != 0);
        let count = extent / size + usize::from(extent % size != 0);
        let sections = (0..count)
            .map(|part| {
                let start = part
                    .checked_mul(size)
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                Ok((extent - start).min(size))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { axis, sections })
    }

    fn bounds(&self, shape: &Shape) -> Result<Vec<Vec<(usize, usize)>>> {
        let mut start = 0usize;
        self.sections
            .iter()
            .map(|section| {
                let end = start
                    .checked_add(*section)
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                let mut bounds = shape
                    .dims()
                    .iter()
                    .map(|extent| (0, *extent))
                    .collect::<Vec<_>>();
                bounds[self.axis] = (start, end);
                start = end;
                Ok(bounds)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Term {
    Axis(String),
    Group(Vec<String>),
    Ellipsis,
}

/// Parsed static einops-compatible rearrangement pattern.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RearrangePattern {
    input: Vec<Term>,
    output: Vec<Term>,
    text: String,
}

impl RearrangePattern {
    /// Parses the static tinygrad/einops rearrangement grammar used by RustGrad.
    pub fn parse(pattern: &str) -> Result<Self> {
        let mut arrow = pattern.split("->");
        let left = arrow.next().unwrap_or_default();
        let right = arrow.next();
        if right.is_none() || arrow.next().is_some() {
            return Err(rearrange_err(pattern, "pattern needs exactly one arrow"));
        }
        let input = parse_side(left, pattern)?;
        let output = parse_side(right.unwrap(), pattern)?;
        if input.iter().filter(|x| has_ellipsis(x)).count() > 1
            || output.iter().filter(|x| has_ellipsis(x)).count() > 1
        {
            return Err(rearrange_err(
                pattern,
                "each side may contain at most one ellipsis",
            ));
        }
        if input
            .iter()
            .any(|x| matches!(x, Term::Group(xs) if xs.iter().any(|x| x == "...")))
        {
            return Err(rearrange_err(pattern, "input ellipsis cannot be grouped"));
        }
        Ok(Self {
            input,
            output,
            text: pattern.into(),
        })
    }
}

impl Graph {
    /// Splits an axis into static sections. A scalar size creates equal
    /// sections plus a smaller final section; explicit sections cover the
    /// axis exactly. Results are immutable `Shrink` views of `input`.
    pub fn split(
        &mut self,
        input: NodeId,
        sizes: impl Into<SplitSizes>,
        axis: isize,
    ) -> Result<Vec<NodeId>> {
        let shape = self.node(input)?.shape.clone();
        let plan = StaticSplitPlan::split(input, &shape, sizes.into(), axis)?;
        plan.bounds(&shape)?
            .into_iter()
            .map(|bounds| self.shrink(input, bounds))
            .collect()
    }

    /// Splits an axis into at most `chunks` near-equal static sections.
    /// Following tinygrad, a zero-length selected axis yields `chunks` empty
    /// sections, while a non-empty axis may yield fewer sections.
    pub fn chunk(&mut self, input: NodeId, chunks: usize, axis: isize) -> Result<Vec<NodeId>> {
        let shape = self.node(input)?.shape.clone();
        let plan = StaticSplitPlan::chunk(input, &shape, chunks, axis)?;
        plan.bounds(&shape)?
            .into_iter()
            .map(|bounds| self.shrink(input, bounds))
            .collect()
    }

    /// Rearranges a tensor through static split, reorder, merge, singleton,
    /// and ellipsis operations. `sizes` supplies named split factors.
    pub fn rearrange(
        &mut self,
        input: NodeId,
        pattern: &str,
        sizes: &BTreeMap<String, usize>,
    ) -> Result<NodeId> {
        let parsed = RearrangePattern::parse(pattern)?;
        let source_shape = self.node(input)?.shape.clone();
        let (left, right) = expand_ellipsis(&parsed, source_shape.rank())?;
        let mut supplied = sizes.clone();
        let mut names = BTreeSet::new();
        let mut left_names = Vec::new();
        for term in &left {
            for name in names_of(term) {
                if !names.insert(name.clone()) {
                    return Err(rearrange_err(pattern, "axis names must be unique"));
                }
                left_names.push(name);
            }
        }
        if supplied.keys().any(|name| !names.contains(name)) {
            return Err(rearrange_err(pattern, "named size is not used in pattern"));
        }
        let right_names = right.iter().flat_map(names_of).collect::<Vec<_>>();
        if right_names.len() != left_names.len()
            || right_names.iter().collect::<BTreeSet<_>>().len() != right_names.len()
            || right_names.iter().any(|name| !names.contains(name))
        {
            return Err(rearrange_err(
                pattern,
                "input and output axis names must match exactly",
            ));
        }
        let mut dims = BTreeMap::new();
        let mut elementary = Vec::new();
        for (term, extent) in left.iter().zip(source_shape.dims()) {
            let axes = names_of(term);
            if axes.is_empty() {
                if *extent != 1 {
                    return Err(rearrange_err(
                        pattern,
                        "empty group needs a size-one input axis",
                    ));
                }
                continue;
            }
            let known = axes
                .iter()
                .filter_map(|name| supplied.get(name))
                .try_fold(1usize, |p, n| p.checked_mul(*n))
                .ok_or_else(|| Error::ShapeOverflow(source_shape.clone()))?;
            let unresolved = axes
                .iter()
                .filter(|name| !supplied.contains_key(*name))
                .cloned()
                .collect::<Vec<_>>();
            if unresolved.len() > 1 {
                return Err(rearrange_err(
                    pattern,
                    "a split group needs all but one factor specified",
                ));
            }
            if let Some(name) = unresolved.first() {
                if known == 0 || extent % known != 0 {
                    return Err(rearrange_err(
                        pattern,
                        "split factors do not divide input axis",
                    ));
                }
                supplied.insert(name.clone(), extent / known);
            } else if known != *extent {
                return Err(rearrange_err(
                    pattern,
                    "provided axis length does not match input",
                ));
            }
            for name in axes {
                dims.insert(name.clone(), supplied[&name]);
                elementary.push(name);
            }
        }
        let intermediate_shape =
            Shape::new(elementary.iter().map(|name| dims[name]).collect::<Vec<_>>());
        let mut node = self.reshape(input, intermediate_shape)?;
        let order = right_names
            .iter()
            .map(|name| {
                elementary
                    .iter()
                    .position(|x| x == name)
                    .ok_or_else(|| rearrange_err(pattern, "axis mismatch"))
            })
            .collect::<Result<Vec<_>>>()?;
        node = self.permute(node, order)?;
        let output_shape = Shape::new(
            right
                .iter()
                .map(|term| {
                    names_of(term).iter().try_fold(1usize, |p, name| {
                        p.checked_mul(dims[name])
                            .ok_or_else(|| Error::ShapeOverflow(source_shape.clone()))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        );
        self.reshape(node, output_shape)
    }

    /// NumPy/tinygrad repeat: left-aligns ranks, inserts broadcast axes, then
    /// reshapes.  Zero repetitions are legal and preserve dense dtype.
    pub fn repeat(&mut self, input: NodeId, repeats: &[isize]) -> Result<NodeId> {
        if repeats.is_empty() {
            return Err(Error::InvalidRepeat {
                reason: "at least one repetition is required",
            });
        }
        let repeats = repeats
            .iter()
            .map(|repeat| {
                usize::try_from(*repeat).map_err(|_| Error::InvalidRepeat {
                    reason: "repetitions must be non-negative",
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let source = self.node(input)?.shape.clone();
        let rank = source.rank().max(repeats.len());
        let mut base = vec![1; rank - source.rank()];
        base.extend_from_slice(source.dims());
        let mut normalized_repeats = vec![1; rank - repeats.len()];
        normalized_repeats.extend_from_slice(&repeats);
        let mut node = self.reshape(input, Shape::new(base.clone()))?;
        let mut final_shape = Vec::with_capacity(rank);
        for (axis, (&extent, &repeat)) in base.iter().zip(&normalized_repeats).enumerate() {
            if repeat != 1 {
                let mut shape = self.shape(node)?.dims().to_vec();
                shape.insert(axis, 1);
                node = self.reshape(node, Shape::new(shape))?;
                let mut expanded = self.shape(node)?.dims().to_vec();
                expanded[axis] = repeat;
                node = self.expand(node, Shape::new(expanded))?;
                let mut collapsed = self.shape(node)?.dims().to_vec();
                let merged = repeat
                    .checked_mul(extent)
                    .ok_or_else(|| Error::ShapeOverflow(source.clone()))?;
                collapsed[axis] = merged;
                collapsed.remove(axis + 1);
                node = self.reshape(node, Shape::new(collapsed))?;
            }
            final_shape.push(
                repeat
                    .checked_mul(extent)
                    .ok_or_else(|| Error::ShapeOverflow(source.clone()))?,
            );
        }
        debug_assert_eq!(self.shape(node)?.dims(), final_shape);
        Ok(node)
    }

    /// Alias for [`Graph::repeat`] with NumPy-style tile semantics.
    pub fn tile(&mut self, input: NodeId, repeats: &[isize]) -> Result<NodeId> {
        self.repeat(input, repeats)
    }

    /// Repeats every element `repeats` times along `axis`, or in flattened
    /// row-major order when `axis` is `None`.
    pub fn repeat_interleave(
        &mut self,
        input: NodeId,
        repeats: isize,
        axis: Option<isize>,
    ) -> Result<NodeId> {
        let repeats = usize::try_from(repeats).map_err(|_| Error::InvalidRepeat {
            reason: "repetitions must be non-negative",
        })?;
        let mut node = input;
        let mut shape = self.node(input)?.shape.clone();
        let axis = match axis {
            Some(axis) => resolve_axis(axis, shape.rank())?,
            None => {
                node = self.reshape(input, Shape::new(vec![shape.numel()?]))?;
                shape = self.shape(node)?.clone();
                0
            }
        };
        let extent = shape.dims()[axis];
        let mut inserted = shape.dims().to_vec();
        inserted.insert(axis + 1, 1);
        node = self.reshape(node, Shape::new(inserted))?;
        let mut expanded = self.shape(node)?.dims().to_vec();
        expanded[axis + 1] = repeats;
        node = self.expand(node, Shape::new(expanded))?;
        let mut output = self.shape(node)?.dims().to_vec();
        output[axis] = extent
            .checked_mul(repeats)
            .ok_or(Error::ShapeOverflow(shape))?;
        output.remove(axis + 1);
        self.reshape(node, Shape::new(output))
    }
}

fn rearrange_err(pattern: &str, reason: &'static str) -> Error {
    Error::InvalidRearrange {
        pattern: pattern.into(),
        reason,
    }
}
fn names_of(term: &Term) -> Vec<String> {
    match term {
        Term::Axis(name) => vec![name.clone()],
        Term::Group(names) => names.clone(),
        Term::Ellipsis => vec![],
    }
}
fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.ends_with('_')
}
fn parse_side(side: &str, pattern: &str) -> Result<Vec<Term>> {
    let chars = side.replace('…', "...").chars().collect::<Vec<_>>();
    let mut i = 0;
    let mut terms = Vec::new();
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        if chars[i] == '(' {
            i += 1;
            let mut names = Vec::new();
            loop {
                while i < chars.len() && chars[i].is_whitespace() {
                    i += 1;
                }
                if i >= chars.len() {
                    return Err(rearrange_err(pattern, "unclosed group"));
                }
                if chars[i] == ')' {
                    i += 1;
                    break;
                }
                let start = i;
                while i < chars.len()
                    && !chars[i].is_whitespace()
                    && chars[i] != ')'
                    && chars[i] != '('
                {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                if name != "..." && !valid_name(&name) {
                    return Err(rearrange_err(pattern, "invalid axis name"));
                }
                names.push(name);
            }
            terms.push(Term::Group(names));
        } else {
            let start = i;
            while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '(' && chars[i] != ')'
            {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            if text == "..." {
                terms.push(Term::Ellipsis);
            } else if text == "1" {
                terms.push(Term::Group(vec![]));
            } else if valid_name(&text) {
                terms.push(Term::Axis(text));
            } else {
                return Err(rearrange_err(pattern, "invalid axis name"));
            }
        }
    }
    Ok(terms)
}
fn expand_ellipsis(pattern: &RearrangePattern, rank: usize) -> Result<(Vec<Term>, Vec<Term>)> {
    let fixed = pattern.input.iter().filter(|t| !has_ellipsis(t)).count();
    if fixed > rank {
        return Err(rearrange_err(&pattern.text, "input rank is too small"));
    }
    let count = rank - fixed;
    let expanded = (0..count)
        .map(|i| Term::Axis(format!("@ellipsis{i}")))
        .collect::<Vec<_>>();
    let replace = |terms: &[Term]| {
        terms
            .iter()
            .flat_map(|term| match term {
                Term::Ellipsis => expanded.clone(),
                Term::Group(names) if names.iter().any(|name| name == "...") => vec![Term::Group(
                    names
                        .iter()
                        .flat_map(|name| {
                            if name == "..." {
                                (0..count)
                                    .map(|i| format!("@ellipsis{i}"))
                                    .collect::<Vec<_>>()
                            } else {
                                vec![name.clone()]
                            }
                        })
                        .collect(),
                )],
                _ => vec![term.clone()],
            })
            .collect::<Vec<_>>()
    };
    Ok((replace(&pattern.input), replace(&pattern.output)))
}
fn has_ellipsis(term: &Term) -> bool {
    matches!(term, Term::Ellipsis)
        || matches!(term, Term::Group(names) if names.iter().any(|name| name == "..."))
}
fn resolve_axis(axis: isize, rank: usize) -> Result<usize> {
    let axis = if axis < 0 {
        axis.checked_add(rank as isize).ok_or(Error::InvalidIndex)?
    } else {
        axis
    };
    usize::try_from(axis)
        .ok()
        .filter(|x| *x < rank)
        .ok_or(Error::InvalidIndex)
}

fn resolve_graph_axis(input: NodeId, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 {
        axis.checked_add(rank as isize)
    } else {
        Some(axis)
    };
    normalized
        .and_then(|axis| usize::try_from(axis).ok())
        .filter(|axis| *axis < rank)
        .ok_or(Error::InvalidAxis {
            node: input,
            axis: usize::try_from(axis).unwrap_or(usize::MAX),
            rank,
        })
}
