use crate::{DType, Error, Graph, NodeId, Result, Shape};
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

/// One literal tinygrad `repeat` axis expansion.
#[derive(Clone, Debug)]
struct RepeatStage {
    axis: usize,
    unsqueezed: Shape,
    expanded: Shape,
    collapsed: Shape,
}

/// Fully describes `Tensor.repeat` before its first movement node exists.
///
/// tinygrad left-aligns the source and repeat ranks, then performs a
/// reshape/expand/reshape triple for every non-unit repeat.  Keeping every
/// descriptor here makes a later zero repeat unable to hide an earlier
/// overflowing expanded extent.
#[derive(Clone, Debug)]
struct RepeatPlan {
    base: Shape,
    stages: Vec<RepeatStage>,
    output: Shape,
}

impl RepeatPlan {
    fn build(source: &Shape, dtype: DType, repeats: &[isize]) -> Result<Self> {
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
        let rank = source.rank().max(repeats.len());
        let mut base = vec![1; rank - source.rank()];
        base.extend_from_slice(source.dims());
        let base = Shape::new(base);
        let mut normalized_repeats = vec![1; rank - repeats.len()];
        normalized_repeats.extend_from_slice(&repeats);

        let extent = |shape: &Shape| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                .map(|_| ())
        };
        // Source and left-aligned base are both concrete views which must be
        // valid before any reshape can be published.
        extent(source)?;
        extent(&base)?;

        let mut current = base.clone();
        let mut stages = Vec::new();
        for (axis, &repeat) in normalized_repeats.iter().enumerate() {
            if repeat == 1 {
                continue;
            }
            let mut unsqueezed = current.dims().to_vec();
            unsqueezed.insert(axis, 1);
            let unsqueezed = Shape::new(unsqueezed);
            let mut expanded = unsqueezed.dims().to_vec();
            expanded[axis] = repeat;
            let expanded = Shape::new(expanded);
            let mut collapsed = expanded.dims().to_vec();
            collapsed[axis] = current.dims()[axis]
                .checked_mul(repeat)
                .ok_or_else(|| Error::ShapeOverflow(current.clone()))?;
            collapsed.remove(axis + 1);
            let collapsed = Shape::new(collapsed);

            extent(&unsqueezed)?;
            extent(&expanded)?;
            extent(&collapsed)?;
            current = collapsed.clone();
            stages.push(RepeatStage {
                axis,
                unsqueezed,
                expanded,
                collapsed,
            });
        }
        // `current` is the literal final shape, including short/equal/long
        // rank alignment and zero repeats.
        extent(&current)?;
        Ok(Self {
            base,
            stages,
            output: current,
        })
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

/// Checked static lowering for tinygrad-style circular movement.
#[derive(Clone, Debug, Eq, PartialEq)]
struct StaticRollPlan {
    repeats: Vec<isize>,
    bounds: Vec<(usize, usize)>,
    zero_domain: bool,
}

impl StaticRollPlan {
    fn new(input: NodeId, shape: &Shape, shifts: &[isize], dims: &[isize]) -> Result<Self> {
        if shifts.len() != dims.len() {
            return Err(Error::InvalidRoll {
                reason: "shift and dimension counts must match",
            });
        }
        let mut axis_shifts = vec![None; shape.rank()];
        for (shift, axis) in shifts.iter().zip(dims) {
            axis_shifts[resolve_graph_axis(input, *axis, shape.rank())?] = Some(*shift);
        }
        if shape.dims().contains(&0) {
            return Ok(Self {
                repeats: vec![1; shape.rank()],
                bounds: shape.dims().iter().map(|extent| (0, *extent)).collect(),
                zero_domain: true,
            });
        }
        let mut repeats = vec![1isize; shape.rank()];
        let mut doubled = shape.dims().to_vec();
        let mut bounds = Vec::with_capacity(shape.rank());
        for (axis, (extent, shift)) in shape.dims().iter().zip(axis_shifts).enumerate() {
            let (start, end) = match shift {
                Some(shift) => {
                    let extent_signed = isize::try_from(*extent)
                        .map_err(|_| Error::ShapeOverflow(shape.clone()))?;
                    let normalized = shift.rem_euclid(extent_signed);
                    if normalized == 0 {
                        (0, *extent)
                    } else {
                        let start = *extent - usize::try_from(normalized).unwrap_or(0);
                        let end = start
                            .checked_add(*extent)
                            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                        repeats[axis] = 2;
                        doubled[axis] = extent
                            .checked_mul(2)
                            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                        (start, end)
                    }
                }
                None => (0, *extent),
            };
            bounds.push((start, end));
        }
        Shape::new(doubled).numel()?;
        Ok(Self {
            repeats,
            bounds,
            zero_domain: false,
        })
    }

    fn apply(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if self.zero_domain || !self.repeats.contains(&2) {
            return Ok(input);
        }
        let repeated = graph.repeat(input, &self.repeats)?;
        graph.shrink(repeated, self.bounds.clone())
    }
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
        if input.is_empty() || output.is_empty() {
            return Err(rearrange_err(
                pattern,
                "input and output sides must each name at least one axis",
            ));
        }
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
    /// Circularly shifts a tensor by static signed amounts. With `dims=None`,
    /// the row-major flattened tensor is shifted and reshaped back.
    pub fn roll_static(
        &mut self,
        input: NodeId,
        shifts: &[isize],
        dims: Option<&[isize]>,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        match dims {
            Some(dims) => StaticRollPlan::new(input, &shape, shifts, dims)?.apply(self, input),
            None => {
                if shifts.len() != 1 {
                    return Err(Error::InvalidRoll {
                        reason: "flattened roll requires exactly one shift",
                    });
                }
                let flattened_shape = Shape::new(vec![shape.numel()?]);
                let plan = StaticRollPlan::new(input, &flattened_shape, shifts, &[0])?;
                if plan.zero_domain {
                    return Ok(input);
                }
                let flattened = self.reshape(input, flattened_shape)?;
                let rolled = plan.apply(self, flattened)?;
                self.reshape(rolled, shape)
            }
        }
    }

    /// Splits an axis into static sections. A scalar size creates equal
    /// sections plus a smaller final section; explicit sections cover the
    /// axis exactly. Results are immutable `Shrink` views of `input`.
    pub fn split_static(
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
    pub fn chunk_static(
        &mut self,
        input: NodeId,
        chunks: usize,
        axis: isize,
    ) -> Result<Vec<NodeId>> {
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
        let source = self.node(input)?;
        let source_shape = source.shape.clone();
        let source_dtype = source.dtype;
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
        let order = right_names
            .iter()
            .map(|name| {
                elementary
                    .iter()
                    .position(|x| x == name)
                    .ok_or_else(|| rearrange_err(pattern, "axis mismatch"))
            })
            .collect::<Result<Vec<_>>>()?;
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
        let permuted_shape = Shape::new(
            order
                .iter()
                .map(|&axis| intermediate_shape.dims()[axis])
                .collect::<Vec<_>>(),
        );
        let extent = |shape: &Shape| {
            shape
                .numel()?
                .checked_mul(source_dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                .map(|_| ())
        };
        // Tinygrad's literal is unflatten → permute → flatten. Validate every
        // concrete view descriptor before the first reshape so malformed
        // factors and late merged-byte overflow cannot publish a prefix.
        extent(&source_shape)?;
        extent(&intermediate_shape)?;
        extent(&permuted_shape)?;
        extent(&output_shape)?;
        let node = self.reshape(input, intermediate_shape)?;
        let node = self.permute(node, order)?;
        self.reshape(node, output_shape)
    }

    /// NumPy/tinygrad repeat: left-aligns ranks, inserts broadcast axes, then
    /// reshapes.  Zero repetitions are legal and preserve dense dtype.
    pub fn repeat(&mut self, input: NodeId, repeats: &[isize]) -> Result<NodeId> {
        let source = self.node(input)?;
        let plan = RepeatPlan::build(&source.shape, source.dtype, repeats)?;

        let mut node = self.reshape(input, plan.base)?;
        for stage in plan.stages {
            debug_assert!(stage.axis < self.shape(node)?.rank());
            node = self.reshape(node, stage.unsqueezed)?;
            node = self.expand(node, stage.expanded)?;
            node = self.reshape(node, stage.collapsed)?;
        }
        debug_assert_eq!(self.shape(node)?, &plan.output);
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
        let source_node = self.node(input)?;
        let source = source_node.shape.clone();
        let dtype = source_node.dtype;
        let (shape, axis, flatten) = match axis {
            Some(axis) => {
                source.numel()?;
                let rank = source.rank();
                (source.clone(), resolve_axis(axis, rank)?, false)
            }
            None => {
                let flat = source.numel()?;
                (Shape::new(vec![flat]), 0, true)
            }
        };
        let extent = shape.dims()[axis];
        let output_extent = extent
            .checked_mul(repeats)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut output = shape.dims().to_vec();
        output[axis] = output_extent;
        let output_shape = Shape::new(output.clone());
        let mut inserted = shape.dims().to_vec();
        inserted.insert(axis + 1, 1);
        let inserted_shape = Shape::new(inserted);
        let mut expanded = inserted_shape.dims().to_vec();
        expanded[axis + 1] = repeats;
        let expanded_shape = Shape::new(expanded);
        let extent = |candidate: &Shape| {
            candidate
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(candidate.clone()))
                .map(|_| ())
        };
        // Validate the source, optional flatten view, inserted singleton,
        // Expand result, and final collapse before publishing any movement.
        extent(&source)?;
        extent(&shape)?;
        extent(&inserted_shape)?;
        extent(&expanded_shape)?;
        extent(&output_shape)?;

        let mut node = if flatten {
            self.reshape(input, shape.clone())?
        } else {
            input
        };
        node = self.reshape(node, inserted_shape)?;
        node = self.expand(node, expanded_shape)?;
        self.reshape(node, output_shape)
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
    // tinygrad uses Python's `str.isidentifier()` (then excludes leading and
    // trailing underscores).  Rust's Unicode character predicates cover the
    // source-tested letter/digit/inner-underscore subset without changing the
    // tokenization or static lowering contract.
    matches!(chars.next(), Some(c) if c.is_alphabetic())
        && chars.all(|c| c.is_alphanumeric() || c == '_')
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Error};

    #[test]
    fn rearrange_matches_concrete_tinygrad_terms_and_preflights_every_view() {
        let mut graph = Graph::new();
        let input = graph.input_dtype_requires_grad("x", [2, 1, 6], DType::F32, true);
        let mut sizes = BTreeMap::new();
        sizes.insert("h".into(), 2);
        let output = graph
            .rearrange(input, "b 1 (h w) -> w b h", &sizes)
            .unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([3, 2, 2]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        let loss = graph.sum_all(output).unwrap();
        assert!(graph.grad(loss, input).is_ok());

        let ellipsis = graph.input_dtype("ellipsis", [2, 0, 3], DType::I16);
        let ellipsis_output = graph
            .rearrange(ellipsis, "b ... c -> ... b c", &BTreeMap::new())
            .unwrap();
        assert_eq!(
            graph.shape(ellipsis_output).unwrap(),
            &Shape::new([0, 2, 3])
        );
        assert_eq!(graph.dtype(ellipsis_output).unwrap(), DType::I16);

        let mut malformed = Graph::new();
        let source = malformed.input_dtype("source", [2, 6], DType::F64);
        let before = malformed.node_count();
        assert!(malformed
            .rearrange(source, "b (h w) -> h b w", &BTreeMap::new())
            .is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed
            .rearrange(source, "b b -> b", &BTreeMap::new())
            .is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed
            .rearrange(source, "b c -> (b c)", &{
                let mut unused = BTreeMap::new();
                unused.insert("unused".into(), 1);
                unused
            })
            .is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(matches!(
            malformed.rearrange(
                NodeId::from_index(usize::MAX),
                "b c -> c b",
                &BTreeMap::new()
            ),
            Err(Error::UnknownNode(_))
        ));
        assert_eq!(malformed.node_count(), before);

        let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
        let overflow_before = malformed.node_count();
        assert!(matches!(
            malformed.rearrange(overflow, "b c -> (b c)", &BTreeMap::new()),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), overflow_before);
    }
}
