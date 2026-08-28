use crate::{DType, Error, Graph, NodeId, Result, Shape};
use std::collections::{BTreeMap, BTreeSet};

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
                (source, resolve_axis(axis, rank)?, false)
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
