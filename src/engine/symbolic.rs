//! Portable symbolic shape schemas and graph-independent specialization.
//!
//! A captured symbolic artifact retains one validated concrete template only as
//! structural evidence. Replay always evaluates this schema into a fresh
//! concrete schedule before allocation, compilation, or execution.
use super::capture::{CapturedSchedule, ReplayError};
use super::symbolic_view::SymbolicViewMap;
use crate::{
    BufferDesc, DType, Graph, IndexValue, MatmulValue, NodeId, Op, Operation, ReductionValue,
    Schedule, Shape, SymbolicDim, SymbolicExpr, SymbolicShape, SymbolicVar, UArgRef, UOp, UOpKind,
};
use std::collections::{BTreeMap, BTreeSet};

/// One portable scalar parameter. Shape parameters are signed 64-bit values;
/// their inclusive domain is owned by the stable [`SymbolicVar`] identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolicParameter {
    pub(crate) variable: SymbolicVar,
    pub(crate) dtype: DType,
}
impl SymbolicParameter {
    pub fn variable(&self) -> &SymbolicVar {
        &self.variable
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// A fail-closed condition evaluated after range validation and before shape
/// arithmetic. Guards are serialized as typed expression trees, never text.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SymbolicGuard {
    Equal {
        left: SymbolicExpr,
        right: SymbolicExpr,
    },
    Divisible {
        value: SymbolicExpr,
        divisor: u64,
    },
}
impl SymbolicGuard {
    pub fn equal(left: SymbolicExpr, right: SymbolicExpr) -> Self {
        Self::Equal { left, right }
    }
    pub fn divisible(value: SymbolicExpr, divisor: u64) -> Result<Self, ReplayError> {
        if divisor == 0 || divisor > i64::MAX as u64 {
            return Err(ReplayError::Symbolic(
                "symbolic divisibility divisor must be in 1..=i64::MAX".into(),
            ));
        }
        Ok(Self::Divisible { value, divisor })
    }
    pub(crate) fn expressions(&self) -> Vec<&SymbolicExpr> {
        match self {
            Self::Equal { left, right } => vec![left, right],
            Self::Divisible { value, .. } => vec![value],
        }
    }
}

/// Capture-time symbolic intent. Input shapes are keyed by graph input IDs;
/// all downstream shapes and item domains are derived from graph semantics.
#[derive(Clone, Debug, Default)]
pub struct SymbolicCaptureSpec {
    input_shapes: BTreeMap<NodeId, SymbolicShape>,
    constant_shapes: BTreeMap<NodeId, SymbolicShape>,
    guards: Vec<SymbolicGuard>,
}
impl SymbolicCaptureSpec {
    pub fn new(input_shapes: BTreeMap<NodeId, SymbolicShape>) -> Self {
        Self {
            input_shapes,
            constant_shapes: BTreeMap::new(),
            guards: Vec::new(),
        }
    }
    /// Adds a symbolic shape for a constant whose template storage is one
    /// exact repeated scalar bit pattern. Specialization may resize only by
    /// repeating that pattern.
    pub fn with_constant_shape(mut self, node: NodeId, shape: SymbolicShape) -> Self {
        self.constant_shapes.insert(node, shape);
        self
    }
    pub fn with_guard(mut self, guard: SymbolicGuard) -> Self {
        self.guards.push(guard);
        self
    }
    pub fn input_shapes(&self) -> &BTreeMap<NodeId, SymbolicShape> {
        &self.input_shapes
    }
    pub fn guards(&self) -> &[SymbolicGuard] {
        &self.guards
    }
    pub fn constant_shapes(&self) -> &BTreeMap<NodeId, SymbolicShape> {
        &self.constant_shapes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SymbolicItemDomain {
    Elementwise {
        output: SymbolicShape,
    },
    Reduction {
        input_buffer: u64,
        input: SymbolicShape,
        output: SymbolicShape,
        reduction: SymbolicShape,
    },
    Matmul {
        lhs_buffer: u64,
        rhs_buffer: u64,
        output: SymbolicShape,
        batch: SymbolicShape,
        m: SymbolicExpr,
        n: SymbolicExpr,
        k: SymbolicExpr,
    },
}
impl SymbolicItemDomain {
    pub(crate) fn expressions(&self) -> Vec<&SymbolicExpr> {
        let mut out = Vec::new();
        match self {
            Self::Elementwise { output } => {
                out.extend(output.dims().iter().map(SymbolicDim::expression));
            }
            Self::Reduction {
                input,
                output,
                reduction,
                ..
            } => {
                out.extend(input.dims().iter().map(SymbolicDim::expression));
                out.extend(output.dims().iter().map(SymbolicDim::expression));
                out.extend(reduction.dims().iter().map(SymbolicDim::expression));
            }
            Self::Matmul {
                output,
                batch,
                m,
                n,
                k,
                ..
            } => {
                out.extend(output.dims().iter().map(SymbolicDim::expression));
                out.extend(batch.dims().iter().map(SymbolicDim::expression));
                out.extend([m, n, k]);
            }
        }
        out
    }
}

/// Crate-owned portable schema. Construction and artifact decoding both route
/// through `validate_against`, so replay never sees an unchecked expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SymbolicSchema {
    pub(crate) parameters: Vec<SymbolicParameter>,
    pub(crate) template_values: Vec<i64>,
    pub(crate) guards: Vec<SymbolicGuard>,
    pub(crate) buffer_shapes: BTreeMap<u64, SymbolicShape>,
    pub(crate) item_domains: BTreeMap<u64, SymbolicItemDomain>,
    pub(crate) views: BTreeMap<(u64, u64), SymbolicViewMap>,
    pub(crate) splat_constants: BTreeSet<u64>,
}

/// Provenance carried by a concrete specialization. It deliberately contains
/// values and artifact identities, not process-specific compiled resources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SpecializedFrom {
    pub(crate) source_identity: u64,
    pub(crate) bindings: Vec<(u64, i64)>,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundDomain {
    pub output: Shape,
    pub reduction: Option<(Shape, Shape, u64)>,
    pub matmul: Option<(Shape, usize, usize, usize, u64, u64)>,
}

impl SymbolicSchema {
    pub(crate) fn parameters(&self) -> &[SymbolicParameter] {
        &self.parameters
    }
    pub(crate) fn guards(&self) -> &[SymbolicGuard] {
        &self.guards
    }
    pub(crate) fn canonical_bindings(
        &self,
        named: &BTreeMap<String, i64>,
    ) -> Result<Vec<(u64, i64)>, ReplayError> {
        let expected = self
            .parameters
            .iter()
            .map(|parameter| parameter.variable.name())
            .collect::<BTreeSet<_>>();
        if let Some(extra) = named.keys().find(|name| !expected.contains(name.as_str())) {
            return Err(ReplayError::Extra(extra.clone()));
        }
        let mut canonical = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            let variable = &parameter.variable;
            let value = *named
                .get(variable.name())
                .ok_or_else(|| ReplayError::Missing(variable.name().into()))?;
            let (min, max) = variable.bounds();
            if value < min || value > max {
                return Err(ReplayError::Symbolic(format!(
                    "binding {}={} is outside [{min}, {max}]",
                    variable.name(),
                    value
                )));
            }
            canonical.push((variable.id(), value));
        }
        let environment = self.environment(&canonical)?;
        self.validate_guards(&environment)?;
        Ok(canonical)
    }

    fn environment(
        &self,
        canonical: &[(u64, i64)],
    ) -> Result<BTreeMap<SymbolicVar, i64>, ReplayError> {
        if canonical.len() != self.parameters.len() {
            return Err(ReplayError::Symbolic(
                "symbolic binding count mismatch".into(),
            ));
        }
        self.parameters
            .iter()
            .zip(canonical)
            .map(|(parameter, (id, value))| {
                if parameter.variable.id() != *id {
                    return Err(ReplayError::Symbolic(
                        "symbolic binding identity mismatch".into(),
                    ));
                }
                Ok((parameter.variable.clone(), *value))
            })
            .collect()
    }

    fn validate_guards(&self, environment: &BTreeMap<SymbolicVar, i64>) -> Result<(), ReplayError> {
        for guard in &self.guards {
            match guard {
                SymbolicGuard::Equal { left, right } => {
                    if evaluate(left, environment)? != evaluate(right, environment)? {
                        return Err(ReplayError::Symbolic(
                            "symbolic equality guard failed".into(),
                        ));
                    }
                }
                SymbolicGuard::Divisible { value, divisor } => {
                    let divisor = i64::try_from(*divisor).map_err(|_| {
                        ReplayError::Symbolic("symbolic divisor does not fit i64".into())
                    })?;
                    if divisor == 0 || evaluate(value, environment)? % divisor != 0 {
                        return Err(ReplayError::Symbolic(
                            "symbolic divisibility guard failed".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn bind_shape(
        &self,
        buffer: u64,
        environment: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<Shape, ReplayError> {
        let shape = self
            .buffer_shapes
            .get(&buffer)
            .ok_or_else(|| ReplayError::Symbolic(format!("missing symbolic buffer {buffer}")))?;
        bind_shape(shape, environment)
    }

    pub(crate) fn bind_domain(
        &self,
        item: u64,
        environment: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<BoundDomain, ReplayError> {
        match self
            .item_domains
            .get(&item)
            .ok_or_else(|| ReplayError::Symbolic(format!("missing symbolic item {item}")))?
        {
            SymbolicItemDomain::Elementwise { output } => Ok(BoundDomain {
                output: bind_shape(output, environment)?,
                reduction: None,
                matmul: None,
            }),
            SymbolicItemDomain::Reduction {
                input_buffer,
                input,
                output,
                reduction,
            } => Ok(BoundDomain {
                output: bind_shape(output, environment)?,
                reduction: Some((
                    bind_shape(input, environment)?,
                    bind_shape(reduction, environment)?,
                    *input_buffer,
                )),
                matmul: None,
            }),
            SymbolicItemDomain::Matmul {
                lhs_buffer,
                rhs_buffer,
                output,
                batch,
                m,
                n,
                k,
            } => Ok(BoundDomain {
                output: bind_shape(output, environment)?,
                reduction: None,
                matmul: Some((
                    bind_shape(batch, environment)?,
                    evaluate_usize(m, environment)?,
                    evaluate_usize(n, environment)?,
                    evaluate_usize(k, environment)?,
                    *lhs_buffer,
                    *rhs_buffer,
                )),
            }),
        }
    }

    pub(crate) fn template_environment(&self) -> Result<BTreeMap<SymbolicVar, i64>, ReplayError> {
        let canonical = self
            .parameters
            .iter()
            .zip(&self.template_values)
            .map(|(parameter, value)| (parameter.variable.id(), *value))
            .collect::<Vec<_>>();
        self.environment(&canonical)
    }
}

pub(crate) fn build_schema(
    graph: &Graph,
    schedule: &Schedule,
    capture: &CapturedSchedule,
    spec: &SymbolicCaptureSpec,
    template_bindings: &BTreeMap<String, i64>,
) -> Result<SymbolicSchema, ReplayError> {
    if schedule.items.iter().any(|item| !item.outputs.is_single()) {
        return Err(ReplayError::Unsupported(
            "symbolic capture has no multi-output schedule specialization".into(),
        ));
    }
    if spec.input_shapes.is_empty() && spec.constant_shapes.is_empty() {
        return Err(ReplayError::Symbolic(
            "symbolic capture requires at least one symbolic input or constant shape".into(),
        ));
    }
    for node in spec.input_shapes.keys() {
        if !matches!(graph.op(*node), Ok(Op::Input { .. })) {
            return Err(ReplayError::Symbolic(
                "symbolic input shape does not name a graph input".into(),
            ));
        }
    }
    for node in spec.constant_shapes.keys() {
        if !matches!(graph.op(*node), Ok(Op::Constant(_))) {
            return Err(ReplayError::Symbolic(
                "symbolic constant shape does not name a graph constant".into(),
            ));
        }
    }
    if spec
        .input_shapes
        .keys()
        .any(|node| spec.constant_shapes.contains_key(node))
    {
        return Err(ReplayError::Symbolic(
            "one symbolic seed has conflicting input and constant roles".into(),
        ));
    }
    let seeds = spec
        .input_shapes
        .iter()
        .chain(&spec.constant_shapes)
        .map(|(node, shape)| (*node, shape.clone()))
        .collect::<BTreeMap<_, _>>();
    let seed_expressions = seeds
        .values()
        .flat_map(|shape| shape.dims().iter().map(SymbolicDim::expression))
        .collect::<Vec<_>>();
    let seed_parameters = collect_parameters(&seed_expressions)?;
    let template_environment = seed_parameters
        .iter()
        .map(|parameter| {
            let value = template_bindings
                .get(parameter.variable.name())
                .copied()
                .ok_or_else(|| ReplayError::Missing(parameter.variable.name().into()))?;
            let (min, max) = parameter.variable.bounds();
            if value < min || value > max {
                return Err(ReplayError::Symbolic(format!(
                    "template binding {}={value} is outside [{min}, {max}]",
                    parameter.variable.name()
                )));
            }
            Ok((parameter.variable.clone(), value))
        })
        .collect::<Result<BTreeMap<_, _>, ReplayError>>()?;
    let movement_candidates = super::symbolic_view::candidates(seeds.values().cloned());
    let mut memo = BTreeMap::new();
    let mut guards = spec.guards.clone();
    let mut relevant = BTreeSet::new();
    for item in &schedule.items {
        relevant.insert(item.node);
        relevant.extend(item.input_bindings.iter().map(|binding| binding.input_node));
    }
    for node in relevant.iter().copied() {
        derive_shape(
            graph,
            node,
            &seeds,
            &movement_candidates,
            &template_environment,
            &mut memo,
            &mut guards,
        )?;
    }
    if seeds.keys().any(|node| !memo.contains_key(node)) {
        return Err(ReplayError::Symbolic(
            "symbolic input or constant shape is unused by the captured schedule".into(),
        ));
    }
    guards.sort();
    guards.dedup();

    let mut buffer_shapes = BTreeMap::new();
    for item in &schedule.items {
        for binding in &item.input_bindings {
            buffer_shapes.insert(
                binding.desc.id,
                memo.get(&binding.input_node)
                    .ok_or_else(|| ReplayError::Symbolic("missing derived input shape".into()))?
                    .clone(),
            );
        }
        buffer_shapes.insert(
            item.primary_output().id,
            memo.get(&item.node)
                .ok_or_else(|| ReplayError::Symbolic("missing derived output shape".into()))?
                .clone(),
        );
    }

    let mut item_domains = BTreeMap::new();
    for item in &schedule.items {
        let output = memo[&item.node].clone();
        let domain = match graph
            .op(item.node)
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?
        {
            Op::Reduce {
                input,
                axes,
                keepdim: _,
                ..
            } => {
                let input_shape = memo[input].clone();
                let reduction = SymbolicShape::new(
                    axes.iter()
                        .map(|axis| input_shape.dims()[*axis].clone())
                        .collect::<Vec<_>>(),
                );
                SymbolicItemDomain::Reduction {
                    input_buffer: input.index() as u64,
                    input: input_shape,
                    output,
                    reduction,
                }
            }
            Op::Matmul { lhs, rhs } => {
                let lhs_shape = &memo[lhs];
                let rhs_shape = &memo[rhs];
                let geometry = matmul_geometry(lhs_shape, rhs_shape, &mut guards)?;
                SymbolicItemDomain::Matmul {
                    lhs_buffer: lhs.index() as u64,
                    rhs_buffer: rhs.index() as u64,
                    output,
                    batch: SymbolicShape::new(geometry.batch),
                    m: geometry.m,
                    n: geometry.n,
                    k: geometry.k,
                }
            }
            _ => SymbolicItemDomain::Elementwise { output },
        };
        item_domains.insert(item.id, domain);
    }
    guards.sort();
    guards.dedup();

    let mut views = BTreeMap::new();
    for item in &schedule.items {
        let mut lowered_views = BTreeSet::new();
        collect_lowered_view_nodes(graph, item.node, &mut lowered_views)?;
        for node in item
            .kernel
            .topological()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?
        {
            let UArgRef::ViewBufferIndex { buffer, view, .. } = node.arg() else {
                continue;
            };
            let mut matches = Vec::new();
            for candidate in lowered_views.iter().copied() {
                let Ok((source, symbolic)) = super::symbolic_view::derive_view(
                    graph,
                    candidate,
                    &memo,
                    &template_environment,
                ) else {
                    continue;
                };
                if source.index() as u64 == *buffer
                    && crate::AffineView::from(symbolic.specialize(&template_environment)?) == *view
                    && !matches.contains(&symbolic)
                {
                    matches.push(symbolic);
                }
            }
            let [symbolic] = matches.as_slice() else {
                return Err(ReplayError::Unsupported(format!(
                    "captured affine view has {} symbolic matches",
                    matches.len()
                )));
            };
            let key = (item.id, *buffer);
            if views.insert(key, symbolic.clone()).is_some() {
                return Err(ReplayError::Unsupported(
                    "one schedule item uses multiple views of one source buffer".into(),
                ));
            }
        }
    }

    let splat_constants = spec
        .constant_shapes
        .keys()
        .map(|node| node.index() as u64)
        .collect::<BTreeSet<_>>();
    for buffer in &splat_constants {
        let value = capture.constants.get(buffer).ok_or_else(|| {
            ReplayError::Symbolic("symbolic constant is absent from captured storage".into())
        })?;
        if value.storage().repeat_exact_splat(value.len()).is_none() {
            return Err(ReplayError::Unsupported(
                "symbolic constant storage is not one exact repeated scalar".into(),
            ));
        }
    }

    let expressions = buffer_shapes
        .values()
        .flat_map(|shape| shape.dims().iter().map(SymbolicDim::expression))
        .chain(guards.iter().flat_map(|guard| guard.expressions()))
        .chain(
            item_domains
                .values()
                .flat_map(SymbolicItemDomain::expressions),
        )
        .chain(views.values().flat_map(SymbolicViewMap::expressions))
        .collect::<Vec<_>>();
    let parameters = collect_parameters(&expressions)?;
    let expected = parameters
        .iter()
        .map(|parameter| parameter.variable.name())
        .collect::<BTreeSet<_>>();
    if let Some(extra) = template_bindings
        .keys()
        .find(|name| !expected.contains(name.as_str()))
    {
        return Err(ReplayError::Extra(extra.clone()));
    }
    let template_values = parameters
        .iter()
        .map(|parameter| {
            template_bindings
                .get(parameter.variable.name())
                .copied()
                .ok_or_else(|| ReplayError::Missing(parameter.variable.name().into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let schema = SymbolicSchema {
        parameters,
        template_values,
        guards,
        buffer_shapes,
        item_domains,
        views,
        splat_constants,
    };
    schema.validate_against(capture)?;
    Ok(schema)
}

impl SymbolicSchema {
    pub(crate) fn validate_against(&self, capture: &CapturedSchedule) -> Result<(), ReplayError> {
        if self.parameters.is_empty()
            || self.parameters.len() != self.template_values.len()
            || self
                .parameters
                .windows(2)
                .any(|pair| pair[0].variable.id() >= pair[1].variable.id())
        {
            return Err(ReplayError::Symbolic(
                "symbolic parameter table is malformed".into(),
            ));
        }
        if self.guards.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ReplayError::Symbolic(
                "symbolic guard table is not canonical".into(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut ids = BTreeMap::new();
        for parameter in &self.parameters {
            let variable = &parameter.variable;
            let (min, max) = variable.bounds();
            if parameter.dtype != DType::I64
                || variable.id() == 0
                || variable.name().is_empty()
                || min > max
                || !names.insert(variable.name())
                || ids.insert(variable.id(), variable.clone()).is_some()
            {
                return Err(ReplayError::Symbolic(
                    "symbolic parameter identity is malformed".into(),
                ));
            }
        }
        let mut expressions = self
            .buffer_shapes
            .values()
            .flat_map(|shape| shape.dims().iter().map(SymbolicDim::expression))
            .chain(self.guards.iter().flat_map(SymbolicGuard::expressions))
            .chain(
                self.item_domains
                    .values()
                    .flat_map(SymbolicItemDomain::expressions),
            )
            .chain(self.views.values().flat_map(SymbolicViewMap::expressions));
        for expression in &mut expressions {
            expression
                .bounds()
                .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
            for variable in expression.variables() {
                if ids.get(&variable.id()) != Some(&variable) {
                    return Err(ReplayError::Symbolic(
                        "symbolic expression references an unknown parameter".into(),
                    ));
                }
            }
        }
        for guard in &self.guards {
            if matches!(guard, SymbolicGuard::Divisible { divisor: 0, .. }) {
                return Err(ReplayError::Symbolic(
                    "symbolic divisibility guard has a zero divisor".into(),
                ));
            }
        }
        let expected_buffers = capture
            .items
            .iter()
            .flat_map(|item| {
                item.input_bindings
                    .iter()
                    .map(|binding| binding.desc.id)
                    .chain(std::iter::once(item.primary_output().id))
            })
            .collect::<BTreeSet<_>>();
        if self.buffer_shapes.keys().copied().collect::<BTreeSet<_>>() != expected_buffers {
            return Err(ReplayError::Symbolic(
                "symbolic buffer-shape coverage is incomplete".into(),
            ));
        }
        if self.item_domains.keys().copied().collect::<BTreeSet<_>>()
            != capture.items.iter().map(|item| item.id).collect()
        {
            return Err(ReplayError::Symbolic(
                "symbolic item-domain coverage is incomplete".into(),
            ));
        }
        let environment = self.template_environment()?;
        self.validate_guards(&environment)?;
        for shape in self.buffer_shapes.values() {
            validate_shape_bounds(shape)?;
        }
        for desc in capture.items.iter().flat_map(|item| {
            item.inputs
                .iter()
                .chain(std::iter::once(item.primary_output()))
        }) {
            let elements = self
                .buffer_shapes
                .get(&desc.id)
                .ok_or_else(|| ReplayError::Symbolic("symbolic buffer shape is absent".into()))?
                .numel()
                .map_err(|error| ReplayError::Symbolic(error.to_string()))?
                .bounds()
                .map_err(|error| ReplayError::Symbolic(error.to_string()))?
                .max;
            usize::try_from(elements)
                .ok()
                .and_then(|elements| elements.checked_mul(desc.dtype.itemsize()))
                .ok_or_else(|| {
                    ReplayError::Symbolic("symbolic buffer byte extent overflows".into())
                })?;
        }
        for view in self.views.values() {
            view.validate_bounds()?;
        }
        let mut expected_views = BTreeSet::new();
        for item in &capture.items {
            for node in item
                .kernel
                .topological()
                .map_err(|error| ReplayError::Symbolic(error.to_string()))?
            {
                if let UArgRef::ViewBufferIndex { buffer, .. } = node.arg() {
                    expected_views.insert((item.id, *buffer));
                }
            }
        }
        if self.views.keys().copied().collect::<BTreeSet<_>>() != expected_views {
            return Err(ReplayError::Symbolic(
                "symbolic view coverage is incomplete".into(),
            ));
        }
        for ((item_id, buffer), symbolic) in &self.views {
            let item = capture
                .items
                .iter()
                .find(|item| item.id == *item_id)
                .ok_or_else(|| ReplayError::Symbolic("symbolic view item is absent".into()))?;
            let descriptor = item
                .input_bindings
                .iter()
                .find(|binding| binding.desc.id == *buffer)
                .ok_or_else(|| ReplayError::Symbolic("symbolic view buffer is absent".into()))?;
            let concrete = crate::AffineView::from(symbolic.specialize(&environment)?);
            if self.buffer_shapes.get(buffer) != Some(&symbolic.source_shape)
                || descriptor.desc.view.as_ref() != Some(&concrete)
            {
                return Err(ReplayError::Symbolic(
                    "symbolic view source or template descriptor is inconsistent".into(),
                ));
            }
        }
        for item in &capture.items {
            for desc in item
                .inputs
                .iter()
                .chain(std::iter::once(item.primary_output()))
            {
                if self.bind_shape(desc.id, &environment)? != desc.shape {
                    return Err(ReplayError::Symbolic(
                        "symbolic template descriptor does not match its binding".into(),
                    ));
                }
            }
            let symbolic_domain = self
                .item_domains
                .get(&item.id)
                .ok_or_else(|| ReplayError::Symbolic("symbolic item domain is absent".into()))?;
            self.validate_item_domain(item, symbolic_domain)?;
            let domain = self.bind_domain(item.id, &environment)?;
            if domain.output != item.primary_output().shape {
                return Err(ReplayError::Symbolic(
                    "symbolic template item domain does not match output".into(),
                ));
            }
            if specialize_kernel(
                &item.kernel,
                self,
                &environment,
                &domain,
                item.id,
                item.primary_output().id,
            )? != item.kernel
            {
                return Err(ReplayError::Symbolic(
                    "symbolic template UOp does not match its expressions".into(),
                ));
            }
        }
        for (id, value) in &capture.constants {
            let shape = self
                .buffer_shapes
                .get(id)
                .ok_or_else(|| ReplayError::Symbolic("constant shape is absent".into()))?;
            let dynamic = shape
                .dims()
                .iter()
                .any(|dim| !dim.expression().variables().is_empty());
            if bind_shape(shape, &environment)? != *value.shape()
                || dynamic
                    && (!self.splat_constants.contains(id)
                        || value.storage().repeat_exact_splat(value.len()).is_none())
            {
                return Err(ReplayError::Unsupported(
                    "symbolically resized constant is not an exact splat".into(),
                ));
            }
        }
        if self
            .splat_constants
            .iter()
            .any(|id| !capture.constants.contains_key(id))
        {
            return Err(ReplayError::Symbolic(
                "symbolic splat constant table references unknown storage".into(),
            ));
        }
        Ok(())
    }

    fn validate_item_domain(
        &self,
        item: &crate::ScheduleItem,
        domain: &SymbolicItemDomain,
    ) -> Result<(), ReplayError> {
        let output_shape = self
            .buffer_shapes
            .get(&item.primary_output().id)
            .ok_or_else(|| {
                ReplayError::Symbolic("symbolic output buffer shape is absent".into())
            })?;
        match domain {
            SymbolicItemDomain::Elementwise { output } => {
                if output != output_shape
                    || item
                        .kernel
                        .topological()
                        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
                        .iter()
                        .any(|node| {
                            matches!(
                                node.arg(),
                                UArgRef::Reduction { .. }
                                    | UArgRef::Matmul(_)
                                    | UArgRef::TiledMatmul(_)
                                    | UArgRef::TensorCoreMatmul(_)
                            )
                        })
                {
                    return Err(ReplayError::Symbolic(
                        "symbolic elementwise domain is inconsistent".into(),
                    ));
                }
            }
            SymbolicItemDomain::Reduction {
                input_buffer,
                input,
                output,
                reduction,
            } => {
                if output != output_shape
                    || self.buffer_shapes.get(input_buffer) != Some(input)
                    || !item
                        .input_bindings
                        .iter()
                        .any(|binding| binding.desc.id == *input_buffer)
                {
                    return Err(ReplayError::Symbolic(
                        "symbolic reduction buffers are inconsistent".into(),
                    ));
                }
                let payloads = item
                    .kernel
                    .topological()
                    .map_err(|error| ReplayError::Symbolic(error.to_string()))?
                    .into_iter()
                    .filter_map(|node| match node.arg() {
                        UArgRef::Reduction { axes, keepdim, .. } => Some((axes.to_vec(), *keepdim)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let [(axes, keepdim)] = payloads.as_slice() else {
                    return Err(ReplayError::Symbolic(
                        "symbolic reduction payload count is inconsistent".into(),
                    ));
                };
                let expected_reduction = SymbolicShape::new(
                    axes.iter()
                        .map(|axis| input.dims()[*axis].clone())
                        .collect::<Vec<_>>(),
                );
                let mut expected_output = input.dims().to_vec();
                if *keepdim {
                    for axis in axes {
                        expected_output[*axis] = 1usize.into();
                    }
                } else {
                    for axis in axes.iter().rev() {
                        expected_output.remove(*axis);
                    }
                }
                if reduction != &expected_reduction
                    || output != &SymbolicShape::new(expected_output)
                {
                    return Err(ReplayError::Symbolic(
                        "symbolic reduction geometry is inconsistent".into(),
                    ));
                }
            }
            SymbolicItemDomain::Matmul {
                lhs_buffer,
                rhs_buffer,
                output,
                batch,
                m,
                n,
                k,
            } => {
                let Some(plan) = item.kernel.arg().matmul_plan() else {
                    return Err(ReplayError::Symbolic(
                        "symbolic matmul payload is absent".into(),
                    ));
                };
                let lhs = self.buffer_shapes.get(lhs_buffer).ok_or_else(|| {
                    ReplayError::Symbolic("symbolic matmul lhs shape is absent".into())
                })?;
                let rhs = self.buffer_shapes.get(rhs_buffer).ok_or_else(|| {
                    ReplayError::Symbolic("symbolic matmul rhs shape is absent".into())
                })?;
                let mut required_guards = Vec::new();
                let geometry = matmul_geometry(lhs, rhs, &mut required_guards)?;
                if plan.lhs.index() as u64 != *lhs_buffer
                    || plan.rhs.index() as u64 != *rhs_buffer
                    || output != output_shape
                    || output.dims() != geometry.output
                    || batch.dims() != geometry.batch
                    || m != &geometry.m
                    || n != &geometry.n
                    || k != &geometry.k
                    || required_guards
                        .iter()
                        .any(|guard| !self.guards.contains(guard))
                {
                    return Err(ReplayError::Symbolic(
                        "symbolic matmul geometry is inconsistent".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn collect_lowered_view_nodes(
    graph: &Graph,
    node: NodeId,
    views: &mut BTreeSet<NodeId>,
) -> Result<(), ReplayError> {
    match graph
        .op(node)
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
    {
        Op::Shrink { .. }
        | Op::Reshape { .. }
        | Op::Permute { .. }
        | Op::Expand { .. }
        | Op::Stride { .. } => {
            views.insert(node);
        }
        Op::Cast { input, .. }
        | Op::Bitcast { input, .. }
        | Op::Contiguous { input }
        | Op::ContiguousBackward { input }
        | Op::Detach { input }
        | Op::Unary { input, .. }
        | Op::Reduce { input, .. } => collect_lowered_view_nodes(graph, *input, views)?,
        Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } | Op::Matmul { lhs, rhs } => {
            collect_lowered_view_nodes(graph, *lhs, views)?;
            collect_lowered_view_nodes(graph, *rhs, views)?;
        }
        Op::Logical { lhs, rhs, .. } => {
            collect_lowered_view_nodes(graph, *lhs, views)?;
            if let Some(rhs) = rhs {
                collect_lowered_view_nodes(graph, *rhs, views)?;
            }
        }
        Op::Select {
            condition,
            on_true,
            on_false,
        } => {
            collect_lowered_view_nodes(graph, *condition, views)?;
            collect_lowered_view_nodes(graph, *on_true, views)?;
            collect_lowered_view_nodes(graph, *on_false, views)?;
        }
        _ => {}
    }
    Ok(())
}

fn collect_parameters(
    expressions: &[&SymbolicExpr],
) -> Result<Vec<SymbolicParameter>, ReplayError> {
    let mut variables = BTreeMap::new();
    let mut names = BTreeSet::new();
    for expression in expressions {
        expression
            .bounds()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
        for variable in expression.variables() {
            if let Some(old) = variables.insert(variable.id(), variable.clone())
                && old != variable
            {
                return Err(ReplayError::Symbolic(
                    "one symbolic ID has conflicting metadata".into(),
                ));
            }
        }
    }
    let mut out = Vec::with_capacity(variables.len());
    for variable in variables.into_values() {
        if !names.insert(variable.name().to_owned()) {
            return Err(ReplayError::Symbolic(
                "symbolic parameter names must be unique".into(),
            ));
        }
        out.push(SymbolicParameter {
            variable,
            dtype: DType::I64,
        });
    }
    Ok(out)
}

fn derive_shape(
    graph: &Graph,
    node: NodeId,
    seeds: &BTreeMap<NodeId, SymbolicShape>,
    movement_candidates: &[SymbolicExpr],
    template_environment: &BTreeMap<SymbolicVar, i64>,
    memo: &mut BTreeMap<NodeId, SymbolicShape>,
    guards: &mut Vec<SymbolicGuard>,
) -> Result<SymbolicShape, ReplayError> {
    if let Some(shape) = memo.get(&node) {
        return Ok(shape.clone());
    }
    let concrete = || {
        graph
            .shape(node)
            .map(symbolic_from_concrete)
            .map_err(|error| ReplayError::Symbolic(error.to_string()))
    };
    let shape = match graph
        .op(node)
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
    {
        Op::Input { .. } | Op::Constant(_) => {
            seeds.get(&node).cloned().map_or_else(concrete, Ok)?
        }
        Op::Cast { input, .. }
        | Op::Contiguous { input }
        | Op::ContiguousBackward { input }
        | Op::Detach { input }
        | Op::Unary { input, .. } => derive_shape(
            graph,
            *input,
            seeds,
            movement_candidates,
            template_environment,
            memo,
            guards,
        )?,
        // Shape-changing bitcasts depend on the concrete final-axis byte
        // extent. Keep that static descriptor instead of inventing a symbolic
        // relation that the guarded replay ABI cannot encode yet.
        Op::Bitcast { .. } => concrete()?,
        Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } => broadcast_shapes(
            &derive_shape(
                graph,
                *lhs,
                seeds,
                movement_candidates,
                template_environment,
                memo,
                guards,
            )?,
            &derive_shape(
                graph,
                *rhs,
                seeds,
                movement_candidates,
                template_environment,
                memo,
                guards,
            )?,
            guards,
        )?,
        Op::Logical { lhs, rhs, .. } => match rhs {
            Some(rhs) => broadcast_shapes(
                &derive_shape(
                    graph,
                    *lhs,
                    seeds,
                    movement_candidates,
                    template_environment,
                    memo,
                    guards,
                )?,
                &derive_shape(
                    graph,
                    *rhs,
                    seeds,
                    movement_candidates,
                    template_environment,
                    memo,
                    guards,
                )?,
                guards,
            )?,
            None => derive_shape(
                graph,
                *lhs,
                seeds,
                movement_candidates,
                template_environment,
                memo,
                guards,
            )?,
        },
        Op::Select {
            condition,
            on_true,
            on_false,
        } => {
            let values = broadcast_shapes(
                &derive_shape(
                    graph,
                    *on_true,
                    seeds,
                    movement_candidates,
                    template_environment,
                    memo,
                    guards,
                )?,
                &derive_shape(
                    graph,
                    *on_false,
                    seeds,
                    movement_candidates,
                    template_environment,
                    memo,
                    guards,
                )?,
                guards,
            )?;
            broadcast_shapes(
                &values,
                &derive_shape(
                    graph,
                    *condition,
                    seeds,
                    movement_candidates,
                    template_environment,
                    memo,
                    guards,
                )?,
                guards,
            )?
        }
        Op::Reduce {
            input,
            axes,
            keepdim,
            ..
        } => {
            let input = derive_shape(
                graph,
                *input,
                seeds,
                movement_candidates,
                template_environment,
                memo,
                guards,
            )?;
            let mut dims = input.dims().to_vec();
            if *keepdim {
                for axis in axes {
                    dims[*axis] = 1usize.into();
                }
            } else {
                for axis in axes.iter().rev() {
                    dims.remove(*axis);
                }
            }
            SymbolicShape::new(dims)
        }
        Op::Matmul { lhs, rhs } => {
            let lhs = derive_shape(
                graph,
                *lhs,
                seeds,
                movement_candidates,
                template_environment,
                memo,
                guards,
            )?;
            let rhs = derive_shape(
                graph,
                *rhs,
                seeds,
                movement_candidates,
                template_environment,
                memo,
                guards,
            )?;
            SymbolicShape::new(matmul_geometry(&lhs, &rhs, guards)?.output)
        }
        Op::Shrink { input, .. }
        | Op::Reshape { input, .. }
        | Op::Permute { input, .. }
        | Op::Expand { input, .. }
        | Op::Stride { input, .. } => {
            let input = derive_shape(
                graph,
                *input,
                seeds,
                movement_candidates,
                template_environment,
                memo,
                guards,
            )?;
            super::symbolic_view::movement_shape(
                graph
                    .op(node)
                    .map_err(|error| ReplayError::Symbolic(error.to_string()))?,
                &input,
                graph
                    .shape(node)
                    .map_err(|error| ReplayError::Symbolic(error.to_string()))?,
                movement_candidates,
                template_environment,
                guards,
            )?
        }
        _ => {
            return Err(ReplayError::Unsupported(
                "operation is outside symbolic captured specialization".into(),
            ));
        }
    };
    validate_shape_bounds(&shape)?;
    memo.insert(node, shape.clone());
    Ok(shape)
}

fn symbolic_from_concrete(shape: &Shape) -> SymbolicShape {
    SymbolicShape::new(
        shape
            .dims()
            .iter()
            .copied()
            .map(SymbolicDim::from)
            .collect::<Vec<_>>(),
    )
}

fn broadcast_shapes(
    lhs: &SymbolicShape,
    rhs: &SymbolicShape,
    guards: &mut Vec<SymbolicGuard>,
) -> Result<SymbolicShape, ReplayError> {
    let rank = lhs.rank().max(rhs.rank());
    let mut reversed = Vec::with_capacity(rank);
    for offset in 0..rank {
        let left = lhs
            .dims()
            .get(lhs.rank().wrapping_sub(offset + 1))
            .cloned()
            .unwrap_or_else(|| 1usize.into());
        let right = rhs
            .dims()
            .get(rhs.rank().wrapping_sub(offset + 1))
            .cloned()
            .unwrap_or_else(|| 1usize.into());
        let left_one = left.expression().bounds().ok().and_then(|x| x.constant()) == Some(1);
        let right_one = right.expression().bounds().ok().and_then(|x| x.constant()) == Some(1);
        reversed.push(if left_one {
            right
        } else if right_one || left == right {
            left
        } else {
            guards.push(SymbolicGuard::equal(
                left.expression().clone(),
                right.expression().clone(),
            ));
            left
        });
    }
    reversed.reverse();
    Ok(SymbolicShape::new(reversed))
}

struct SymbolicMatmulGeometry {
    output: Vec<SymbolicDim>,
    batch: Vec<SymbolicDim>,
    m: SymbolicExpr,
    n: SymbolicExpr,
    k: SymbolicExpr,
}
fn matmul_geometry(
    lhs: &SymbolicShape,
    rhs: &SymbolicShape,
    guards: &mut Vec<SymbolicGuard>,
) -> Result<SymbolicMatmulGeometry, ReplayError> {
    if lhs.rank() == 0 || rhs.rank() == 0 {
        return Err(ReplayError::Symbolic(
            "symbolic matmul operands must have rank at least one".into(),
        ));
    }
    let lhs_vector = lhs.rank() == 1;
    let rhs_vector = rhs.rank() == 1;
    let lhs_k = lhs.dims()[lhs.rank() - 1].expression().clone();
    let rhs_k = rhs.dims()[rhs.rank() - usize::from(!rhs_vector) - 1]
        .expression()
        .clone();
    if lhs_k != rhs_k {
        guards.push(SymbolicGuard::equal(lhs_k.clone(), rhs_k));
    }
    let m = if lhs_vector {
        SymbolicExpr::constant(1)
    } else {
        lhs.dims()[lhs.rank() - 2].expression().clone()
    };
    let n = if rhs_vector {
        SymbolicExpr::constant(1)
    } else {
        rhs.dims()[rhs.rank() - 1].expression().clone()
    };
    let lhs_batch = if lhs_vector {
        SymbolicShape::new(Vec::new())
    } else {
        SymbolicShape::new(lhs.dims()[..lhs.rank() - 2].to_vec())
    };
    let rhs_batch = if rhs_vector {
        SymbolicShape::new(Vec::new())
    } else {
        SymbolicShape::new(rhs.dims()[..rhs.rank() - 2].to_vec())
    };
    let batch = broadcast_shapes(&lhs_batch, &rhs_batch, guards)?;
    let mut output = batch.dims().to_vec();
    if !lhs_vector {
        output.push(SymbolicDim::new(m.clone()));
    }
    if !rhs_vector {
        output.push(SymbolicDim::new(n.clone()));
    }
    Ok(SymbolicMatmulGeometry {
        output,
        batch: batch.dims().to_vec(),
        m,
        n,
        k: lhs_k,
    })
}

fn validate_shape_bounds(shape: &SymbolicShape) -> Result<(), ReplayError> {
    for dim in shape.dims() {
        let bounds = dim
            .expression()
            .bounds()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
        if bounds.min < 0 {
            return Err(ReplayError::Symbolic(
                "symbolic shape dimension may be negative".into(),
            ));
        }
        usize::try_from(bounds.max)
            .map_err(|_| ReplayError::Symbolic("symbolic dimension exceeds usize".into()))?;
    }
    let numel = shape
        .numel()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
    let bounds = numel
        .bounds()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
    usize::try_from(bounds.max)
        .map_err(|_| ReplayError::Symbolic("symbolic element count exceeds usize".into()))?;
    Ok(())
}

fn bind_shape(
    shape: &SymbolicShape,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<Shape, ReplayError> {
    shape
        .dims()
        .iter()
        .map(|dim| evaluate_usize(dim.expression(), environment))
        .collect::<Result<Vec<_>, _>>()
        .map(Shape::new)
}

fn evaluate(
    expression: &SymbolicExpr,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<i64, ReplayError> {
    let projected = expression
        .variables()
        .into_iter()
        .filter_map(|variable| environment.get(&variable).map(|value| (variable, *value)))
        .collect();
    expression
        .evaluate(&projected)
        .map_err(|error| ReplayError::Symbolic(error.to_string()))
}
fn evaluate_usize(
    expression: &SymbolicExpr,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<usize, ReplayError> {
    let value = evaluate(expression, environment)?;
    usize::try_from(value)
        .map_err(|_| ReplayError::Symbolic("symbolic value is not a usize".into()))
}

pub(crate) fn specialize_kernel(
    kernel: &UOp,
    schema: &SymbolicSchema,
    environment: &BTreeMap<SymbolicVar, i64>,
    domain: &BoundDomain,
    item_id: u64,
    output_buffer: u64,
) -> Result<UOp, ReplayError> {
    let output_extent = domain
        .output
        .numel()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
    let nodes = kernel
        .topological()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
    let mut range_extents = BTreeMap::new();
    for node in &nodes {
        let (buffer, range) = match (node.arg(), node.sources().get(1)) {
            (
                UArgRef::BufferIndex { buffer, .. } | UArgRef::ViewBufferIndex { buffer, .. },
                Some(range),
            ) => (*buffer, range),
            _ => continue,
        };
        let iteration_shape = if let Some((input, _, _)) = &domain.reduction {
            if buffer == output_buffer {
                &domain.output
            } else {
                input
            }
        } else {
            &domain.output
        };
        let extent = iteration_shape
            .numel()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
        if range_extents
            .insert(range.clone(), extent)
            .is_some_and(|old| old != extent)
        {
            return Err(ReplayError::Corrupt(
                "one symbolic range serves conflicting domains".into(),
            ));
        }
    }
    let mut rebuilt = BTreeMap::new();
    for node in nodes {
        let mut sources = node
            .sources()
            .iter()
            .map(|source| {
                rebuilt
                    .get(source)
                    .cloned()
                    .ok_or_else(|| ReplayError::Corrupt("symbolic UOp source order".into()))
            })
            .collect::<Result<Vec<UOp>, _>>()?;
        let operation = match node.operation() {
            Operation::Index(IndexValue::Buffer { buffer, .. }) => {
                let input_shape = schema.bind_shape(*buffer, environment)?;
                let output_shape = if let Some((reduction_input, _, _)) = &domain.reduction {
                    if *buffer == output_buffer {
                        domain.output.clone()
                    } else {
                        reduction_input.clone()
                    }
                } else {
                    domain.output.clone()
                };
                Operation::Index(IndexValue::Buffer {
                    buffer: *buffer,
                    elements: input_shape
                        .numel()
                        .map_err(|error| ReplayError::Symbolic(error.to_string()))?,
                    input_shape,
                    output_shape,
                })
            }
            Operation::Index(IndexValue::View { buffer, .. }) => {
                let source_shape = schema.bind_shape(*buffer, environment)?;
                let view = schema
                    .views
                    .get(&(item_id, *buffer))
                    .ok_or_else(|| ReplayError::Corrupt("symbolic view is absent".into()))?
                    .specialize(environment)?;
                if view.source_shape != source_shape {
                    return Err(ReplayError::Corrupt(
                        "symbolic view source shape disagrees with its buffer".into(),
                    ));
                }
                let output_shape = if let Some((reduction_input, _, _)) = &domain.reduction {
                    reduction_input.clone()
                } else {
                    domain.output.clone()
                };
                Operation::Index(IndexValue::View {
                    buffer: *buffer,
                    elements: view
                        .logical_shape
                        .numel()
                        .map_err(|error| ReplayError::Symbolic(error.to_string()))?,
                    input_shape: view.logical_shape.clone(),
                    output_shape,
                    view: view.into(),
                })
            }
            Operation::ReduceInit(ReductionValue {
                axes,
                keepdim,
                kind,
                mean,
                ..
            }) => {
                let (input_shape, _, _) = domain.reduction.as_ref().ok_or_else(|| {
                    ReplayError::Corrupt("reduction payload lacks symbolic domain".into())
                })?;
                Operation::ReduceInit(ReductionValue {
                    input_shape: input_shape.clone(),
                    output_shape: domain.output.clone(),
                    axes: axes.to_vec(),
                    keepdim: *keepdim,
                    kind: *kind,
                    mean: *mean,
                })
            }
            Operation::Matmul(value) => {
                let (batch, m, n, k, lhs, rhs) = domain.matmul.as_ref().ok_or_else(|| {
                    ReplayError::Corrupt("matmul payload lacks symbolic domain".into())
                })?;
                let lhs_shape = schema.bind_shape(*lhs, environment)?;
                let rhs_shape = schema.bind_shape(*rhs, environment)?;
                let specialized = node
                    .arg()
                    .matmul_plan()
                    .ok_or_else(|| ReplayError::Corrupt("matmul plan is absent".into()))?
                    .specialize_shapes(lhs_shape, rhs_shape)
                    .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
                if specialized.batch_shape.as_slice() != batch.dims()
                    || specialized.m != *m
                    || specialized.n != *n
                    || specialized.k != *k
                {
                    return Err(ReplayError::Corrupt(
                        "symbolic matmul domain disagrees with its payload".into(),
                    ));
                }
                match value {
                    MatmulValue::TensorCore(payload) => {
                        match crate::TensorCoreMatmulPayload::select(
                            specialized.clone(),
                            payload.tensor_core.target.clone(),
                        )
                        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
                        {
                            Some(payload) => {
                                Operation::Matmul(MatmulValue::TensorCore(Box::new(payload)))
                            }
                            None => Operation::Matmul(MatmulValue::Serial(Box::new(specialized))),
                        }
                    }
                    MatmulValue::Tiled(payload) => match crate::TiledMatmulPayload::select(
                        specialized.clone(),
                        payload.tile.target.clone(),
                    )
                    .map_err(|error| ReplayError::Symbolic(error.to_string()))?
                    {
                        Some(payload) => Operation::Matmul(MatmulValue::Tiled(Box::new(payload))),
                        None => Operation::Matmul(MatmulValue::Serial(Box::new(specialized))),
                    },
                    MatmulValue::Serial(_) => {
                        Operation::Matmul(MatmulValue::Serial(Box::new(specialized)))
                    }
                    MatmulValue::Quantized(_) => {
                        return Err(ReplayError::Unsupported(
                            "symbolic quantized matmul specialization".into(),
                        ));
                    }
                }
            }
            other => other.clone(),
        };
        if let Operation::Range(axis) = node.operation() {
            if *axis > 1 {
                return Err(ReplayError::Unsupported(
                    "symbolic range axis is outside captured specialization".into(),
                ));
            }
            let extent = range_extents.get(&node).copied().unwrap_or(output_extent);
            let extent = i64::try_from(extent)
                .map_err(|_| ReplayError::Symbolic("symbolic range exceeds i64".into()))?;
            let ty = node
                .sources()
                .first()
                .and_then(UOp::ty)
                .ok_or_else(|| ReplayError::Corrupt("range extent type is absent".into()))?;
            sources = vec![UOp::constant(extent, ty)];
        }
        let replacement = UOp::from_operation(operation, node.ty(), sources);
        rebuilt.insert(node, replacement);
    }
    let root = rebuilt
        .get(kernel)
        .cloned()
        .ok_or_else(|| ReplayError::Corrupt("symbolic UOp root is absent".into()))?;
    root.validate()
        .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
    if let Some((_, _, _, _, _, _)) = &domain.matmul
        && root.arg().matmul_plan().is_none()
    {
        return Err(ReplayError::Corrupt(
            "symbolic matmul item has the wrong UOp payload".into(),
        ));
    }
    let output = root
        .topological()
        .map_err(|error| ReplayError::Corrupt(error.to_string()))?
        .into_iter()
        .filter_map(|node| match node.arg() {
            UArgRef::BufferIndex { buffer, .. } if *buffer == output_buffer => Some(()),
            _ => None,
        })
        .count();
    if !matches!(root.kind(), UOpKind::Matmul) && output != 1 {
        return Err(ReplayError::Corrupt(
            "symbolic kernel output ABI is ambiguous".into(),
        ));
    }
    Ok(root)
}

pub(crate) fn specialize_capture(
    capture: &CapturedSchedule,
    canonical: &[(u64, i64)],
) -> Result<CapturedSchedule, ReplayError> {
    if capture
        .items
        .iter()
        .any(|item| matches!(item.kernel.kind(), crate::UOpKind::TensorGuard))
    {
        return Err(ReplayError::Unsupported(
            "tensor guard symbolic specialization is unsupported".into(),
        ));
    }
    if capture.items.iter().any(|item| !item.outputs.is_single()) {
        return Err(ReplayError::Unsupported(
            "multi-output symbolic specialization is unavailable".into(),
        ));
    }
    let schema = capture
        .symbolic
        .as_ref()
        .ok_or_else(|| ReplayError::Symbolic("artifact is already concrete".into()))?;
    schema.validate_against(capture)?;
    let environment = schema.environment(canonical)?;
    schema.validate_guards(&environment)?;
    let mut concrete = capture.clone();
    concrete.symbolic = None;
    concrete.specialized_from = Some(SpecializedFrom {
        source_identity: capture.identity,
        bindings: canonical.to_vec(),
    });
    for item in &mut concrete.items {
        let domain = schema.bind_domain(item.id, &environment)?;
        item.inputs = item
            .inputs
            .iter()
            .map(|desc| specialize_desc(schema, Some(item.id), desc, &environment))
            .collect::<Result<Vec<_>, _>>()?;
        item.input_bindings = item
            .input_bindings
            .iter()
            .map(|binding| {
                Ok(crate::ScheduleInputBinding {
                    input_node: binding.input_node,
                    desc: specialize_desc(schema, Some(item.id), &binding.desc, &environment)?,
                    abi_index: binding.abi_index,
                })
            })
            .collect::<Result<Vec<_>, ReplayError>>()?;
        let outputs = item
            .outputs
            .iter()
            .map(|output| specialize_desc(schema, Some(item.id), output, &environment))
            .collect::<Result<Vec<_>, _>>()?;
        item.outputs = crate::ScheduledOutputs::new(outputs)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        item.output = item.primary_output().clone();
        item.kernel = specialize_kernel(
            &item.kernel,
            schema,
            &environment,
            &domain,
            item.id,
            item.primary_output().id,
        )?;
        item.cache_key =
            crate::schedule::specialized_item_cache_key(item, capture.identity, canonical)
                .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        item.validate_input_bindings()
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
    }
    for input in &mut concrete.inputs {
        input.desc = specialize_desc(schema, None, &input.desc, &environment)?;
    }
    for (buffer, value) in &mut concrete.constants {
        if schema.splat_constants.contains(buffer) {
            let shape = schema.bind_shape(*buffer, &environment)?;
            *value = value
                .resize_exact_splat(shape)
                .map_err(|_| ReplayError::Symbolic("constant is not an exact splat".into()))?;
        }
    }
    concrete.identity = 0;
    concrete.identity = crate::schedule::artifact::identity(&concrete)
        .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
    crate::schedule::artifact::validate_capture(&concrete)
        .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
    Ok(concrete)
}

fn specialize_desc(
    schema: &SymbolicSchema,
    item_id: Option<u64>,
    desc: &BufferDesc,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<BufferDesc, ReplayError> {
    let shape = schema.bind_shape(desc.id, environment)?;
    let bytes = shape
        .numel()
        .ok()
        .and_then(|elements| elements.checked_mul(desc.dtype.itemsize()))
        .ok_or_else(|| ReplayError::Symbolic("specialized buffer size overflows".into()))?;
    let view = match (item_id, &desc.view) {
        (Some(item), Some(_)) => Some(
            schema
                .views
                .get(&(item, desc.id))
                .ok_or_else(|| ReplayError::Corrupt("symbolic descriptor view is absent".into()))?
                .specialize(environment)?,
        ),
        (None, Some(template)) => {
            let template_environment = schema.template_environment()?;
            let mut matches = schema
                .views
                .iter()
                .filter(|((_, buffer), _)| *buffer == desc.id)
                .filter_map(|(_, view)| {
                    (view
                        .specialize(&template_environment)
                        .ok()
                        .map(crate::AffineView::from)
                        .as_ref()
                        == Some(template))
                    .then_some(view)
                })
                .collect::<Vec<_>>();
            matches.dedup();
            let [symbolic] = matches.as_slice() else {
                if matches.is_empty() {
                    return Err(ReplayError::Corrupt(
                        "symbolic replay-input view is absent".into(),
                    ));
                }
                return Err(ReplayError::Unsupported(
                    "one replay input carries multiple symbolic views".into(),
                ));
            };
            Some(symbolic.specialize(environment)?)
        }
        (_, None) => None,
    };
    Ok(BufferDesc {
        id: desc.id,
        shape,
        dtype: desc.dtype,
        bytes,
        alignment: desc.alignment,
        read_only: desc.read_only,
        view: view.map(Into::into),
    })
}
