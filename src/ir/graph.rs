use super::creation::{LazyArangePlan, lazy_arange_default_int_plan};
use super::*;
use crate::nn::{ParameterId, ParameterSnapshot};
use crate::{
    CompileTrace, DType, EinsumPlan, Error, LiteralScalar, ReduceKind, ReductionDType, Result,
    Scalar, Shape, SplitSections, SymbolicShape, SymbolicVar, TensorData, TraceStep,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

fn nonzero_coordinate_dtype(shape: &Shape) -> Result<DType> {
    let mut dtype = DType::I32;
    for &extent in shape.dims() {
        let maximum_coordinate = extent.saturating_sub(1);
        i64::try_from(maximum_coordinate).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        if maximum_coordinate > i32::MAX as usize {
            dtype = DType::I64;
        }
    }
    Ok(dtype)
}

fn nonzero_coordinate_range_plan(extent: usize) -> Result<LazyArangePlan> {
    let shape = Shape::from([extent]);
    let dtype = nonzero_coordinate_dtype(&shape)?;
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    Ok(LazyArangePlan {
        shape,
        dtype,
        step: TensorData::scalar_with_dtype(Scalar::I(1), dtype),
        offset: TensorData::scalar_with_dtype(Scalar::I(-1), dtype),
    })
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub op: Op,
    pub shape: Shape,
    pub dtype: DType,
    pub requires_grad: bool,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) dynamic_nodes: Vec<DynamicNode>,
    id: u64,
    pub(crate) grad_enabled: bool,
    parameter_bindings: BTreeMap<(ParameterId, u64), ParameterBinding>,
}

/// One heterogeneous source `Tensor.sequential` transform.
///
/// Box this alias to compose closures and function items with distinct concrete
/// types in one ordered list.
pub type GraphSequentialTransform = Box<dyn Fn(&mut Graph, NodeId) -> Result<NodeId>>;

/// Exact closed mode set accepted by tinygrad's concrete public `Tensor.pad`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PadMode {
    Constant,
    Circular,
    Reflect,
    Replicate,
}

/// Exact closed reduction set accepted by tinygrad's public
/// `Tensor.scatter_reduce`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScatterReduceKind {
    Sum,
    Prod,
    Mean,
    Amax,
    Amin,
}

/// Source argument accepted by tinygrad's public `Tensor.scatter`: either a
/// live tensor or one concrete Python-style scalar expanded lazily to the
/// index shape at the base tensor's storage dtype.
#[derive(Clone, Copy, Debug)]
pub enum ScatterSource {
    Tensor(NodeId),
    Scalar(Scalar),
}

/// Exact closed `reduce` set for public tinygrad `Tensor.scatter`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScatterMode {
    Replace,
    Add,
    Multiply,
}

/// Descriptor-first source plan for public `Tensor.scatter_reduce`.
///
/// The source deliberately does not use a scatter primitive: it crops the
/// update payload, creates a live one-hot predicate, pads its update lanes to
/// the base geometry, and reduces the final synthetic axis.  Rehearsing that
/// literal graph on a clone validates every view, lazy range, Select and
/// reduction boundary before the first live node is published.
#[derive(Clone, Debug)]
struct ScatterReducePlan {
    dim: usize,
    index_shape: Shape,
    base_shape: Shape,
    base_dtype: DType,
    kind: ScatterReduceKind,
    include_self: bool,
    output_shape: Shape,
    output_dtype: DType,
}

#[derive(Clone, Debug)]
struct ScatterPlan {
    dim: usize,
    index_shape: Shape,
    base_shape: Shape,
    base_dtype: DType,
    source: ScatterSource,
    mode: ScatterMode,
    output_shape: Shape,
    output_dtype: DType,
}

#[derive(Clone, Debug)]
struct PadModePlan {
    padding: Vec<(i64, i64)>,
    mode: PadMode,
    fill: Scalar,
    output_shape: Shape,
    output_dtype: DType,
}

/// Concrete target-shape contract for tinygrad's public `Tensor.pad_to`.
/// `None` retains that source extent; concrete targets crop trailing elements
/// when smaller and append only trailing zero/fill lanes when larger.
#[derive(Clone, Debug)]
struct PadToPlan {
    source_shape: Shape,
    bounds: Vec<(usize, usize)>,
    positive: Vec<(usize, usize)>,
    changed: bool,
    fill: Scalar,
    output_shape: Shape,
}

/// Concrete checked bounds for tinygrad's public `Tensor.shrink_to`.
#[derive(Clone, Debug)]
struct ShrinkToPlan {
    bounds: Vec<(usize, usize)>,
    output_shape: Shape,
}

/// Fully resolved concrete `Tensor.chunk` views before any Shrink node is
/// published. Tinygrad computes its chunk widths through `split`, so an
/// over-large nonempty chunk count intentionally yields fewer views while a
/// zero extent yields exactly `chunks` empty views.
#[derive(Clone, Debug)]
struct ChunkPlan {
    bounds: Vec<Vec<(usize, usize)>>,
}

/// Fully resolved concrete `Tensor.split` views before any Shrink node is
/// published. This retains source's two section forms: a uniform maximum
/// width (including its one-empty-view zero-axis case), or every explicit
/// section including zero-width sections.
#[derive(Clone, Debug)]
struct SplitPlan {
    bounds: Vec<Vec<(usize, usize)>>,
}

#[derive(Clone, Debug)]
struct ParameterBinding {
    node: NodeId,
    input_name: String,
    data: TensorData,
}

/// Fully resolved public tinygrad `Tensor.cat` operation before any stack,
/// pad, cast, or ADD node has been published.
#[derive(Clone, Debug)]
struct CatPlan {
    inputs: Vec<NodeId>,
    axis: usize,
    output_shape: Shape,
    output_dtype: DType,
    identity: bool,
    lowering: CatLowering,
}

#[derive(Clone, Debug)]
enum CatLowering {
    Stack,
    PadSum { paddings: Vec<Vec<(usize, usize)>> },
}

/// Descriptor-first lowering plan for tinygrad's public `Tensor.dot`.
///
/// The public source is deliberately not RustGrad's raw Matmul op: it reshapes
/// both operands into a broadcastable contraction, multiplies in the source
/// LUB, runs a typed Sum on the final axis, then casts the result back to the
/// requested/default output storage.  QR and other composite linalg helpers
/// rely on these accumulator boundaries being observable.
#[derive(Clone, Debug)]
struct SourceDotPlan {
    lhs_shape: Shape,
    rhs_reshape: Shape,
    rhs_shape: Shape,
    rhs_axis: isize,
    operand_dtype: DType,
    product_shape: Shape,
    sum_dtypes: ReductionDType,
    output_shape: Shape,
    output_dtype: DType,
}

/// Whole-operation contract for tinygrad's public `Tensor.linear` helper.
///
/// Unlike a conventional framework linear layer, the source receives a
/// weight already in `Tensor.dot` geometry. A requested dtype is an
/// operand-storage cast followed by the ordinary default-Dot path, never an
/// instruction to raw Matmul or to Dot's explicit accumulator override.
#[derive(Clone, Debug)]
struct LinearPlan {
    dtype: Option<DType>,
    rank_one_weight: bool,
    output_shape: Shape,
    output_dtype: DType,
}

/// Whole-operation descriptor contract for tinygrad's static Householder QR.
///
/// The source unrolls exactly `min(m, n)` reflector updates over concrete
/// trailing matrix dimensions, retains a full `[..., m, m]` Q, and leaves R
/// at the source descriptor. The stage rehearsal in [`Graph::qr`] validates
/// every concrete view, scalar, typed Dot, and update before the live graph
/// receives its first node.
#[derive(Clone, Debug)]
struct QrPlan {
    q_shape: Shape,
    r_shape: Shape,
    dtype: DType,
    m: usize,
    stages: usize,
}

/// Whole-operation descriptor contract for checked-in tinygrad's static
/// Jacobi SVD composition.
///
/// SVD remains a source graph, not a new operation family: QR, two explicit
/// Contiguous identities, the fixed round-robin Jacobi network, coupled Sort,
/// source Gather, and the final padding/matmul composition retain their own
/// typed contracts. The private rehearsal in [`Graph::svd`] validates that
/// complete graph before the live graph receives its first node.
#[derive(Clone, Debug)]
struct SvdPlan {
    input_shape: Shape,
    batch: Vec<usize>,
    dtype: DType,
    m: usize,
    n: usize,
    num: usize,
    q_num: usize,
    h: usize,
    rounds: usize,
    transpose_input: bool,
    full_matrices: bool,
    singular_shape: Shape,
    left_shape: Shape,
    right_shape: Shape,
}

#[derive(Clone, Copy, Debug)]
struct SvdJacobiState {
    u: NodeId,
    v: NodeId,
    permutation: NodeId,
}

/// Concrete whole-operation contract for tinygrad's public
/// `Tensor.newton_schulz(steps, params, eps)`.
#[derive(Clone, Debug)]
struct NewtonSchulzPlan {
    input_shape: Shape,
    dtype: DType,
    transpose_input: bool,
    iterations: usize,
}

fn newton_schulz_plan(
    graph: &Graph,
    input: NodeId,
    steps: isize,
    params: &[i64],
) -> Result<NewtonSchulzPlan> {
    let source = graph.node(input)?;
    let input_shape = source.shape.clone();
    let dtype = source.dtype;
    input_shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
    if input_shape.rank() < 2 {
        return Err(Error::InvalidMatmul {
            lhs: input_shape.clone(),
            rhs: input_shape,
        });
    }
    // Python's `range(negative)` simply skips the polynomial loop. Its
    // `functools.reduce` is reached only for a positive iteration count, at
    // which point an empty generator raises rather than supplying an identity.
    let iterations = usize::try_from(steps).unwrap_or(0);
    if iterations > 0 && params.is_empty() {
        return Err(Error::InvalidRandom {
            reason: "newton_schulz requires nonempty params for positive steps",
        });
    }
    let rows = input_shape.dims()[input_shape.rank() - 2];
    let columns = input_shape.dims()[input_shape.rank() - 1];
    let transpose_input = rows > columns;
    if transpose_input {
        let mut transposed = input_shape.dims().to_vec();
        let last = transposed.len() - 1;
        transposed.swap(last, last - 1);
        let transposed = Shape::new(transposed);
        transposed
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or(Error::ShapeOverflow(transposed))?;
    }
    Ok(NewtonSchulzPlan {
        input_shape,
        dtype,
        transpose_input,
        iterations,
    })
}

fn qr_plan(graph: &Graph, input: NodeId) -> Result<QrPlan> {
    let source = graph.node(input)?;
    let r_shape = source.shape.clone();
    let dtype = source.dtype;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&r_shape, dtype)?;
    if r_shape.rank() < 2 {
        return Err(Error::InvalidMatmul {
            lhs: r_shape.clone(),
            rhs: r_shape,
        });
    }
    let m = r_shape.dims()[r_shape.rank() - 2];
    let n = r_shape.dims()[r_shape.rank() - 1];
    let mut q_dims = r_shape.dims()[..r_shape.rank() - 2].to_vec();
    q_dims.extend([m, m]);
    let q_shape = Shape::new(q_dims);
    extent(&q_shape, dtype)?;
    let m_i64 = i64::try_from(m).map_err(|_| Error::ShapeOverflow(q_shape.clone()))?;
    let index = lazy_arange_default_int_plan(0, m_i64, 1)?;
    extent(&index.shape, index.dtype)?;
    Ok(QrPlan {
        q_shape,
        r_shape,
        dtype,
        m,
        stages: m.min(n),
    })
}

fn svd_plan(graph: &Graph, input: NodeId, full_matrices: bool) -> Result<SvdPlan> {
    let source = graph.node(input)?;
    let input_shape = source.shape.clone();
    let dtype = source.dtype;
    let checked_extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    checked_extent(&input_shape, dtype)?;
    if input_shape.rank() < 2 {
        return Err(Error::InvalidMatmul {
            lhs: input_shape.clone(),
            rhs: input_shape,
        });
    }
    let rank = input_shape.rank();
    let batch = input_shape.dims()[..rank - 2].to_vec();
    let m = input_shape.dims()[rank - 2];
    let n = input_shape.dims()[rank - 1];
    let num = m.min(n);
    let q_num = m.max(n);
    let h = num / 2;
    // Checked-in tinygrad reaches `split(0)` during each of the four Jacobi
    // rounds for a one-column core. That split returns one empty section, so
    // the source tuple destructure fails. Preserve the observable rejection,
    // but report it before RustGrad publishes any of the preceding QR graph.
    if num == 1 {
        return Err(Error::InvalidSplit {
            reason: "source svd split(0) produces one Jacobi section",
        });
    }
    let rounds = num
        .checked_mul(4)
        .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
    let _ = h
        .checked_mul(2)
        .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
    i64::try_from(num).map_err(|_| Error::ShapeOverflow(input_shape.clone()))?;
    i64::try_from(h).map_err(|_| Error::ShapeOverflow(input_shape.clone()))?;

    let mut singular_dims = batch.clone();
    singular_dims.push(num);
    let singular_shape = Shape::new(singular_dims);
    checked_extent(&singular_shape, dtype)?;

    let mut left_dims = batch.clone();
    left_dims.extend(if full_matrices { [m, m] } else { [m, num] });
    let left_shape = Shape::new(left_dims);
    checked_extent(&left_shape, dtype)?;

    let mut right_dims = batch.clone();
    right_dims.extend(if full_matrices { [n, n] } else { [num, n] });
    let right_shape = Shape::new(right_dims);
    checked_extent(&right_shape, dtype)?;

    for extent in [num, q_num] {
        checked_extent(&Shape::new([extent, extent]), dtype)?;
        let mut square = batch.clone();
        square.extend([extent, extent]);
        checked_extent(&Shape::new(square), dtype)?;
    }

    Ok(SvdPlan {
        input_shape,
        batch,
        dtype,
        m,
        n,
        num,
        q_num,
        h,
        rounds,
        transpose_input: m < n,
        full_matrices,
        singular_shape,
        left_shape,
        right_shape,
    })
}

fn svd_jacobi_round(
    graph: &mut Graph,
    plan: &SvdPlan,
    state: SvdJacobiState,
    columns: NodeId,
    eye: NodeId,
) -> Result<SvdJacobiState> {
    let columns = graph.unsqueeze(columns, 1)?;
    let round_permutation = graph.unsqueeze(state.permutation, 0)?;
    let selectors = graph.eq(columns, round_permutation)?;
    let u_dtype = graph.dtype(state.u)?;
    let selectors = graph.cast(selectors, u_dtype)?;
    let selector_shape = graph.shape(selectors)?.clone();
    let last = selector_shape.rank() - 1;
    let two_h = plan
        .h
        .checked_mul(2)
        .ok_or_else(|| Error::ShapeOverflow(plan.input_shape.clone()))?;
    let pair_bounds = selector_shape
        .dims()
        .iter()
        .enumerate()
        .map(|(axis, &extent)| {
            if axis == last {
                (0, two_h)
            } else {
                (0, extent)
            }
        })
        .collect::<Vec<_>>();
    let pair_selectors = graph.shrink(selectors, pair_bounds)?;

    let u_pair = graph.dot_default(state.u, pair_selectors)?;
    let mut u_halves = graph.split(u_pair, plan.h, -1)?;
    if u_halves.len() != 2 {
        return Err(Error::InvalidSplit {
            reason: "svd Jacobi column split must produce two sections",
        });
    }
    let u_right = u_halves.pop().expect("two SVD halves preflighted");
    let u_left = u_halves.pop().expect("two SVD halves preflighted");
    let gamma_product = graph.mul(u_left, u_right)?;
    let gamma = graph.sum_with_options(gamma_product, Some(vec![-2]), false, None)?;
    let mut gamma_shape = plan.batch.clone();
    gamma_shape.extend([1, plan.h]);
    let gamma = graph.reshape(gamma, Shape::new(gamma_shape))?;

    let u_pair_squared = graph.square(u_pair)?;
    let norms = graph.sum_with_options(u_pair_squared, Some(vec![-2]), false, None)?;
    let norms = graph.unsqueeze(norms, -2)?;
    let mut norm_halves = graph.split(norms, plan.h, -1)?;
    if norm_halves.len() != 2 {
        return Err(Error::InvalidSplit {
            reason: "svd Jacobi norm split must produce two sections",
        });
    }
    let beta = norm_halves.pop().expect("two SVD norm halves preflighted");
    let alpha = norm_halves.pop().expect("two SVD norm halves preflighted");
    let rotate = graph.ne_scalar(gamma, Scalar::I(0))?;
    let safe_gamma = graph.where_false_scalar(rotate, gamma, Scalar::I(1))?;
    let twice_gamma = graph.mul_scalar(safe_gamma, Scalar::I(2))?;
    let beta_minus_alpha = graph.sub(beta, alpha)?;
    let tau = graph.div(beta_minus_alpha, twice_gamma)?;
    let tau_nonzero = graph.ne_scalar(tau, Scalar::I(0))?;
    let tau_sign = graph.sign(tau)?;
    let numerator = graph.where_false_scalar(tau_nonzero, tau_sign, Scalar::I(1))?;
    let tau_squared = graph.square(tau)?;
    let root = graph.add_scalar(tau_squared, Scalar::I(1))?;
    let root = graph.sqrt(root)?;
    let tau_abs = graph.abs(tau)?;
    let denominator = graph.add(tau_abs, root)?;
    let tangent = graph.div(numerator, denominator)?;
    let tangent = graph.where_false_scalar(rotate, tangent, Scalar::I(0))?;
    let tangent_squared = graph.square(tangent)?;
    let cosine = graph.add_scalar(tangent_squared, Scalar::I(1))?;
    let cosine = graph.sqrt(cosine)?;
    let cosine = graph.reciprocal(cosine)?;
    let sine = graph.mul(cosine, tangent)?;

    let pair_selectors_t = graph.transpose(pair_selectors, -2, -1)?;
    let mut selector_halves = graph.split(pair_selectors_t, plan.h, -2)?;
    if selector_halves.len() != 2 {
        return Err(Error::InvalidSplit {
            reason: "svd Jacobi selector split must produce two sections",
        });
    }
    let right_selector = selector_halves
        .pop()
        .expect("two SVD selector halves preflighted");
    let left_selector = selector_halves
        .pop()
        .expect("two SVD selector halves preflighted");
    let left_column = graph.unsqueeze(left_selector, -1)?;
    let left_row = graph.unsqueeze(left_selector, -2)?;
    let right_column = graph.unsqueeze(right_selector, -1)?;
    let right_row = graph.unsqueeze(right_selector, -2)?;
    let left_diagonal = graph.mul(left_column, left_row)?;
    let right_diagonal = graph.mul(right_column, right_row)?;
    let diagonal = graph.add(left_diagonal, right_diagonal)?;
    let left_right = graph.mul(left_column, right_row)?;
    let right_left = graph.mul(right_column, left_row)?;
    let cross = graph.sub(left_right, right_left)?;

    let cosine_delta = graph.sub_scalar(cosine, Scalar::I(1))?;
    let mut coefficient_shape = plan.batch.clone();
    coefficient_shape.extend([plan.h, 1, 1]);
    let coefficient_shape = Shape::new(coefficient_shape);
    let cosine_delta = graph.reshape(cosine_delta, coefficient_shape.clone())?;
    let sine = graph.reshape(sine, coefficient_shape)?;
    let diagonal = graph.mul(cosine_delta, diagonal)?;
    let cross = graph.mul(sine, cross)?;
    let delta = graph.add(diagonal, cross)?;
    let delta = graph.sum_with_options(delta, Some(vec![-3]), false, None)?;
    let rotation = graph.add(eye, delta)?;
    let u = graph.dot_default(state.u, rotation)?;
    let v = graph.dot_default(state.v, rotation)?;

    let num_i64 =
        i64::try_from(plan.num).map_err(|_| Error::ShapeOverflow(plan.input_shape.clone()))?;
    let permutation = if plan.num % 2 == 1 {
        let shifted = graph.sub_scalar(state.permutation, Scalar::I(1))?;
        graph.modulo_scalar(shifted, Scalar::I(num_i64))?
    } else {
        let permutation_shape = graph.shape(state.permutation)?.clone();
        let first = graph.shrink(state.permutation, vec![(0, 1)])?;
        let rest = graph.shrink(state.permutation, vec![(1, permutation_shape.dims()[0])])?;
        let rest = graph.sub_scalar(rest, Scalar::I(2))?;
        let rest = graph.modulo_scalar(rest, Scalar::I(num_i64 - 1))?;
        let rest = graph.add_scalar(rest, Scalar::I(1))?;
        graph.cat(first, vec![rest], 0)?
    };
    Ok(SvdJacobiState { u, v, permutation })
}

fn source_dot_plan(
    graph: &Graph,
    lhs: NodeId,
    rhs: NodeId,
    dtype: Option<DType>,
) -> Result<SourceDotPlan> {
    let lhs_node = graph.node(lhs)?;
    let lhs_source_shape = lhs_node.shape.clone();
    let lhs_dtype = lhs_node.dtype;
    let rhs_node = graph.node(rhs)?;
    let rhs_source_shape = rhs_node.shape.clone();
    let rhs_dtype = rhs_node.dtype;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&lhs_source_shape, lhs_dtype)?;
    extent(&rhs_source_shape, rhs_dtype)?;
    if lhs_source_shape.rank() == 0 || rhs_source_shape.rank() == 0 {
        return Err(Error::InvalidMatmul {
            lhs: lhs_source_shape,
            rhs: rhs_source_shape,
        });
    }
    let rhs_axis = -(rhs_source_shape.rank().min(2) as isize);
    let rhs_contract_axis = if rhs_source_shape.rank() == 1 {
        0
    } else {
        rhs_source_shape.rank() - 2
    };
    if lhs_source_shape.dims()[lhs_source_shape.rank() - 1]
        != rhs_source_shape.dims()[rhs_contract_axis]
    {
        return Err(Error::InvalidMatmul {
            lhs: lhs_source_shape,
            rhs: rhs_source_shape,
        });
    }

    // Exact source reshape rule: each operand receives one singleton only
    // when both original operands have a non-batch matrix axis.
    let insert_singleton = lhs_source_shape.rank() > 1 && rhs_source_shape.rank() > 1;
    let mut lhs_dims = lhs_source_shape.dims()[..lhs_source_shape.rank() - 1].to_vec();
    if insert_singleton {
        lhs_dims.push(1);
    }
    lhs_dims.push(lhs_source_shape.dims()[lhs_source_shape.rank() - 1]);
    let lhs_shape = Shape::new(lhs_dims);

    let mut rhs_dims =
        rhs_source_shape.dims()[..rhs_source_shape.rank().saturating_sub(2)].to_vec();
    if insert_singleton {
        rhs_dims.push(1);
    }
    if rhs_source_shape.rank() == 1 {
        rhs_dims.push(rhs_source_shape.dims()[0]);
    } else {
        rhs_dims.extend_from_slice(&rhs_source_shape.dims()[rhs_source_shape.rank() - 2..]);
    }
    let rhs_reshaped = Shape::new(rhs_dims);
    let mut rhs_transposed_dims = rhs_reshaped.dims().to_vec();
    let rhs_last = rhs_transposed_dims.len() - 1;
    let rhs_transpose_axis = if rhs_reshaped.rank() == 1 {
        0
    } else {
        rhs_reshaped.rank() - 2
    };
    rhs_transposed_dims.swap(rhs_last, rhs_transpose_axis);
    let rhs_shape = Shape::new(rhs_transposed_dims);
    let product_shape = lhs_shape.broadcast_with(&rhs_shape)?;
    let mut output_dims = product_shape.dims().to_vec();
    output_dims
        .pop()
        .expect("source dot inputs have rank at least one");
    let output_shape = Shape::new(output_dims);
    let operand_dtype = super::elementwise::source_lub(lhs_dtype, rhs_dtype);
    let sum_dtypes = dtype
        .map(|dtype| ReductionDType::new(dtype, dtype))
        .unwrap_or_else(|| ReductionDType::sum_default(operand_dtype));
    let output_dtype = dtype.unwrap_or(operand_dtype);

    // Every source descriptor, source-LUB cast, elementwise product, typed
    // reduction boundary, and final storage cast is checked before lowerer
    // publication. The source Sum takes the product descriptor as input.
    for (shape, storage) in [
        (&lhs_source_shape, lhs_dtype),
        (&rhs_source_shape, rhs_dtype),
        (&lhs_shape, lhs_dtype),
        (&rhs_reshaped, rhs_dtype),
        (&rhs_shape, rhs_dtype),
        (&lhs_shape, operand_dtype),
        (&rhs_shape, operand_dtype),
        (&product_shape, operand_dtype),
        (&product_shape, sum_dtypes.accumulator),
        (&output_shape, sum_dtypes.accumulator),
        (&output_shape, sum_dtypes.output),
        (&output_shape, output_dtype),
    ] {
        extent(shape, storage)?;
    }
    if product_shape.rank() == 0
        || product_shape.dims().last()
            != Some(&lhs_source_shape.dims()[lhs_source_shape.rank() - 1])
        || super::elementwise::source_lub(lhs_dtype, rhs_dtype) != operand_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "source dot promotion",
            actual: operand_dtype,
        });
    }
    Ok(SourceDotPlan {
        lhs_shape,
        rhs_reshape: rhs_reshaped,
        rhs_shape,
        rhs_axis,
        operand_dtype,
        product_shape,
        sum_dtypes,
        output_shape,
        output_dtype,
    })
}

fn linear_plan(
    graph: &Graph,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    dtype: Option<DType>,
) -> Result<LinearPlan> {
    let input_node = graph.node(input)?;
    let weight_node = graph.node(weight)?;
    let bias_node = bias.map(|node| graph.node(node)).transpose()?;
    if weight_node.shape.rank() == 0 {
        return Err(Error::InvalidMatmul {
            lhs: input_node.shape.clone(),
            rhs: weight_node.shape.clone(),
        });
    }
    let extent = |shape: &Shape, storage: DType| {
        shape
            .numel()?
            .checked_mul(storage.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&input_node.shape, input_node.dtype)?;
    extent(&weight_node.shape, weight_node.dtype)?;
    if let Some(bias) = bias_node {
        extent(&bias.shape, bias.dtype)?;
    }
    if let Some(dtype) = dtype {
        // `self.cast(dt).linear(weight.cast(dt), bias.cast(dt))`: all three
        // cast descriptors must be valid before a source-Dot reshape or a
        // scalar can appear in the caller graph.
        extent(&input_node.shape, dtype)?;
        extent(&weight_node.shape, dtype)?;
        if let Some(bias) = bias_node {
            extent(&bias.shape, dtype)?;
        }
    }

    // The private clone is the complete remaining descriptor pass: Dot owns
    // its source-LUB reshape/transpose/Mul/typed-Sum contract, and Add owns
    // the final live bias broadcast. It leaves the caller graph unchanged on
    // every late shape, dtype, or byte failure.
    let mut rehearsal = graph.clone();
    let output = lower_linear(&mut rehearsal, input, weight, bias, dtype)?;
    let output_shape = rehearsal.shape(output)?.clone();
    let output_dtype = rehearsal.dtype(output)?;
    extent(&output_shape, output_dtype)?;
    Ok(LinearPlan {
        dtype,
        rank_one_weight: weight_node.shape.rank() == 1,
        output_shape,
        output_dtype,
    })
}

fn lower_linear(
    graph: &mut Graph,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    dtype: Option<DType>,
) -> Result<NodeId> {
    let input = if let Some(dtype) = dtype {
        graph.cast(input, dtype)?
    } else {
        input
    };
    let weight = if let Some(dtype) = dtype {
        graph.cast(weight, dtype)?
    } else {
        weight
    };
    let bias = if let (Some(bias), Some(dtype)) = (bias, dtype) {
        Some(graph.cast(bias, dtype)?)
    } else {
        bias
    };
    let output = if graph.shape(weight)?.rank() == 1 {
        graph.mul(input, weight)?
    } else {
        graph.dot_default(input, weight)?
    };
    if let Some(bias) = bias {
        graph.add(output, bias)
    } else {
        Ok(output)
    }
}

fn pad_zero(value: Scalar) -> bool {
    match value {
        Scalar::Bool(value) => !value,
        Scalar::I(value) => value == 0,
        Scalar::U(value) => value == 0,
        Scalar::F(value) => value == 0.0,
    }
}

fn signed_pad_bounds(shape: &Shape, padding: &[(i64, i64)]) -> Result<Vec<(usize, usize)>> {
    shape
        .dims()
        .iter()
        .zip(padding)
        .enumerate()
        .map(|(axis, (&dimension, &(before, after)))| {
            let dimension =
                i128::try_from(dimension).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            let start = (-i128::from(before)).max(0);
            let end = (dimension + i128::from(after)).min(dimension);
            if end < 0 || start > end {
                return Err(Error::InvalidBounds {
                    axis,
                    start: usize::try_from(start).unwrap_or(usize::MAX),
                    end: usize::try_from(end.max(0)).unwrap_or(usize::MAX),
                    dim: usize::try_from(dimension).unwrap_or(usize::MAX),
                });
            }
            Ok((
                usize::try_from(start).map_err(|_| Error::ShapeOverflow(shape.clone()))?,
                usize::try_from(end).map_err(|_| Error::ShapeOverflow(shape.clone()))?,
            ))
        })
        .collect()
}

fn positive_pads(shape: &Shape, padding: &[(i64, i64)]) -> Result<Vec<(usize, usize)>> {
    padding
        .iter()
        .map(|&(before, after)| {
            Ok((
                usize::try_from(before.max(0)).map_err(|_| Error::ShapeOverflow(shape.clone()))?,
                usize::try_from(after.max(0)).map_err(|_| Error::ShapeOverflow(shape.clone()))?,
            ))
        })
        .collect()
}

fn lower_pad_mode(
    graph: &mut Graph,
    input: NodeId,
    padding: &[(i64, i64)],
    mode: PadMode,
    fill: Scalar,
) -> Result<NodeId> {
    let shape = graph.shape(input)?.clone();
    if padding.len() != shape.rank() {
        return Err(Error::InvalidMovementRank {
            op: "pad_with_mode",
            expected: shape.rank(),
            actual: padding.len(),
        });
    }
    let shrink_first = matches!(mode, PadMode::Constant | PadMode::Circular);
    let mut value = input;
    if shrink_first {
        let bounds = signed_pad_bounds(&shape, padding)?;
        value = graph.shrink(value, bounds)?;
    }
    let positive = positive_pads(&shape, padding)?;
    match mode {
        PadMode::Constant => {
            let base = graph.pad(value, positive.clone(), Scalar::I(0))?;
            if pad_zero(fill) {
                Ok(base)
            } else {
                // `_pad_constant`: Bool const_like -> zero Pad ->
                // `mask.where(base, Python_fill)`. The scalar false branch
                // owns tinygrad's weak commitment and possible output lift.
                let mask = graph.lazy_full_with_dtype(
                    graph.shape(value)?.clone(),
                    Scalar::Bool(true),
                    DType::Bool,
                )?;
                let mask = graph.pad(mask, positive, Scalar::Bool(false))?;
                graph.where_false_scalar(mask, base, fill)
            }
        }
        PadMode::Circular => {
            let cropped_shape = graph.shape(value)?.clone();
            for (axis, (&dimension, &(before, after))) in
                cropped_shape.dims().iter().zip(&positive).enumerate()
            {
                if before > dimension || after > dimension {
                    return Err(Error::InvalidBounds {
                        axis,
                        start: before,
                        end: after,
                        dim: dimension,
                    });
                }
            }
            let repeats = positive
                .iter()
                .map(|&(before, after)| 1isize + (before != 0) as isize + (after != 0) as isize)
                .collect::<Vec<_>>();
            // tinygrad permits `repeat(())` for the rank-zero circular
            // no-op, while RustGrad's public repeat intentionally rejects an
            // empty repeat list. Keep that source-local scalar identity here.
            let repeated = if repeats.is_empty() {
                value
            } else {
                graph.repeat(value, &repeats)?
            };
            let repeated_shape = graph.shape(repeated)?.clone();
            let bounds = positive
                .iter()
                .zip(cropped_shape.dims())
                .zip(repeated_shape.dims())
                .map(|((&(before, after), &original), &expanded)| {
                    let start = if before == 0 {
                        0
                    } else {
                        original
                            .checked_sub(before)
                            .ok_or_else(|| Error::ShapeOverflow(cropped_shape.clone()))?
                    };
                    let end = if after == 0 {
                        expanded
                    } else {
                        expanded
                            .checked_sub(original)
                            .and_then(|value| value.checked_add(after))
                            .ok_or_else(|| Error::ShapeOverflow(cropped_shape.clone()))?
                    };
                    Ok((start, end))
                })
                .collect::<Result<Vec<_>>>()?;
            graph.shrink(repeated, bounds)
        }
        PadMode::Reflect | PadMode::Replicate => {
            for axis in 0..shape.rank() {
                let (before, after) = positive[axis];
                let current_shape = graph.shape(value)?.clone();
                let dimension = current_shape.dims()[axis];
                if mode == PadMode::Reflect && (before >= dimension || after >= dimension) {
                    return Err(Error::InvalidBounds {
                        axis,
                        start: before,
                        end: after,
                        dim: dimension,
                    });
                }
                if mode == PadMode::Replicate && (before != 0 || after != 0) && dimension == 0 {
                    return Err(Error::InvalidBounds {
                        axis,
                        start: before,
                        end: after,
                        dim: dimension,
                    });
                }
                let mut pieces = Vec::with_capacity(3);
                if before != 0 {
                    let part = if mode == PadMode::Reflect {
                        let mut slices = vec![
                            Slice {
                                start: None,
                                stop: None,
                                step: 1
                            };
                            shape.rank()
                        ];
                        slices[axis] = Slice {
                            start: Some(
                                isize::try_from(before)
                                    .map_err(|_| Error::ShapeOverflow(current_shape.clone()))?,
                            ),
                            stop: Some(0),
                            step: -1,
                        };
                        graph.stride(value, slices)?
                    } else {
                        let mut bounds = current_shape
                            .dims()
                            .iter()
                            .map(|&end| (0, end))
                            .collect::<Vec<_>>();
                        bounds[axis] = (0, 1);
                        let part = graph.shrink(value, bounds)?;
                        let mut expanded = current_shape.dims().to_vec();
                        expanded[axis] = before;
                        graph.expand(part, Shape::new(expanded))?
                    };
                    pieces.push(part);
                }
                pieces.push(value);
                if after != 0 {
                    let part = if mode == PadMode::Reflect {
                        let start = dimension
                            .checked_sub(2)
                            .ok_or_else(|| Error::ShapeOverflow(current_shape.clone()))?;
                        let stop = dimension
                            .checked_sub(2)
                            .and_then(|value| value.checked_sub(after));
                        let mut slices = vec![
                            Slice {
                                start: None,
                                stop: None,
                                step: 1
                            };
                            shape.rank()
                        ];
                        slices[axis] = Slice {
                            start: Some(
                                isize::try_from(start)
                                    .map_err(|_| Error::ShapeOverflow(current_shape.clone()))?,
                            ),
                            stop: stop
                                .map(|value| {
                                    isize::try_from(value)
                                        .map_err(|_| Error::ShapeOverflow(current_shape.clone()))
                                })
                                .transpose()?,
                            step: -1,
                        };
                        graph.stride(value, slices)?
                    } else {
                        let mut bounds = current_shape
                            .dims()
                            .iter()
                            .map(|&end| (0, end))
                            .collect::<Vec<_>>();
                        bounds[axis] = (dimension - 1, dimension);
                        let part = graph.shrink(value, bounds)?;
                        let mut expanded = current_shape.dims().to_vec();
                        expanded[axis] = after;
                        graph.expand(part, Shape::new(expanded))?
                    };
                    pieces.push(part);
                }
                value = if pieces.len() == 1 {
                    value
                } else {
                    graph.cat(pieces[0], pieces[1..].to_vec(), axis as isize)?
                };
            }
            let bounds = signed_pad_bounds(graph.shape(value)?, padding)?;
            graph.shrink(value, bounds)
        }
    }
}

fn pad_mode_plan(
    graph: &Graph,
    input: NodeId,
    padding: Vec<(i64, i64)>,
    mode: PadMode,
    fill: Scalar,
) -> Result<PadModePlan> {
    let node = graph.node(input)?;
    if padding.len() != node.shape.rank() {
        return Err(Error::InvalidMovementRank {
            op: "pad_with_mode",
            expected: node.shape.rank(),
            actual: padding.len(),
        });
    }
    node.shape
        .numel()?
        .checked_mul(node.dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(node.shape.clone()))?;
    let mut rehearsal = graph.clone();
    let output = lower_pad_mode(&mut rehearsal, input, &padding, mode, fill)?;
    let output_shape = rehearsal.shape(output)?.clone();
    let output_dtype = rehearsal.dtype(output)?;
    output_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
    Ok(PadModePlan {
        padding,
        mode,
        fill,
        output_shape,
        output_dtype,
    })
}

fn pad_to_plan(
    graph: &Graph,
    input: NodeId,
    target: Vec<Option<usize>>,
    fill: Scalar,
) -> Result<PadToPlan> {
    let source = graph.node(input)?;
    let source_shape = source.shape.clone();
    if target.len() != source_shape.rank() {
        return Err(Error::InvalidMovementRank {
            op: "pad_to",
            expected: source_shape.rank(),
            actual: target.len(),
        });
    }
    source_shape
        .numel()?
        .checked_mul(source.dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(source_shape.clone()))?;
    let mut bounds = Vec::with_capacity(source_shape.rank());
    let mut positive = Vec::with_capacity(source_shape.rank());
    let mut output_dims = Vec::with_capacity(source_shape.rank());
    let mut changed = false;
    for (&current, wanted) in source_shape.dims().iter().zip(target) {
        let wanted = wanted.unwrap_or(current);
        changed |= wanted != current;
        bounds.push((0, current.min(wanted)));
        positive.push((0, wanted.saturating_sub(current)));
        output_dims.push(wanted);
    }
    let output_shape = Shape::new(output_dims);
    output_shape
        .numel()?
        .checked_mul(source.dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
    Ok(PadToPlan {
        source_shape,
        bounds,
        positive,
        changed,
        fill,
        output_shape,
    })
}

fn lower_pad_to(graph: &mut Graph, input: NodeId, plan: &PadToPlan) -> Result<NodeId> {
    let cropped = plan
        .bounds
        .iter()
        .zip(plan.source_shape.dims())
        .any(|(&(start, end), &dimension)| start != 0 || end != dimension);
    let padded = plan
        .positive
        .iter()
        .any(|&(before, after)| before != 0 || after != 0);
    let base = if cropped {
        graph.shrink(input, plan.bounds.clone())?
    } else {
        input
    };
    let base = if padded {
        graph.pad(base, plan.positive.clone(), cat_zero(graph.dtype(base)?))?
    } else {
        base
    };
    // In source `MovementMixin.pad_to` returns self for an unchanged target,
    // so the OpMixin's nonzero fill shell must not materialize or promote.
    if !plan.changed || pad_zero(plan.fill) {
        return Ok(base);
    }
    let mask =
        graph.lazy_full_with_dtype(plan.source_shape.clone(), Scalar::Bool(true), DType::Bool)?;
    let mask = if cropped {
        graph.shrink(mask, plan.bounds.clone())?
    } else {
        mask
    };
    let mask = if padded {
        graph.pad(mask, plan.positive.clone(), Scalar::Bool(false))?
    } else {
        mask
    };
    graph.where_false_scalar(mask, base, plan.fill)
}

fn shrink_to_plan(
    graph: &Graph,
    input: NodeId,
    target: Vec<Option<usize>>,
) -> Result<ShrinkToPlan> {
    let source = graph.node(input)?;
    let shape = source.shape.clone();
    if target.len() != shape.rank() {
        return Err(Error::InvalidMovementRank {
            op: "shrink_to",
            expected: shape.rank(),
            actual: target.len(),
        });
    }
    shape
        .numel()?
        .checked_mul(source.dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let bounds = shape
        .dims()
        .iter()
        .zip(target)
        .enumerate()
        .map(|(axis, (&dimension, target))| {
            let end = target.unwrap_or(dimension);
            if end > dimension {
                return Err(Error::InvalidBounds {
                    axis,
                    start: 0,
                    end,
                    dim: dimension,
                });
            }
            Ok((0, end))
        })
        .collect::<Result<Vec<_>>>()?;
    let output_shape = Shape::new(bounds.iter().map(|&(_, end)| end).collect::<Vec<_>>());
    output_shape
        .numel()?
        .checked_mul(source.dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
    Ok(ShrinkToPlan {
        bounds,
        output_shape,
    })
}

fn cat_source_lub(lhs: DType, rhs: DType) -> DType {
    if matches!(
        (lhs, rhs),
        (DType::I64, DType::U64) | (DType::U64, DType::I64)
    ) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

fn cat_zero(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(0),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => Scalar::U(0),
        DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Scalar::F(0.0),
    }
}

fn scatter_one(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(true),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(1),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => Scalar::U(1),
        DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Scalar::F(1.0),
    }
}

fn scatter_max_identity(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        DType::I8 => Scalar::I(i8::MIN.into()),
        DType::U8 => Scalar::U(0),
        DType::I16 => Scalar::I(i16::MIN.into()),
        DType::U16 => Scalar::U(0),
        DType::I32 => Scalar::I(i32::MIN.into()),
        DType::U32 => Scalar::U(0),
        DType::I64 => Scalar::I(i64::MIN),
        DType::U64 => Scalar::U(0),
        DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Scalar::F(f64::NEG_INFINITY),
    }
}

fn scatter_min_identity(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(true),
        DType::I8 => Scalar::I(i8::MAX.into()),
        DType::U8 => Scalar::U(u8::MAX.into()),
        DType::I16 => Scalar::I(i16::MAX.into()),
        DType::U16 => Scalar::U(u16::MAX.into()),
        DType::I32 => Scalar::I(i32::MAX.into()),
        DType::U32 => Scalar::U(u32::MAX.into()),
        DType::I64 => Scalar::I(i64::MAX),
        DType::U64 => Scalar::U(u64::MAX),
        DType::F8E4M3
        | DType::F8E5M2
        | DType::F8E4M3FNUZ
        | DType::F8E5M2FNUZ
        | DType::F16
        | DType::BF16
        | DType::F32
        | DType::F64 => Scalar::F(f64::INFINITY),
    }
}

fn scatter_pad_to(
    graph: &mut Graph,
    input: NodeId,
    target: &Shape,
    fill: Scalar,
) -> Result<NodeId> {
    let shape = graph.shape(input)?.clone();
    if shape.rank() != target.rank() {
        return Err(Error::InvalidMovementRank {
            op: "scatter_reduce pad_to",
            expected: target.rank(),
            actual: shape.rank(),
        });
    }
    let padding = shape
        .dims()
        .iter()
        .zip(target.dims())
        .enumerate()
        .map(|(axis, (&current, &wanted))| {
            wanted
                .checked_sub(current)
                .map(|after| (0, after))
                .ok_or(Error::InvalidBounds {
                    axis,
                    start: current,
                    end: current,
                    dim: wanted,
                })
        })
        .collect::<Result<Vec<_>>>()?;
    graph.pad(input, padding, fill)
}

fn scatter_reduce_plan(
    graph: &Graph,
    base: NodeId,
    index: NodeId,
    src: NodeId,
    dim: isize,
    kind: ScatterReduceKind,
    include_self: bool,
) -> Result<ScatterReducePlan> {
    let base_node = graph.node(base)?;
    let index_node = graph.node(index)?;
    let src_node = graph.node(src)?;
    let base_shape = base_node.shape.clone();
    let index_shape = index_node.shape.clone();
    let src_shape = src_node.shape.clone();
    let base_dtype = base_node.dtype;
    if !index_node.dtype.is_integer() {
        return Err(Error::InvalidRandom {
            reason: "scatter_reduce requires integer indices",
        });
    }
    if base_shape.rank() != index_shape.rank() || base_shape.rank() != src_shape.rank() {
        return Err(Error::InvalidMovementRank {
            op: "scatter_reduce",
            expected: base_shape.rank(),
            actual: index_shape.rank().max(src_shape.rank()),
        });
    }
    if src_node.dtype != base_dtype {
        return Err(Error::InvalidElementwiseDType {
            op: "scatter_reduce",
            actual: src_node.dtype,
        });
    }
    let dim = normalize_axes(base, base_shape.rank(), Some(vec![dim]))?[0];
    for (axis, ((&base_extent, &index_extent), &src_extent)) in base_shape
        .dims()
        .iter()
        .zip(index_shape.dims())
        .zip(src_shape.dims())
        .enumerate()
    {
        if src_extent < index_extent || (axis != dim && base_extent < index_extent) {
            return Err(Error::ShapeMismatch {
                op: "scatter_reduce",
                lhs: base_shape.clone(),
                rhs: index_shape.clone(),
            });
        }
    }
    for (shape, dtype) in [
        (&base_shape, base_dtype),
        (&index_shape, index_node.dtype),
        (&src_shape, src_node.dtype),
    ] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    }
    // Clone rehearsal is deliberately the complete source composition. It
    // includes the synthetic update axis and catches a late expanded/padded
    // byte overflow before a live constant, cast, view, or reduction exists.
    let mut rehearsal = graph.clone();
    let plan = ScatterReducePlan {
        dim,
        index_shape,
        base_shape,
        base_dtype,
        kind,
        include_self,
        output_shape: base_node.shape.clone(),
        output_dtype: base_dtype,
    };
    let output = lower_scatter_reduce(&mut rehearsal, base, index, src, &plan)?;
    let output_shape = rehearsal.shape(output)?.clone();
    let output_dtype = rehearsal.dtype(output)?;
    output_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
    Ok(ScatterReducePlan {
        output_shape,
        output_dtype,
        ..plan
    })
}

fn lower_pre_scatter(
    graph: &mut Graph,
    index: NodeId,
    src: NodeId,
    dim: usize,
    index_shape: &Shape,
    base_shape: &Shape,
    base_dtype: DType,
) -> Result<(NodeId, NodeId)> {
    let crop = index_shape
        .dims()
        .iter()
        .map(|&extent| (0, extent))
        .collect::<Vec<_>>();
    let src = graph.shrink(src, crop)?;
    let mut expanded = index_shape.dims().to_vec();
    expanded.push(base_shape.dims()[dim]);
    let src = graph.unsqueeze(src, -1)?;
    let src = graph.expand(src, Shape::new(expanded))?;
    let src = graph.transpose(src, -1, dim as isize)?;
    let mask = graph.one_hot_bool(index, base_shape.dims()[dim])?;
    let mask = graph.transpose(mask, -1, dim as isize)?;
    let mut target = base_shape.dims().to_vec();
    target.push(index_shape.dims()[dim]);
    let target = Shape::new(target);
    let src = scatter_pad_to(graph, src, &target, cat_zero(base_dtype))?;
    let mask = scatter_pad_to(graph, mask, &target, Scalar::Bool(false))?;
    Ok((src, mask))
}

fn lower_scatter_reduce(
    graph: &mut Graph,
    base: NodeId,
    index: NodeId,
    src: NodeId,
    plan: &ScatterReducePlan,
) -> Result<NodeId> {
    let (src, mask) = lower_pre_scatter(
        graph,
        index,
        src,
        plan.dim,
        &plan.index_shape,
        &plan.base_shape,
        plan.base_dtype,
    )?;
    let axes = Some(vec![-1]);
    let inverse_mask = |graph: &mut Graph| -> Result<NodeId> {
        let any = graph.any(mask, axes.clone(), false)?;
        graph.logical_not(any)
    };
    let no_self = |graph: &mut Graph, a: NodeId, b: Scalar| -> Result<NodeId> {
        let inverse = inverse_mask(graph)?;
        graph.where_false_scalar(inverse, a, b)
    };
    let selected_sum = |graph: &mut Graph| -> Result<NodeId> {
        let selected = graph.where_false_scalar(mask, src, cat_zero(plan.base_dtype))?;
        graph.sum_with_options(selected, axes.clone(), false, None)
    };
    match plan.kind {
        ScatterReduceKind::Sum => {
            let updates = selected_sum(graph)?;
            let self_or_zero = if plan.include_self {
                base
            } else {
                no_self(graph, base, cat_zero(plan.base_dtype))?
            };
            graph.add(updates, self_or_zero)
        }
        ScatterReduceKind::Prod => {
            let selected = graph.where_false_scalar(mask, src, scatter_one(plan.base_dtype))?;
            let updates = graph.prod_with_options(selected, axes.clone(), false, None)?;
            let self_or_one = if plan.include_self {
                base
            } else {
                no_self(graph, base, scatter_one(plan.base_dtype))?
            };
            graph.mul(updates, self_or_one)
        }
        ScatterReduceKind::Amax => {
            let identity = scatter_max_identity(plan.base_dtype);
            let selected = graph.where_false_scalar(mask, src, identity)?;
            let updates = graph.max_with_axes(selected, axes.clone(), false)?;
            let self_or_identity = if plan.include_self {
                base
            } else {
                no_self(graph, base, identity)?
            };
            graph.maximum(updates, self_or_identity)
        }
        ScatterReduceKind::Amin => {
            let identity = scatter_min_identity(plan.base_dtype);
            let selected = graph.where_false_scalar(mask, src, identity)?;
            let updates = graph.min_with_axes(selected, axes.clone(), false)?;
            let self_or_identity = if plan.include_self {
                base
            } else {
                no_self(graph, base, identity)?
            };
            graph.minimum(updates, self_or_identity)
        }
        ScatterReduceKind::Mean => {
            let one = Scalar::I(1);
            let zero = Scalar::I(0);
            let counted = graph.where_scalars(mask, one, zero)?;
            let count = graph.sum_with_options(counted, axes.clone(), false, None)?;
            let count = if plan.include_self {
                graph.add_scalar(count, one)?
            } else {
                let inverse = inverse_mask(graph)?;
                let missing_self = graph.where_scalars(inverse, one, zero)?;
                graph.add(count, missing_self)?
            };
            let updates = selected_sum(graph)?;
            let self_or_zero = if plan.include_self {
                base
            } else {
                no_self(graph, base, cat_zero(plan.base_dtype))?
            };
            let values = graph.add(updates, self_or_zero)?;
            graph.div(values, count)
        }
    }
}

fn scatter_plan(
    graph: &Graph,
    base: NodeId,
    dim: isize,
    index: NodeId,
    source: ScatterSource,
    mode: ScatterMode,
) -> Result<ScatterPlan> {
    let base_node = graph.node(base)?;
    let index_node = graph.node(index)?;
    let base_shape = base_node.shape.clone();
    let index_shape = index_node.shape.clone();
    if !index_node.dtype.is_integer() {
        return Err(Error::InvalidRandom {
            reason: "scatter requires integer indices",
        });
    }
    if base_shape.rank() != index_shape.rank() {
        return Err(Error::InvalidMovementRank {
            op: "scatter",
            expected: base_shape.rank(),
            actual: index_shape.rank(),
        });
    }
    let dim = normalize_axes(base, base_shape.rank(), Some(vec![dim]))?[0];
    let (src_shape, src_dtype) = match source {
        ScatterSource::Tensor(src) => {
            if mode != ScatterMode::Replace {
                return Err(Error::InvalidRandom {
                    reason: "non-scalar src is not supported with scatter reduce; use scatter_reduce",
                });
            }
            let src_node = graph.node(src)?;
            (src_node.shape.clone(), src_node.dtype)
        }
        ScatterSource::Scalar(_) => (index_shape.clone(), base_node.dtype),
    };
    if src_shape.rank() != base_shape.rank() {
        return Err(Error::InvalidMovementRank {
            op: "scatter",
            expected: base_shape.rank(),
            actual: src_shape.rank(),
        });
    }
    if src_dtype != base_node.dtype {
        return Err(Error::InvalidElementwiseDType {
            op: "scatter",
            actual: src_dtype,
        });
    }
    for (axis, ((&base_extent, &index_extent), &src_extent)) in base_shape
        .dims()
        .iter()
        .zip(index_shape.dims())
        .zip(src_shape.dims())
        .enumerate()
    {
        if src_extent < index_extent || (axis != dim && base_extent < index_extent) {
            return Err(Error::ShapeMismatch {
                op: "scatter",
                lhs: base_shape.clone(),
                rhs: index_shape.clone(),
            });
        }
    }
    for (shape, dtype) in [
        (&base_shape, base_node.dtype),
        (&index_shape, index_node.dtype),
        (&src_shape, src_dtype),
    ] {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    }
    let plan = ScatterPlan {
        dim,
        index_shape,
        base_shape,
        base_dtype: base_node.dtype,
        source,
        mode,
        output_shape: base_node.shape.clone(),
        output_dtype: base_node.dtype,
    };
    let mut rehearsal = graph.clone();
    let output = lower_scatter(&mut rehearsal, base, index, &plan)?;
    let output_shape = rehearsal.shape(output)?.clone();
    let output_dtype = rehearsal.dtype(output)?;
    output_shape
        .numel()?
        .checked_mul(output_dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
    Ok(ScatterPlan {
        output_shape,
        output_dtype,
        ..plan
    })
}

fn lower_scatter(
    graph: &mut Graph,
    base: NodeId,
    index: NodeId,
    plan: &ScatterPlan,
) -> Result<NodeId> {
    let src = match plan.source {
        ScatterSource::Tensor(src) => src,
        ScatterSource::Scalar(value) => {
            graph.lazy_full_with_dtype(plan.index_shape.clone(), value, plan.base_dtype)?
        }
    };
    match plan.mode {
        ScatterMode::Add | ScatterMode::Multiply => {
            let reduction = ScatterReducePlan {
                dim: plan.dim,
                index_shape: plan.index_shape.clone(),
                base_shape: plan.base_shape.clone(),
                base_dtype: plan.base_dtype,
                kind: if plan.mode == ScatterMode::Add {
                    ScatterReduceKind::Sum
                } else {
                    ScatterReduceKind::Prod
                },
                include_self: true,
                output_shape: plan.base_shape.clone(),
                output_dtype: plan.base_dtype,
            };
            lower_scatter_reduce(graph, base, index, src, &reduction)
        }
        ScatterMode::Replace => {
            let (values, mask) = lower_pre_scatter(
                graph,
                index,
                src,
                plan.dim,
                &plan.index_shape,
                &plan.base_shape,
                plan.base_dtype,
            )?;
            let mut parts = graph
                .split(mask, SplitSections::Uniform(1), -1)?
                .into_iter()
                .zip(graph.split(values, SplitSections::Uniform(1), -1)?);
            let (mut mask, mut values) = parts.next().ok_or(Error::InvalidRandom {
                reason: "scatter masked merge requires a synthetic axis",
            })?;
            for (next_mask, next_value) in parts {
                values = graph.select(next_mask, next_value, values)?;
                mask = graph.logical_or(mask, next_mask)?;
            }
            let mask = graph.squeeze(mask, Some(-1))?;
            let values = graph.squeeze(values, Some(-1))?;
            graph.select(mask, values, base)
        }
    }
}

fn chunk_plan(graph: &Graph, input: NodeId, chunks: usize, axis: isize) -> Result<ChunkPlan> {
    if chunks == 0 {
        return Err(Error::InvalidSplit {
            reason: "chunk count must be positive",
        });
    }
    let source = graph.node(input)?;
    let shape = source.shape.clone();
    shape
        .numel()?
        .checked_mul(source.dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let rank = shape.rank() as isize;
    let axis = if axis < 0 {
        axis.checked_add(rank).ok_or(Error::InvalidAxis {
            node: input,
            axis: usize::MAX,
            rank: rank as usize,
        })?
    } else {
        axis
    };
    if axis < 0 || axis >= rank {
        return Err(Error::InvalidAxis {
            node: input,
            axis: usize::MAX,
            rank: rank as usize,
        });
    }
    let axis = axis as usize;
    let axis_len = shape.dims()[axis];
    let ranges = if axis_len == 0 {
        vec![(0, 0); chunks]
    } else {
        let width = axis_len / chunks + usize::from(axis_len % chunks != 0);
        (0..axis_len)
            .step_by(width)
            .map(|start| (start, start.saturating_add(width).min(axis_len)))
            .collect::<Vec<_>>()
    };
    let bounds = ranges
        .into_iter()
        .map(|(start, end)| {
            shape
                .dims()
                .iter()
                .enumerate()
                .map(|(dimension, &size)| {
                    if dimension == axis {
                        (start, end)
                    } else {
                        (0, size)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // `shrink` will receive these exact checked bounds below. Validate every
    // descriptor and byte extent here so no earlier member of this multi-view
    // result can publish when a later descriptor is malformed or too large.
    for view in &bounds {
        let output = Shape::new(
            view.iter()
                .map(|(start, end)| end - start)
                .collect::<Vec<_>>(),
        );
        output
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output.clone()))?;
    }
    Ok(ChunkPlan { bounds })
}

fn split_plan(
    graph: &Graph,
    input: NodeId,
    sections: SplitSections,
    axis: isize,
) -> Result<SplitPlan> {
    let source = graph.node(input)?;
    let shape = source.shape.clone();
    shape
        .numel()?
        .checked_mul(source.dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let rank = shape.rank() as isize;
    let axis = if axis < 0 {
        axis.checked_add(rank).ok_or(Error::InvalidAxis {
            node: input,
            axis: usize::MAX,
            rank: rank as usize,
        })?
    } else {
        axis
    };
    if axis < 0 || axis >= rank {
        return Err(Error::InvalidAxis {
            node: input,
            axis: usize::MAX,
            rank: rank as usize,
        });
    }
    let axis = axis as usize;
    let axis_len = shape.dims()[axis];
    let lengths = match sections {
        SplitSections::Uniform(size) => {
            if axis_len == 0 {
                // tinygrad's `range(0, max(1, 0), max(1, size))` makes one
                // empty section, including when the requested size is zero.
                vec![0]
            } else if size == 0 {
                return Err(Error::InvalidSplit {
                    reason: "split size must be positive for a non-empty axis",
                });
            } else {
                (0..axis_len)
                    .step_by(size)
                    .map(|start| size.min(axis_len - start))
                    .collect()
            }
        }
        SplitSections::Explicit(lengths) => {
            let total = lengths.iter().try_fold(0usize, |total, &length| {
                total
                    .checked_add(length)
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            })?;
            if total != axis_len {
                return Err(Error::InvalidSplit {
                    reason: "section sizes must sum exactly to the selected axis",
                });
            }
            lengths
        }
    };
    let mut start = 0usize;
    let ranges = lengths
        .into_iter()
        .map(|length| {
            let end = start
                .checked_add(length)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            let range = (start, end);
            start = end;
            Ok(range)
        })
        .collect::<Result<Vec<_>>>()?;
    let bounds = ranges
        .into_iter()
        .map(|(start, end)| {
            shape
                .dims()
                .iter()
                .enumerate()
                .map(|(dimension, &size)| {
                    if dimension == axis {
                        (start, end)
                    } else {
                        (0, size)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // A split is a multi-output operation. Ensure every source and output
    // descriptor is representable before its first Shrink is appended.
    for view in &bounds {
        let output = Shape::new(
            view.iter()
                .map(|(start, end)| end - start)
                .collect::<Vec<_>>(),
        );
        output
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output.clone()))?;
    }
    Ok(SplitPlan { bounds })
}

fn cat_plan(graph: &Graph, input: NodeId, args: Vec<NodeId>, dim: isize) -> Result<CatPlan> {
    let mut inputs = Vec::with_capacity(args.len() + 1);
    inputs.push(input);
    inputs.extend(args);
    let descriptors = inputs
        .iter()
        .map(|&input| {
            let node = graph.node(input)?;
            node.shape
                .numel()?
                .checked_mul(node.dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(node.shape.clone()))?;
            Ok((node.shape.clone(), node.dtype))
        })
        .collect::<Result<Vec<_>>>()?;
    let first_shape = &descriptors[0].0;
    let rank = first_shape.rank();
    // `Tensor._resolve_dim` accepts 0 for a scalar but `cat` subsequently
    // needs an existing concatenation axis, so the literal stack/flatten path
    // rejects scalar inputs before creating its first view.
    if rank == 0 {
        return Err(Error::InvalidAxis {
            node: input,
            axis: 0,
            rank,
        });
    }
    let rank_isize =
        isize::try_from(rank).map_err(|_| Error::ShapeOverflow(first_shape.clone()))?;
    let resolved = if dim < 0 {
        dim.checked_add(rank_isize).ok_or(Error::InvalidAxis {
            node: input,
            axis: usize::MAX,
            rank,
        })?
    } else {
        dim
    };
    if resolved < 0 || resolved >= rank_isize {
        return Err(Error::InvalidAxis {
            node: input,
            axis: usize::MAX,
            rank,
        });
    }
    let axis = resolved as usize;
    let shapes = descriptors
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    if shapes.iter().any(|shape| {
        shape.rank() != rank
            || shape
                .dims()
                .iter()
                .enumerate()
                .any(|(index, extent)| index != axis && *extent != first_shape.dims()[index])
    }) {
        return Err(Error::InvalidConcat { axis, shapes });
    }
    let axis_extent = descriptors.iter().try_fold(0usize, |total, (shape, _)| {
        total
            .checked_add(shape.dims()[axis])
            .ok_or_else(|| Error::ShapeOverflow(first_shape.clone()))
    })?;
    let mut output_dims = first_shape.dims().to_vec();
    output_dims[axis] = axis_extent;
    let output_shape = Shape::new(output_dims);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    let equal_axis_extents = descriptors
        .iter()
        .all(|(shape, _)| shape.dims()[axis] == first_shape.dims()[axis]);
    let (output_dtype, lowering) = if equal_axis_extents {
        let output_dtype = descriptors
            .iter()
            .skip(1)
            .fold(descriptors[0].1, |dtype, (_, next)| {
                cat_source_lub(dtype, *next)
            });
        let mut stack_dims = first_shape.dims().to_vec();
        stack_dims.insert(axis, inputs.len());
        let stack_shape = Shape::new(stack_dims);
        for (shape, _) in &descriptors {
            extent(shape, output_dtype)?; // stack's all-input cast
        }
        extent(&stack_shape, output_dtype)?;
        extent(&output_shape, output_dtype)?; // flatten
        (output_dtype, CatLowering::Stack)
    } else {
        let mut offset = 0usize;
        let paddings = descriptors
            .iter()
            .map(|(shape, dtype)| {
                let before = offset;
                offset = offset
                    .checked_add(shape.dims()[axis])
                    .ok_or_else(|| Error::ShapeOverflow(first_shape.clone()))?;
                let after = axis_extent
                    .checked_sub(offset)
                    .ok_or_else(|| Error::ShapeOverflow(first_shape.clone()))?;
                extent(&output_shape, *dtype)?; // source-typed zero pad
                Ok(shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        if index == axis {
                            (before, after)
                        } else {
                            (0, 0)
                        }
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut output_dtype = descriptors[0].1;
        for (_, dtype) in descriptors.iter().skip(1) {
            let prior = output_dtype;
            output_dtype = cat_source_lub(prior, *dtype);
            // Source-order `usum` invokes `_broadcasted` at every ADD.
            extent(&output_shape, prior)?;
            extent(&output_shape, *dtype)?;
            extent(&output_shape, output_dtype)?;
        }
        (output_dtype, CatLowering::PadSum { paddings })
    };
    Ok(CatPlan {
        identity: inputs.len() == 1,
        inputs,
        axis,
        output_shape,
        output_dtype,
        lowering,
    })
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            dynamic_nodes: Vec::new(),
            id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            grad_enabled: true,
            parameter_bindings: BTreeMap::new(),
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preserves a floating probability tensor after CPU validation that every
    /// lane is finite and nonnegative and every row along `axis` has positive
    /// total weight. Validation happens at realization, before dependent work.
    pub fn tensor_guard_distribution(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let source = self.node(input)?;
        if !source.dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "tensor guard distribution requires floating dtype",
            });
        }
        let rank = source.shape.rank();
        if !(1..=2).contains(&rank) {
            return Err(Error::InvalidRandom {
                reason: "tensor guard distribution requires rank one or two",
            });
        }
        let axis = if axis < 0 { axis + rank as isize } else { axis };
        if axis < 0 || axis >= rank as isize {
            return Err(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank,
            });
        }
        Ok(self.push(
            Op::TensorGuard {
                input,
                axis: axis as usize,
            },
            source.shape.clone(),
            source.dtype,
        ))
    }

    /// Stable graph identity used by diagnostics and graph-owned resources.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number of graph nodes currently allocated.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns row-major coordinates of every nonzero input element. Its
    /// concrete shape is `[count, input_rank]` after realization.
    pub fn nonzero(&mut self, input: NodeId) -> Result<DynamicNodeId> {
        let shape = &self.node(input)?.shape;
        let rank = shape.rank();
        let dtype = nonzero_coordinate_dtype(shape)?;
        let id = DynamicNodeId {
            graph: self.id,
            index: self.dynamic_nodes.len(),
        };
        self.dynamic_nodes
            .push(DynamicNode::nonzero(id, input, rank, dtype));
        Ok(id)
    }

    /// Fixed-shape `nonzero(size=...)`, matching tinygrad's row-major
    /// pad/truncate form without introducing a new dynamic-result primitive.
    ///
    /// The result has shape `[size, input_rank]`. Every extent and the complete
    /// literal composition are clone-rehearsed before this method appends its
    /// comparison, movement, or selection nodes. Coordinate ranges are planned
    /// from their exact length, so the final valid I64 coordinate needs no
    /// unrepresentable exclusive endpoint.
    pub fn nonzero_fixed(&mut self, input: NodeId, size: usize, fill: Scalar) -> Result<NodeId> {
        let mut rehearsal = self.clone();
        rehearsal.lower_nonzero_fixed(input, size, fill)?;
        self.lower_nonzero_fixed(input, size, fill)
    }

    fn lower_nonzero_fixed(&mut self, input: NodeId, size: usize, fill: Scalar) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let count = shape.numel()?;
        let rank = shape.rank();
        let coordinate_dtype = nonzero_coordinate_dtype(&shape)?;
        let selection_len = size
            .checked_mul(rank)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        if rank == 0 || count == 0 {
            return self.lazy_full_with_dtype([size, rank], fill, coordinate_dtype);
        }
        let range_plans = shape
            .dims()
            .iter()
            .copied()
            .map(nonzero_coordinate_range_plan)
            .collect::<Result<Vec<_>>>()?;

        let zero = self.constant(TensorData::scalar_with_dtype(Scalar::I(0), dtype));
        let mask = self.ne(input, zero)?;
        let flattened_mask = self.reshape(mask, Shape::from([count]))?;
        let mut coordinates = Vec::with_capacity(rank);
        for (axis, plan) in range_plans.into_iter().enumerate() {
            let dimension = plan.shape.dims()[0];
            let range = self.lower_lazy_arange(plan)?;
            let mut coordinate_shape = vec![1; rank];
            coordinate_shape[axis] = dimension;
            let range = self.reshape(range, Shape::new(coordinate_shape))?;
            let range = self.expand(range, shape.clone())?;
            coordinates.push(self.reshape(range, Shape::from([count]))?);
        }
        let coordinates = self.stack(coordinates, -1)?;
        let expanded_mask = self.unsqueeze(flattened_mask, -1)?;
        let expanded_mask = self.expand(expanded_mask, Shape::from([count, rank]))?;
        let selected = self.masked_select(coordinates, expanded_mask, selection_len, fill)?;
        self.reshape(selected, Shape::from([size, rank]))
    }

    /// Unbounded, row-major boolean selection. Unlike [`Self::masked_select`]
    /// this result has runtime shape `[selected_count]`.
    pub fn masked_select_dynamic(&mut self, input: NodeId, mask: NodeId) -> Result<DynamicNodeId> {
        let source = self.node(input)?;
        let mask_node = self.node(mask)?;
        if mask_node.dtype != DType::Bool {
            return Err(Error::InvalidLogicalDType {
                op: "masked_select_dynamic",
                actual: mask_node.dtype,
            });
        }
        if mask_node.shape.broadcast_with(&source.shape).as_ref() != Ok(&source.shape) {
            return Err(Error::InvalidIndexedShape {
                op: "masked_select_dynamic",
                input: source.shape.clone(),
                index: mask_node.shape.clone(),
            });
        }
        let id = DynamicNodeId {
            graph: self.id,
            index: self.dynamic_nodes.len(),
        };
        self.dynamic_nodes
            .push(DynamicNode::masked_select(id, input, mask, source.dtype));
        Ok(id)
    }

    /// Builds the exact CPU allocation contract for one runtime-cardinality
    /// result without introducing a bounded placeholder into the static graph.
    pub fn dynamic_allocation_plan(
        &self,
        output: DynamicNodeId,
    ) -> std::result::Result<DynamicAllocationPlan, DynamicAllocationError> {
        DynamicAllocationPlan::for_output(self, output)
    }

    /// Reduces a dynamic result to a scalar dynamic loss.
    pub fn dynamic_sum(&mut self, input: DynamicNodeId) -> Result<DynamicNodeId> {
        let source = self.dynamic_node(input)?;
        let dtype = super::dynamic_reduction_dtypes(source.dtype, ReduceKind::Sum)
            .expect("dynamic Sum is supported")
            .output;
        let id = DynamicNodeId {
            graph: self.id,
            index: self.dynamic_nodes.len(),
        };
        self.dynamic_nodes.push(DynamicNode::sum(input, dtype));
        Ok(id)
    }

    /// Reduces every realized element of a dynamic result to one scalar using
    /// the ordinary mean dtype policy.
    pub fn dynamic_mean(&mut self, input: DynamicNodeId) -> Result<DynamicNodeId> {
        let source = self.dynamic_node(input)?;
        let dtype = super::dynamic_reduction_dtypes(source.dtype, ReduceKind::Mean)
            .expect("dynamic Mean is supported")
            .output;
        let id = DynamicNodeId {
            graph: self.id,
            index: self.dynamic_nodes.len(),
        };
        self.dynamic_nodes.push(DynamicNode::mean(input, dtype));
        Ok(id)
    }

    /// Applies a supported unary operation pointwise to a dynamic value.
    pub fn dynamic_unary(&mut self, input: DynamicNodeId, op: UnaryOp) -> Result<DynamicNodeId> {
        if !matches!(op, UnaryOp::Neg | UnaryOp::Square) {
            return Err(Error::NonDifferentiableIndexing(
                "unsupported dynamic unary",
            ));
        }
        let source = self.dynamic_node(input)?;
        let dtype = source.dtype;
        let output = source.output;
        if !dtype.is_float() {
            return Err(Error::InvalidElementwiseDType {
                op: op.name(),
                actual: dtype,
            });
        }
        let id = DynamicNodeId {
            graph: self.id,
            index: self.dynamic_nodes.len(),
        };
        self.dynamic_nodes
            .push(DynamicNode::unary(op, input, output, dtype));
        Ok(id)
    }

    /// Pointwise dynamic arithmetic. Static operands must be scalar.
    pub fn dynamic_binary(
        &mut self,
        lhs: DynamicNodeId,
        rhs: DynamicInput,
        op: BinaryOp,
    ) -> Result<DynamicNodeId> {
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) {
            return Err(Error::NonDifferentiableIndexing(
                "unsupported dynamic binary",
            ));
        }
        let lhs_node = self.dynamic_node(lhs)?;
        let lhs_output = lhs_node.output;
        let mut output = lhs_output;
        let rhs_dtype = match rhs {
            DynamicInput::Dynamic(id) => {
                let rhs_node = self.dynamic_node(id)?;
                output = match (lhs_output, rhs_node.output) {
                    (DynamicOutputShape::Scalar, rhs) => rhs,
                    (lhs, DynamicOutputShape::Scalar) => lhs,
                    (lhs, rhs) if lhs == rhs => lhs,
                    _ => return Err(Error::InvalidIndex),
                };
                rhs_node.dtype
            }
            DynamicInput::StaticScalar(id) => {
                let node = self.node(id)?;
                if node.shape.numel()? != 1 {
                    return Err(Error::InvalidIndex);
                }
                node.dtype
            }
        };
        let dtype = source_lub(lhs_node.dtype, rhs_dtype);
        if !dtype.is_float() {
            return Err(Error::InvalidElementwiseDType {
                op: "dynamic_binary",
                actual: dtype,
            });
        }
        let id = DynamicNodeId {
            graph: self.id,
            index: self.dynamic_nodes.len(),
        };
        self.dynamic_nodes.push(DynamicNode::binary(
            op,
            DynamicInput::Dynamic(lhs),
            rhs,
            output,
            dtype,
        ));
        Ok(id)
    }

    pub(crate) fn dynamic_node(&self, id: DynamicNodeId) -> Result<&DynamicNode> {
        if id.graph != self.id {
            return Err(Error::ParameterGraphMismatch);
        }
        self.dynamic_nodes.get(id.index).ok_or(Error::InvalidIndex)
    }

    pub(crate) fn bind_parameter(&mut self, snapshot: ParameterSnapshot) -> Result<NodeId> {
        let key = (snapshot.identity, snapshot.version);
        if let Some(binding) = self.parameter_bindings.get(&key) {
            return Ok(binding.node);
        }
        let input_name = format!("{}_v{}", snapshot.input_name, snapshot.version);
        let node = self.input_dtype_requires_grad(
            input_name.clone(),
            snapshot.shape,
            snapshot.dtype,
            snapshot.trainable,
        );
        self.parameter_bindings.insert(
            key,
            ParameterBinding {
                node,
                input_name,
                data: snapshot.data,
            },
        );
        Ok(node)
    }

    pub(crate) fn bound_parameter_node(
        &self,
        identity: ParameterId,
        version: u64,
    ) -> Option<NodeId> {
        self.parameter_bindings
            .get(&(identity, version))
            .map(|binding| binding.node)
    }

    /// Returns every immutable parameter value captured by this graph.
    pub fn parameter_bindings(&self) -> HashMap<String, TensorData> {
        self.parameter_bindings
            .values()
            .map(|binding| (binding.input_name.clone(), binding.data.clone()))
            .collect()
    }

    pub(crate) fn parameter_bindings_for(
        &self,
        identities: &BTreeSet<ParameterId>,
    ) -> HashMap<String, TensorData> {
        self.parameter_bindings
            .iter()
            .filter(|((identity, _), _)| identities.contains(identity))
            .map(|(_, binding)| (binding.input_name.clone(), binding.data.clone()))
            .collect()
    }

    pub fn input(&mut self, name: impl Into<String>, shape: impl Into<Shape>) -> NodeId {
        self.input_dtype(name, shape, DType::F32)
    }

    /// Adds an input after explicitly specializing a symbolic shape.  This is
    /// intentionally a one-way boundary: graph nodes and CPU allocation retain
    /// the existing concrete `Shape` invariant.
    pub fn input_symbolic(
        &mut self,
        name: impl Into<String>,
        shape: &SymbolicShape,
        bindings: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<NodeId> {
        Ok(self.input(name, shape.bind_for_graph(bindings)?))
    }

    pub fn input_dtype(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> NodeId {
        self.input_dtype_requires_grad(name, shape, dtype, dtype.is_float())
    }

    /// Adds an input leaf with an explicit gradient-tracking contract.
    pub fn input_dtype_requires_grad(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
        requires_grad: bool,
    ) -> NodeId {
        self.push_with_grad(
            Op::Input { name: name.into() },
            shape.into(),
            dtype,
            requires_grad && dtype.is_float(),
        )
    }

    pub fn constant(&mut self, data: TensorData) -> NodeId {
        let shape = data.shape().clone();
        let dtype = data.dtype();
        self.push_with_grad(Op::Constant(data), shape, dtype, false)
    }

    /// Adds a scalar literal after resolving it to a concrete default dtype.
    pub fn constant_literal(&mut self, literal: LiteralScalar) -> Result<NodeId> {
        let data = TensorData::from_scalars([], literal.default_dtype(), [literal.scalar()])?;
        Ok(self.constant(data))
    }

    /// Applies a binary operation with a right scalar literal resolved against
    /// the left node's concrete dtype before lowering.
    pub fn binary_literal(
        &mut self,
        op: BinaryOp,
        lhs: NodeId,
        literal: LiteralScalar,
    ) -> Result<NodeId> {
        let dtype = self.node(lhs)?.dtype;
        let data = TensorData::from_scalars([], literal.dtype_against(dtype), [literal.scalar()])?;
        let rhs = self.constant(data);
        self.binary(op, lhs, rhs)
    }

    /// Applies a binary operation with a left scalar literal resolved against
    /// the right node's concrete dtype before lowering.
    pub fn literal_binary(
        &mut self,
        literal: LiteralScalar,
        op: BinaryOp,
        rhs: NodeId,
    ) -> Result<NodeId> {
        let dtype = self.node(rhs)?.dtype;
        let data = TensorData::from_scalars([], literal.dtype_against(dtype), [literal.scalar()])?;
        let lhs = self.constant(data);
        self.binary(op, lhs, rhs)
    }

    /// Returns whether future graph operations record reverse-mode edges.
    pub fn grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    /// Runs a graph-building closure with reverse-mode recording disabled.
    /// The guard is stored on this graph only, so it is thread-safe and cannot
    /// leak to another graph instance.
    pub fn no_grad<T>(&mut self, build: impl FnOnce(&mut Self) -> T) -> T {
        let previous = self.grad_enabled;
        self.grad_enabled = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(self)));
        self.grad_enabled = previous;
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Creates a value-sharing node that is a new gradient leaf.
    pub fn detach(&mut self, input: NodeId) -> Result<NodeId> {
        let node = self.node(input)?;
        node.shape
            .numel()?
            .checked_mul(node.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(node.shape.clone()))?;
        Ok(self.push_with_grad(
            Op::Detach { input },
            node.shape.clone(),
            node.dtype,
            node.dtype.is_float(),
        ))
    }

    /// Returns the explicit gradient-tracking state of a graph node.
    pub fn requires_grad(&self, id: NodeId) -> Result<bool> {
        Ok(self.node(id)?.requires_grad)
    }

    pub(crate) fn backward_slice_contains(&self, loss: NodeId, target: NodeId) -> Result<bool> {
        self.node(loss)?;
        self.node(target)?;
        let mut pending = vec![loss];
        let mut seen = BTreeSet::new();
        while let Some(node) = pending.pop() {
            if !seen.insert(node) {
                continue;
            }
            if node == target {
                return Ok(true);
            }
            pending.extend(self.reverse_inputs(node)?);
        }
        Ok(false)
    }

    /// Graph-checked value edges that a local reverse rule can actually
    /// traverse. `Op::backward_inputs` owns the structural projection; this
    /// seam adds descriptor-dependent barriers without creating another op
    /// taxonomy.
    pub(crate) fn reverse_inputs(&self, node: NodeId) -> Result<Vec<NodeId>> {
        let current = self.node(node)?;
        if let Op::Cast { input, .. } = &current.op {
            return Ok(
                if current.dtype.is_float() && self.node(*input)?.dtype.is_float() {
                    vec![*input]
                } else {
                    vec![]
                },
            );
        }
        Ok(current.op.backward_inputs())
    }

    pub(crate) fn value_slice_contains(&self, loss: NodeId, target: NodeId) -> Result<bool> {
        self.node(loss)?;
        self.node(target)?;
        self.reaches_input(loss, target, |op| op.value_inputs())
    }

    pub(crate) fn value_slice_contains_detach(&self, loss: NodeId, target: NodeId) -> Result<bool> {
        self.node(loss)?;
        self.node(target)?;
        let mut pending = vec![(loss, false)];
        let mut seen = BTreeSet::new();
        while let Some((node, detached)) = pending.pop() {
            if !seen.insert((node.index(), detached)) {
                continue;
            }
            if node == target && detached {
                return Ok(true);
            }
            let op = &self.node(node)?.op;
            let detached = detached || matches!(op, Op::Detach { .. });
            pending.extend(op.value_inputs().into_iter().map(|input| (input, detached)));
        }
        Ok(false)
    }

    fn reaches_input(
        &self,
        root: NodeId,
        target: NodeId,
        inputs: impl Fn(&Op) -> Vec<NodeId>,
    ) -> Result<bool> {
        let mut pending = vec![root];
        let mut seen = BTreeSet::new();
        while let Some(node) = pending.pop() {
            if !seen.insert(node.index()) {
                continue;
            }
            if node == target {
                return Ok(true);
            }
            pending.extend(inputs(&self.node(node)?.op));
        }
        Ok(false)
    }

    pub fn sum(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        self.reduce(input, ReduceKind::Sum, Some(vec![axis as isize]), false)
    }

    /// Inclusive cumulative sum along one signed axis.
    pub fn cumsum(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        self.prefix_scan(input, axis, PrefixScanKind::Sum)
    }

    /// Inclusive cumulative product along one signed axis.
    pub fn cumprod(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        self.prefix_scan(input, axis, PrefixScanKind::Product)
    }

    /// Inclusive cumulative maxima and the first matching I32 index per prefix.
    pub fn cummax(&mut self, input: NodeId, axis: isize) -> Result<(NodeId, NodeId)> {
        self.prefix_extrema(input, axis, PrefixScanKind::Max)
    }

    /// Inclusive cumulative minima and the first matching I32 index per prefix.
    pub fn cummin(&mut self, input: NodeId, axis: isize) -> Result<(NodeId, NodeId)> {
        self.prefix_extrema(input, axis, PrefixScanKind::Min)
    }

    /// Builds one typed static prefix scan after validating the signed axis
    /// before mutating the graph. Cumulative sums use the established widened
    /// output policy; all other scans retain source storage.
    fn prefix_scan(&mut self, input: NodeId, axis: isize, kind: PrefixScanKind) -> Result<NodeId> {
        let source = self.node(input)?;
        let axis = self.prefix_scan_axis(input, source.shape.rank(), axis)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let dtype = prefix_scan_output_dtype(input_dtype, kind, PrefixScanOutput::Values)
            .expect("value prefix scans have an output dtype");
        let elements = shape.numel()?;
        elements
            .checked_mul(input_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        elements
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        Ok(self.push(
            Op::PrefixScan {
                input,
                axis,
                kind,
                output: PrefixScanOutput::Values,
            },
            shape,
            dtype,
        ))
    }

    fn prefix_scan_axis(&self, input: NodeId, rank: usize, axis: isize) -> Result<usize> {
        if rank == 0 {
            if matches!(axis, -1 | 0) {
                Ok(0)
            } else {
                Err(Error::InvalidReductionAxes {
                    node: input,
                    axes: vec![usize::try_from(axis).unwrap_or(usize::MAX)],
                    rank: 0,
                })
            }
        } else {
            Ok(*normalize_axes(input, rank, Some(vec![axis]))?
                .first()
                .expect("one scan axis"))
        }
    }

    fn prefix_extrema(
        &mut self,
        input: NodeId,
        axis: isize,
        kind: PrefixScanKind,
    ) -> Result<(NodeId, NodeId)> {
        debug_assert!(matches!(kind, PrefixScanKind::Max | PrefixScanKind::Min));
        let source = self.node(input)?;
        let axis = self.prefix_scan_axis(input, source.shape.rank(), axis)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let values_dtype = prefix_scan_output_dtype(input_dtype, kind, PrefixScanOutput::Values)
            .expect("extrema values have an output dtype");
        let indices_dtype = prefix_scan_output_dtype(input_dtype, kind, PrefixScanOutput::Indices)
            .expect("extrema indices have an output dtype");
        let elements = shape.numel()?;
        elements
            .checked_mul(values_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        elements
            .checked_mul(indices_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let values = self.push(
            Op::PrefixScan {
                input,
                axis,
                kind,
                output: PrefixScanOutput::Values,
            },
            shape.clone(),
            values_dtype,
        );
        let indices = self.push(
            Op::PrefixScan {
                input,
                axis,
                kind,
                output: PrefixScanOutput::Indices,
            },
            shape,
            indices_dtype,
        );
        Ok((values, indices))
    }

    pub fn reduce(
        &mut self,
        input: NodeId,
        kind: ReduceKind,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let dtype = match kind {
            ReduceKind::Mean if !source.dtype.is_float() => DType::F32,
            ReduceKind::Sum => sum_dtype(source.dtype),
            _ => source.dtype,
        };
        let accumulator =
            if source.dtype.is_float8() && matches!(kind, ReduceKind::Sum | ReduceKind::Mean) {
                DType::F32
            } else {
                dtype
            };
        self.reduce_with_accumulator_dtype(input, kind, axes, keepdim, accumulator, dtype)
    }

    /// Appends a reduction whose storage dtype is supplied by the caller.
    ///
    /// This is the checked same-storage boundary for typed callers whose
    /// accumulator and result descriptors are identical. Raw reductions with
    /// distinct work/result storage use the private generalized constructor.
    pub(crate) fn reduce_with_output_dtype(
        &mut self,
        input: NodeId,
        kind: ReduceKind,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        dtype: DType,
    ) -> Result<NodeId> {
        self.reduce_with_accumulator_dtype(input, kind, axes, keepdim, dtype, dtype)
    }

    fn reduce_with_accumulator_dtype(
        &mut self,
        input: NodeId,
        kind: ReduceKind,
        axes: Option<Vec<isize>>,
        keepdim: bool,
        accumulator: DType,
        dtype: DType,
    ) -> Result<NodeId> {
        let (input_shape, input_dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let axes = normalize_axes(input, input_shape.rank(), axes)?;
        if matches!(kind, ReduceKind::Any | ReduceKind::All) && input_dtype != DType::Bool {
            return Err(Error::InvalidElementwiseDType {
                op: match kind {
                    ReduceKind::Any => "any",
                    ReduceKind::All => "all",
                    _ => unreachable!(),
                },
                actual: input_dtype,
            });
        }
        let shape = reduction_shape(&input_shape, &axes, keepdim);
        if matches!(kind, ReduceKind::Max | ReduceKind::Min)
            && has_empty_reduction_domain(&input_shape, &shape, &axes)
        {
            return Err(Error::EmptyReduction {
                op: match kind {
                    ReduceKind::Max => "max",
                    ReduceKind::Min => "min",
                    _ => unreachable!(),
                },
                shape: input_shape.clone(),
                axes,
            });
        }
        input_shape
            .numel()?
            .checked_mul(input_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let effective_axes = axes
            .iter()
            .copied()
            .filter(|axis| input_shape.dims()[*axis] != 1)
            .collect::<Vec<_>>();
        if effective_axes.is_empty() {
            let value = if input_dtype == dtype {
                input
            } else {
                self.cast(input, dtype)?
            };
            return if input_shape == shape {
                Ok(value)
            } else {
                self.reshape(value, shape)
            };
        }
        let filtered = effective_axes.len() != axes.len();
        let operation_keepdim = keepdim || filtered;
        let operation_shape = reduction_shape(&input_shape, &effective_axes, operation_keepdim);
        crate::reduction_native::NativeReductionPlan::new(
            input_shape.clone(),
            operation_shape.clone(),
            effective_axes.clone(),
            operation_keepdim,
            kind,
            input_dtype,
            crate::ReductionDType::new(accumulator, dtype),
        )
        .map_err(|reason| Error::Serialization {
            reason: reason.into(),
        })?;
        let reduced = self.push(
            Op::Reduce {
                input,
                kind,
                axes: effective_axes,
                keepdim: operation_keepdim,
                accumulator,
            },
            operation_shape.clone(),
            dtype,
        );
        if operation_shape == shape {
            Ok(reduced)
        } else {
            self.reshape(reduced, shape)
        }
    }
    pub fn argmax(&mut self, input: NodeId, axis: Option<isize>, keepdim: bool) -> Result<NodeId> {
        self.arg_reduce(input, true, axis, keepdim)
    }
    pub fn argmin(&mut self, input: NodeId, axis: Option<isize>, keepdim: bool) -> Result<NodeId> {
        self.arg_reduce(input, false, axis, keepdim)
    }

    /// Stable ordering along one signed axis. The two returned NodeIds are
    /// distinct selectors of one shared producer: values retain the input
    /// dtype and indices are I32 source positions.
    pub fn sort(
        &mut self,
        input: NodeId,
        axis: isize,
        descending: bool,
    ) -> Result<(NodeId, NodeId)> {
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        let values_require_grad = self.grad_enabled && dtype.is_float() && source.requires_grad;
        let elements = shape.numel()?;
        elements
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        elements
            .checked_mul(DType::I32.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        // tinygrad resolves `dim` then indexes `shape[dim]`; rank-zero sort
        // and argsort therefore reject rather than manufacturing a scalar
        // pair. Reject before either selector becomes observable.
        if shape.rank() == 0 {
            return Err(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: 0,
            });
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
        if shape.dims()[axis] > i32::MAX as usize {
            return Err(Error::ShapeOverflow(shape));
        }
        // tinygrad returns the original value identity plus an I32 zero
        // `const_like` index tensor for empty and singleton lanes.  Keep that
        // observable early return, after all source/output byte checks and
        // before any Sort selector is published.
        if shape.dims()[axis] <= 1 {
            let indices = self.lazy_full_with_dtype(shape.clone(), Scalar::I(0), DType::I32)?;
            return Ok((input, indices));
        }
        // The literal source pads to the next power of two before its
        // bitonic stages.  Validate that transient descriptor as part of the
        // all-or-nothing construction plan rather than leaving an overflow
        // for execution after Sort selectors have become visible.
        let padded_extent = shape.dims()[axis]
            .checked_next_power_of_two()
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut padded_dims = shape.dims().to_vec();
        padded_dims[axis] = padded_extent;
        let padded_shape = Shape::new(padded_dims);
        padded_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(padded_shape.clone()))?;
        let pair = self.nodes.len() as u64;
        let values = self.push_with_grad(
            Op::Sort {
                input,
                axis,
                descending,
                pair,
                output: SortOutput::Values,
            },
            shape.clone(),
            dtype,
            values_require_grad,
        );
        let indices = self.push_with_grad(
            Op::Sort {
                input,
                axis,
                descending,
                pair,
                output: SortOutput::Indices,
            },
            shape,
            DType::I32,
            false,
        );
        Ok((values, indices))
    }

    /// Checked-in tinygrad's `Tensor.sort()` defaults: final dimension,
    /// ascending direction.
    pub fn sort_default(&mut self, input: NodeId) -> Result<(NodeId, NodeId)> {
        self.sort(input, -1, false)
    }

    /// Stable I32 source positions from [`Self::sort`].
    pub fn argsort(&mut self, input: NodeId, axis: isize, descending: bool) -> Result<NodeId> {
        Ok(self.sort(input, axis, descending)?.1)
    }

    /// Checked-in tinygrad's `Tensor.argsort()` defaults: final dimension,
    /// ascending direction.
    pub fn argsort_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.argsort(input, -1, false)
    }

    /// Returns canonical metadata for one exact stable Sort selector pair.
    /// This crate-visible inspection seam is deliberately read-only: the
    /// isolated CPU plan route may describe a pair, but cannot schedule or
    /// execute it through the generic ScheduleItem ABI.
    pub(crate) fn stable_sort_pair_for_cpu_plan(
        &self,
        source: NodeId,
        values: NodeId,
        indices: NodeId,
    ) -> Option<(Shape, DType, usize, bool, u64)> {
        let values_node = self.node(values).ok()?;
        let (axis, descending, pair) = match &values_node.op {
            Op::Sort {
                input,
                axis,
                descending,
                pair,
                output: SortOutput::Values,
            } if *input == source => (*axis, *descending, *pair),
            _ => return None,
        };
        let source_node = self.node(source).ok()?;
        let shape = source_node.shape.clone();
        let dtype = source_node.dtype;
        shape.numel().ok()?;
        if shape.rank() == 0 || axis >= shape.rank() || shape.dims()[axis] > i32::MAX as usize {
            return None;
        }
        if values_node.shape != shape || values_node.dtype != dtype {
            return None;
        }
        let indices_node = self.node(indices).ok()?;
        if !matches!(
            &indices_node.op,
            Op::Sort {
                input,
                axis: candidate_axis,
                descending: candidate_descending,
                pair: candidate_pair,
                output: SortOutput::Indices,
            } if *input == source
                && *candidate_axis == axis
                && *candidate_descending == descending
                && *candidate_pair == pair
        ) || indices_node.shape != shape
            || indices_node.dtype != DType::I32
        {
            return None;
        }
        let (value_selectors, index_selectors) = self.nodes.iter().fold(
            (0usize, 0usize),
            |(value_count, index_count), node| match &node.op {
                Op::Sort {
                    input,
                    axis: candidate_axis,
                    descending: candidate_descending,
                    pair: candidate_pair,
                    output,
                } if *input == source
                    && *candidate_axis == axis
                    && *candidate_descending == descending
                    && *candidate_pair == pair
                    && node.shape == shape =>
                {
                    match output {
                        SortOutput::Values if node.dtype == dtype => (value_count + 1, index_count),
                        SortOutput::Indices if node.dtype == DType::I32 => {
                            (value_count, index_count + 1)
                        }
                        _ => (value_count, index_count),
                    }
                }
                _ => (value_count, index_count),
            },
        );
        (value_selectors == 1 && index_selectors == 1)
            .then_some((shape, dtype, axis, descending, pair))
    }

    /// Returns the largest or smallest `k` values and their stable I32 source
    /// positions along one signed axis. This is deliberately the checked
    /// `sort`-then-`shrink` form used by tinygrad; unsorted TopK has no local
    /// implementation contract.
    pub fn topk(
        &mut self,
        input: NodeId,
        k: usize,
        axis: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(NodeId, NodeId)> {
        let shape = self.node(input)?.shape.clone();
        if !sorted {
            return Err(Error::UnsupportedTopKUnsorted);
        }
        shape.numel()?;
        if shape.rank() == 0 {
            return Err(Error::InvalidAxis {
                node: input,
                axis: 0,
                rank: 0,
            });
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
        let dim = shape.dims()[axis];
        if k > dim {
            return Err(Error::InvalidBounds {
                axis,
                start: 0,
                end: k,
                dim,
            });
        }
        let mut bounds = shape
            .dims()
            .iter()
            .copied()
            .map(|extent| (0, extent))
            .collect::<Vec<_>>();
        bounds[axis].1 = k;
        // All three operations have been fully preflighted above, so neither
        // selector nor the first view can be published on a later rejection.
        let (values, indices) = self.sort(input, axis as isize, largest)?;
        let values = self.shrink(values, bounds.clone())?;
        let indices = self.shrink(indices, bounds)?;
        Ok((values, indices))
    }

    /// Checked-in tinygrad's `Tensor.topk(k)` defaults: final dimension,
    /// largest values first, and the required sorted result.
    pub fn topk_default(&mut self, input: NodeId, k: usize) -> Result<(NodeId, NodeId)> {
        self.topk(input, k, -1, true, true)
    }

    fn arg_reduce(
        &mut self,
        input: NodeId,
        max: bool,
        axis: Option<isize>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let axis = axis
            .map(|a| normalize_axes(input, source.shape.rank(), Some(vec![a])))
            .transpose()?
            .map(|v| v[0]);
        let axes = axis.map_or_else(|| (0..source.shape.rank()).collect(), |a| vec![a]);
        let shape = reduction_shape(&source.shape, &axes, keepdim);
        if has_empty_reduction_domain(&source.shape, &shape, &axes) {
            return Err(Error::EmptyReduction {
                op: if max { "argmax" } else { "argmin" },
                shape: source.shape.clone(),
                axes,
            });
        }
        Ok(self.push(
            Op::ArgReduce {
                input,
                max,
                axis,
                keepdim,
            },
            shape,
            DType::I32,
        ))
    }
    pub(crate) fn reduce_grad(
        &mut self,
        input: NodeId,
        upstream: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let dtype = self.node(upstream)?.dtype;
        Ok(self.push(
            Op::ReduceGrad {
                input,
                upstream,
                kind,
                axes,
                keepdim,
            },
            shape,
            dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_grad_vjp(
        &mut self,
        cotangent: NodeId,
        input: NodeId,
        upstream: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = match wrt {
            0 => self.node(input)?.shape.clone(),
            1 => self.node(upstream)?.shape.clone(),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(self.push(
            Op::ReduceGradVjp {
                cotangent,
                input,
                upstream,
                kind,
                axes,
                keepdim,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    pub fn sum_to(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = shape.into();
        if shape.broadcast_with(&source.shape).as_ref() != Ok(&source.shape) {
            return Err(Error::InvalidSumTo {
                from: source.shape.clone(),
                to: shape,
            });
        }
        Ok(self.push(
            Op::SumTo {
                input,
                shape: shape.clone(),
            },
            shape,
            source.dtype,
        ))
    }

    pub fn reshape(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = shape.into();
        let source_numel = source.shape.numel()?;
        source_numel
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        let output_numel = shape.numel()?;
        output_numel
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        if source_numel != output_numel {
            return Err(Error::InvalidReshape {
                from: source.shape.clone(),
                to: shape,
            });
        }
        if shape == source.shape {
            Ok(input)
        } else {
            Ok(self.push(
                Op::Reshape {
                    input,
                    shape: shape.clone(),
                },
                shape,
                source.dtype,
            ))
        }
    }

    /// Reshapes using tinygrad's public concrete, copied-extent, and single
    /// inferred-extent forms. Existing concrete `reshape` callers retain their
    /// direct `Shape` API.
    pub fn reshape_with_extents(
        &mut self,
        input: NodeId,
        extents: impl Into<Vec<crate::ReshapeExtent>>,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let source_shape = source.shape.clone();
        let source_numel = source_shape.numel()?;
        source_numel
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source_shape.clone()))?;
        let extents = extents.into();
        let inferred = extents
            .iter()
            .enumerate()
            .filter_map(|(index, extent)| {
                matches!(extent, crate::ReshapeExtent::Infer).then_some(index)
            })
            .collect::<Vec<_>>();
        if inferred.len() > 1 {
            return Err(Error::InvalidReshape {
                from: source_shape,
                to: Shape::new(Vec::new()),
            });
        }
        let mut output = Vec::with_capacity(extents.len());
        let mut known_product = 1usize;
        for (index, extent) in extents.iter().enumerate() {
            let extent = match extent {
                crate::ReshapeExtent::Exact(extent) => Some(*extent),
                crate::ReshapeExtent::Copy => Some(*source_shape.dims().get(index).ok_or_else(
                    || Error::InvalidReshape {
                        from: source_shape.clone(),
                        to: Shape::new(Vec::new()),
                    },
                )?),
                crate::ReshapeExtent::Infer => None,
            };
            if let Some(extent) = extent {
                known_product = known_product
                    .checked_mul(extent)
                    .ok_or_else(|| Error::ShapeOverflow(source_shape.clone()))?;
                output.push(extent);
            } else {
                output.push(0);
            }
        }
        if let Some(index) = inferred.first().copied() {
            // tinygrad evaluates `-prod(old) // prod(new_shape)`; a zero
            // denominator therefore fails even when the input itself is empty.
            if known_product == 0 {
                return Err(Error::InvalidReshape {
                    from: source_shape,
                    to: Shape::new(output),
                });
            }
            output[index] = source_numel / known_product;
        }
        let output_shape = Shape::new(output);
        let output_numel = output_shape.numel()?;
        output_numel
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        if source_numel != output_numel {
            return Err(Error::InvalidReshape {
                from: source_shape,
                to: output_shape,
            });
        }
        self.reshape(input, output_shape)
    }

    /// Checked-in tinygrad `Tensor.view(shape)` is an exact alias for its
    /// public reshape surface, including concrete, copied (`None`), and one
    /// inferred (`-1`) extent forms.
    pub fn view(
        &mut self,
        input: NodeId,
        extents: impl Into<Vec<crate::ReshapeExtent>>,
    ) -> Result<NodeId> {
        self.reshape_with_extents(input, extents)
    }

    pub fn permute(&mut self, input: NodeId, axes: impl Into<Vec<usize>>) -> Result<NodeId> {
        let source = self.node(input)?;
        let axes = axes.into();
        let mut sorted = axes.clone();
        sorted.sort_unstable();
        if sorted != (0..source.shape.rank()).collect::<Vec<_>>() {
            return Err(Error::InvalidPermutation {
                shape: source.shape.clone(),
                axes,
            });
        }
        let shape = Shape::new(
            axes.iter()
                .map(|axis| source.shape.dims()[*axis])
                .collect::<Vec<_>>(),
        );
        source
            .shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        if axes.iter().copied().eq(0..source.shape.rank()) {
            Ok(input)
        } else {
            Ok(self.push(Op::Permute { input, axes }, shape, source.dtype))
        }
    }

    /// Permutes an explicit sequence of signed axes, matching tinygrad's
    /// public `Tensor.permute(order)` axis normalization. The legacy unsigned
    /// `permute` remains available for existing callers.
    pub fn permute_signed(&mut self, input: NodeId, axes: impl Into<Vec<isize>>) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let rank = isize::try_from(shape.rank()).map_err(|_| Error::InvalidPermutation {
            shape: shape.clone(),
            axes: Vec::new(),
        })?;
        let raw_axes = axes.into();
        if raw_axes.len() != shape.rank() {
            return Err(Error::InvalidPermutation {
                shape,
                axes: Vec::new(),
            });
        }
        let mut normalized = Vec::with_capacity(raw_axes.len());
        for axis in raw_axes {
            let axis = if axis < 0 {
                axis.checked_add(rank)
                    .ok_or_else(|| Error::InvalidPermutation {
                        shape: shape.clone(),
                        axes: Vec::new(),
                    })?
            } else {
                axis
            };
            if axis < 0 || axis >= rank {
                return Err(Error::InvalidPermutation {
                    shape,
                    axes: Vec::new(),
                });
            }
            normalized.push(axis as usize);
        }
        self.permute(input, normalized)
    }

    /// Swaps two signed axes, matching tinygrad's public `transpose(dim0,
    /// dim1)` composition over `permute`.
    ///
    /// Each axis is normalized independently so equal axes remain a source
    /// no-op rather than being rejected as a duplicate permutation. Both
    /// normalizations complete before a permutation node can be appended.
    pub fn transpose(&mut self, input: NodeId, dim0: isize, dim1: isize) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        let dim0 = normalize_axes(input, rank, Some(vec![dim0]))?[0];
        let dim1 = normalize_axes(input, rank, Some(vec![dim1]))?[0];
        if dim0 == dim1 {
            return Ok(input);
        }
        let mut order = (0..rank).collect::<Vec<_>>();
        order.swap(dim0, dim1);
        self.permute(input, order)
    }

    /// Applies tinygrad's public `Tensor.transpose()` defaults, swapping axes
    /// one and zero. Explicit signed-axis callers should use `transpose`.
    pub fn transpose_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.transpose(input, 1, 0)
    }

    /// Checked-in tinygrad `Tensor.T`: the rank-two-only public transpose
    /// surface, lowered as the literal `transpose(1, 0)` permutation.
    ///
    /// This is deliberately distinct from [`Self::transpose_default`], whose
    /// legacy/general transpose contract remains available for higher ranks.
    /// Input and output typed byte extents are completely checked before the
    /// Permute node is published.
    pub fn t_tinygrad(&mut self, input: NodeId) -> Result<NodeId> {
        let source = self.node(input)?;
        let input_shape = source.shape.clone();
        let dtype = source.dtype;
        if input_shape.rank() != 2 {
            return Err(Error::InvalidMovementRank {
                op: "Tensor.T",
                expected: 2,
                actual: input_shape.rank(),
            });
        }
        input_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        let output_shape = Shape::new([input_shape.dims()[1], input_shape.dims()[0]]);
        output_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        Ok(self.push(
            Op::Permute {
                input,
                axes: vec![1, 0],
            },
            output_shape,
            dtype,
        ))
    }

    /// Replaces one signed axis with concrete extents and, at most, one
    /// source-compatible inferred extent.
    ///
    /// This is tinygrad's `unflatten(dim, sizes)` expressed without a general
    /// negative-shape API. Every axis, inference, product, and output extent
    /// is checked before the final existing `reshape` node is appended.
    pub fn unflatten(
        &mut self,
        input: NodeId,
        dim: isize,
        sizes: impl Into<Vec<crate::UnflattenExtent>>,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        shape.numel()?;
        let axis = normalize_axes(input, shape.rank(), Some(vec![dim]))?[0];
        let sizes = sizes.into();
        let inferred = sizes
            .iter()
            .enumerate()
            .filter_map(|(index, size)| {
                matches!(size, crate::UnflattenExtent::Infer).then_some(index)
            })
            .collect::<Vec<_>>();
        if inferred.len() > 1 {
            return Err(Error::InvalidRandom {
                reason: "unflatten permits at most one inferred extent",
            });
        }
        let known_product = sizes.iter().try_fold(1usize, |product, size| match size {
            crate::UnflattenExtent::Exact(size) => product
                .checked_mul(*size)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone())),
            crate::UnflattenExtent::Infer => Ok(product),
        })?;
        let source_extent = shape.dims()[axis];
        let inferred_extent = if inferred.is_empty() {
            None
        } else {
            if known_product == 0 {
                return Err(Error::InvalidRandom {
                    reason: "unflatten cannot infer through zero extent product",
                });
            }
            // `unflatten` delegates to `reshape` with the surrounding axes
            // intact.  tinygrad's `-prod(old) // prod(new_shape)` therefore
            // rejects an inferred extent whenever any *other* output axis is
            // zero, rather than allowing the whole-tensor zero product to
            // mask an invalid split of the selected axis.
            let inference_denominator = shape.dims()[..axis]
                .iter()
                .chain(sizes.iter().filter_map(|size| match size {
                    crate::UnflattenExtent::Exact(size) => Some(size),
                    crate::UnflattenExtent::Infer => None,
                }))
                .chain(shape.dims()[axis + 1..].iter())
                .try_fold(1usize, |product, extent| {
                    product
                        .checked_mul(*extent)
                        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                })?;
            if inference_denominator == 0 || source_extent % known_product != 0 {
                return Err(Error::InvalidReshape {
                    from: shape.clone(),
                    to: Shape::new(Vec::new()),
                });
            }
            Some(source_extent / known_product)
        };
        let output_capacity = shape
            .rank()
            .checked_sub(1)
            .and_then(|rank| rank.checked_add(sizes.len()))
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut output = Vec::with_capacity(output_capacity);
        output.extend_from_slice(&shape.dims()[..axis]);
        output.extend(sizes.into_iter().map(|size| match size {
            crate::UnflattenExtent::Exact(size) => size,
            crate::UnflattenExtent::Infer => inferred_extent.expect("inferred extent prevalidated"),
        }));
        output.extend_from_slice(&shape.dims()[axis + 1..]);
        let output = Shape::new(output);
        if output.numel()? != shape.numel()? {
            return Err(Error::InvalidReshape {
                from: shape,
                to: output,
            });
        }
        self.reshape(input, output)
    }

    pub fn expand(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let requested = shape.into();
        let rank = source.shape.rank().max(requested.rank());
        let padding = rank
            .checked_sub(requested.rank())
            .ok_or_else(|| Error::ShapeOverflow(requested.clone()))?;
        let mut dims = Vec::with_capacity(rank);
        dims.resize(padding, 1);
        dims.extend_from_slice(requested.dims());
        let shape = Shape::new(dims);
        source
            .shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        if source.shape.broadcast_with(&shape).as_ref() != Ok(&shape) {
            return Err(Error::InvalidExpand {
                from: source.shape.clone(),
                to: shape,
            });
        }
        if shape == source.shape {
            Ok(input)
        } else {
            Ok(self.push(
                Op::Expand {
                    input,
                    shape: shape.clone(),
                },
                shape,
                source.dtype,
            ))
        }
    }

    /// Expands using tinygrad's public concrete and copied-extent forms.
    /// Existing concrete `expand` callers retain their direct `Shape` API.
    pub fn expand_with_extents(
        &mut self,
        input: NodeId,
        extents: impl Into<Vec<crate::ExpandExtent>>,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let source_shape = source.shape.clone();
        source_shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source_shape.clone()))?;
        let extents = extents.into();
        let rank = source_shape.rank().max(extents.len());
        let source_padding = rank
            .checked_sub(source_shape.rank())
            .ok_or_else(|| Error::ShapeOverflow(source_shape.clone()))?;
        let extent_padding = rank
            .checked_sub(extents.len())
            .ok_or_else(|| Error::ShapeOverflow(source_shape.clone()))?;
        let mut output = Vec::with_capacity(rank);
        for axis in 0..rank {
            let source_extent = if axis < source_padding {
                1
            } else {
                source_shape.dims()[axis - source_padding]
            };
            let extent = if axis < extent_padding {
                crate::ExpandExtent::Exact(1)
            } else {
                extents[axis - extent_padding]
            };
            output.push(match extent {
                crate::ExpandExtent::Exact(extent) => extent,
                crate::ExpandExtent::Copy => source_extent,
            });
        }
        let output_shape = Shape::new(output);
        output_shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        self.expand(input, output_shape)
    }

    /// Takes checked, half-open bounds for every input axis.
    pub fn shrink(
        &mut self,
        input: NodeId,
        bounds: impl Into<Vec<(usize, usize)>>,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let bounds = bounds.into();
        if bounds.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "shrink",
                expected: source.shape.rank(),
                actual: bounds.len(),
            });
        }
        source
            .shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        let mut dims = Vec::with_capacity(bounds.len());
        for (axis, ((start, end), dim)) in bounds.iter().zip(source.shape.dims()).enumerate() {
            if start > end || *end > *dim {
                return Err(Error::InvalidBounds {
                    axis,
                    start: *start,
                    end: *end,
                    dim: *dim,
                });
            }
            dims.push(end - start);
        }
        let output_shape = Shape::new(dims);
        output_shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        if bounds
            .iter()
            .zip(source.shape.dims())
            .all(|((start, end), dim)| *start == 0 && *end == *dim)
        {
            Ok(input)
        } else {
            Ok(self.push(Op::Shrink { input, bounds }, output_shape, source.dtype))
        }
    }

    /// Shrinks using tinygrad's public per-axis `None` or concrete half-open
    /// range form. Existing concrete `shrink` callers retain their pair API.
    pub fn shrink_with_ranges(
        &mut self,
        input: NodeId,
        ranges: impl Into<Vec<crate::ShrinkRange>>,
    ) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let ranges = ranges.into();
        if ranges.len() != shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "shrink_with_ranges",
                expected: shape.rank(),
                actual: ranges.len(),
            });
        }
        let bounds = ranges
            .into_iter()
            .zip(shape.dims())
            .map(|(range, &dimension)| match range {
                crate::ShrinkRange::Full => (0, dimension),
                crate::ShrinkRange::Bounds { start, end } => (start, end),
            })
            .collect::<Vec<_>>();
        self.shrink(input, bounds)
    }

    /// Checked-in tinygrad `Tensor.shrink_to(shape)` source surface. The
    /// strict target list is Rust's concrete representation of source's tuple
    /// or separate extent arguments; `None` retains the complete axis.
    pub fn shrink_to(
        &mut self,
        input: NodeId,
        target: impl Into<Vec<Option<usize>>>,
    ) -> Result<NodeId> {
        let plan = shrink_to_plan(self, input, target.into())?;
        let output = self.shrink(input, plan.bounds)?;
        debug_assert_eq!(
            self.shape(output).expect("shrink_to preflighted"),
            &plan.output_shape
        );
        Ok(output)
    }

    /// Pads every axis with `(before, after)`. `fill` is deterministically
    /// converted to the input dtype; padding never changes tensor dtype.
    pub fn pad(
        &mut self,
        input: NodeId,
        padding: impl Into<Vec<(usize, usize)>>,
        fill: Scalar,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let padding = padding.into();
        if padding.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "pad",
                expected: source.shape.rank(),
                actual: padding.len(),
            });
        }
        let dims = source
            .shape
            .dims()
            .iter()
            .zip(&padding)
            .map(|(dim, (before, after))| {
                dim.checked_add(*before)
                    .and_then(|x| x.checked_add(*after))
                    .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let output_shape = Shape::new(dims);
        // Movement Pad allocates a concrete output buffer; establish both
        // descriptors before publishing the graph node.
        source.shape.numel()?;
        output_shape.numel()?;
        source
            .shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        output_shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        Ok(self.push(
            Op::Pad {
                input,
                padding,
                fill,
            },
            output_shape,
            source.dtype,
        ))
    }

    /// Constant grouped padding with tinygrad-style signed crop support.
    ///
    /// Each pair is `(before, after)` in source axis order. Negative values
    /// crop first, then the remaining nonnegative padding is applied with the
    /// provided fill value. This intentionally covers tinygrad's default
    /// constant mode only; reflect, replicate, and circular require their
    /// own source-index constructions.
    pub fn pad_signed(
        &mut self,
        input: NodeId,
        padding: impl Into<Vec<(i64, i64)>>,
        fill: Scalar,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        shape.numel()?;
        let padding = padding.into();
        if padding.len() != shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "pad_signed",
                expected: shape.rank(),
                actual: padding.len(),
            });
        }
        let mut bounds = Vec::with_capacity(shape.rank());
        let mut positive = Vec::with_capacity(shape.rank());
        let mut final_dims = Vec::with_capacity(shape.rank());
        let mut cropped = false;
        let mut padded = false;
        for (axis, (&dimension, &(before, after))) in shape.dims().iter().zip(&padding).enumerate()
        {
            let dimension_i128 = dimension as i128;
            let start = (-(before as i128)).max(0);
            let end = (dimension_i128 + after as i128).min(dimension_i128);
            if end < 0 {
                return Err(Error::InvalidRandom {
                    reason: "signed padding crops beyond axis",
                });
            }
            let start = usize::try_from(start).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            let end = usize::try_from(end).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            if start > end || end > dimension {
                return Err(Error::InvalidBounds {
                    axis,
                    start,
                    end,
                    dim: dimension,
                });
            }
            let before =
                usize::try_from(before.max(0)).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            let after =
                usize::try_from(after.max(0)).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            let retained = end - start;
            retained
                .checked_add(before)
                .and_then(|value| value.checked_add(after))
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            cropped |= start != 0 || end != dimension;
            padded |= before != 0 || after != 0;
            bounds.push((start, end));
            positive.push((before, after));
            final_dims.push(
                retained
                    .checked_add(before)
                    .and_then(|x| x.checked_add(after))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?,
            );
        }
        let final_shape = Shape::new(final_dims);
        shape.numel()?;
        final_shape.numel()?;
        shape
            .numel()?
            .checked_mul(self.node(input)?.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        final_shape
            .numel()?
            .checked_mul(self.node(input)?.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(final_shape.clone()))?;
        let value = if cropped {
            self.shrink(input, bounds)?
        } else {
            input
        };
        if padded {
            self.pad(value, positive, fill)
        } else {
            Ok(value)
        }
    }

    /// Checked-in tinygrad `Tensor.pad_to(shape, value=0)` with the default
    /// zero fill. `None` retains an input extent; concrete targets must have
    /// exactly one entry per input axis.
    pub fn pad_to(
        &mut self,
        input: NodeId,
        target: impl Into<Vec<Option<usize>>>,
    ) -> Result<NodeId> {
        self.pad_to_with_value(input, target, Scalar::I(0))
    }

    /// Checked-in tinygrad `Tensor.pad_to(shape, value=...)` with its literal
    /// zero-Pad then Bool-mask `where(base, value)` fill path. The target list
    /// is Rust's concrete representation of source's tuple or separate shape
    /// arguments, including `None` entries.
    pub fn pad_to_with_value(
        &mut self,
        input: NodeId,
        target: impl Into<Vec<Option<usize>>>,
        fill: Scalar,
    ) -> Result<NodeId> {
        let plan = pad_to_plan(self, input, target.into(), fill)?;
        // The source composes movement Pad and (for a nonzero fill) scalar
        // Where. Rehearse that complete path so a late scalar promotion or
        // mask descriptor never leaves a partial caller graph behind.
        let mut rehearsal = self.clone();
        let rehearsed = lower_pad_to(&mut rehearsal, input, &plan)?;
        let rehearsed_shape = rehearsal.shape(rehearsed)?.clone();
        let rehearsed_dtype = rehearsal.dtype(rehearsed)?;
        rehearsed_shape
            .numel()?
            .checked_mul(rehearsed_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(rehearsed_shape.clone()))?;
        debug_assert_eq!(rehearsed_shape, plan.output_shape);
        let output = lower_pad_to(self, input, &plan)?;
        debug_assert_eq!(
            self.shape(output).expect("pad_to preflighted"),
            &plan.output_shape
        );
        Ok(output)
    }

    /// Checked-in tinygrad grouped signed `Tensor.pad` composition.
    ///
    /// This deliberately leaves raw [`Self::pad`] and constant
    /// [`Self::pad_signed`] intact. The public source modes are composed from
    /// existing movement and elementwise operations rather than new backend
    /// padding variants.
    pub fn pad_with_mode(
        &mut self,
        input: NodeId,
        padding: impl Into<Vec<(i64, i64)>>,
        mode: PadMode,
        fill: Scalar,
    ) -> Result<NodeId> {
        let plan = pad_mode_plan(self, input, padding.into(), mode, fill)?;
        let output = lower_pad_mode(self, input, &plan.padding, plan.mode, plan.fill)?;
        debug_assert_eq!(
            self.shape(output).expect("pad mode preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("pad mode preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// Applies Python-style signed slices, including negative steps and flips.
    pub fn stride(&mut self, input: NodeId, slices: impl Into<Vec<Slice>>) -> Result<NodeId> {
        let source = self.node(input)?;
        let slices = slices.into();
        if slices.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "stride",
                expected: source.shape.rank(),
                actual: slices.len(),
            });
        }
        let dims = slices
            .iter()
            .zip(source.shape.dims())
            .enumerate()
            .map(|(axis, (slice, dim))| {
                normalized_slice(*dim, *slice, axis).map(|(_, _, _, length)| length)
            })
            .collect::<Result<Vec<_>>>()?;
        let output_shape = Shape::new(dims);
        source
            .shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        output_shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        Ok(self.push(Op::Stride { input, slices }, output_shape, source.dtype))
    }

    /// Alias for [`Graph::stride`], emphasizing ordinary slicing semantics.
    pub fn slice(&mut self, input: NodeId, slices: impl Into<Vec<Slice>>) -> Result<NodeId> {
        self.stride(input, slices)
    }

    /// Reverses each distinct signed axis through the existing checked stride
    /// view. An empty axis list is the same no-op as tinygrad's `flip(())`.
    pub fn flip(&mut self, input: NodeId, axes: impl AsRef<[isize]>) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let dtype = self.dtype(input)?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let rank = isize::try_from(shape.rank()).map_err(|_| Error::InvalidAxis {
            node: input,
            axis: usize::MAX,
            rank: usize::MAX,
        })?;
        let mut normalized = axes
            .as_ref()
            .iter()
            .copied()
            .map(|axis| {
                let axis = if axis < 0 {
                    axis.checked_add(rank).ok_or(Error::InvalidAxis {
                        node: input,
                        axis: usize::MAX,
                        rank: rank as usize,
                    })?
                } else {
                    axis
                };
                if axis < 0 || axis >= rank {
                    return Err(Error::InvalidAxis {
                        node: input,
                        axis: usize::try_from(axis).unwrap_or(usize::MAX),
                        rank: rank as usize,
                    });
                }
                Ok(axis as usize)
            })
            .collect::<Result<Vec<_>>>()?;
        normalized.sort_unstable();
        if normalized.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidFlip {
                reason: "flip axes must be unique",
            });
        }
        if normalized.is_empty() {
            return Ok(input);
        }
        let slices = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(axis, _)| Slice {
                start: None,
                stop: None,
                step: if normalized.binary_search(&axis).is_ok() {
                    -1
                } else {
                    1
                },
            })
            .collect::<Vec<_>>();
        self.stride(input, slices)
    }

    /// Splits along tinygrad's default axis zero.
    ///
    /// This is equivalent to `chunk(input, chunks, 0)`.
    pub fn chunk_default(&mut self, input: NodeId, chunks: usize) -> Result<Vec<NodeId>> {
        self.chunk(input, chunks, 0)
    }

    /// Splits a concrete axis into at most `chunks` ordered, contiguous
    /// shrink views. Uneven nonempty axes use the tinygrad tail rule; a
    /// zero-sized axis returns exactly `chunks` empty views.
    pub fn chunk(&mut self, input: NodeId, chunks: usize, axis: isize) -> Result<Vec<NodeId>> {
        let plan = chunk_plan(self, input, chunks, axis)?;
        plan.bounds
            .into_iter()
            .map(|bounds| self.shrink(input, bounds))
            .collect()
    }

    /// Splits along tinygrad's default axis zero.
    ///
    /// This accepts both source section forms and is equivalent to
    /// `split(input, sections, 0)`.
    pub fn split_default(
        &mut self,
        input: NodeId,
        sections: impl Into<SplitSections>,
    ) -> Result<Vec<NodeId>> {
        self.split(input, sections, 0)
    }

    /// Splits a concrete axis into ordered contiguous shrink views, using the
    /// exact uniform-tail or explicit-coverage form selected by `sections`.
    pub fn split(
        &mut self,
        input: NodeId,
        sections: impl Into<SplitSections>,
        axis: isize,
    ) -> Result<Vec<NodeId>> {
        let plan = split_plan(self, input, sections.into(), axis)?;
        plan.bounds
            .into_iter()
            .map(|bounds| self.shrink(input, bounds))
            .collect()
    }

    /// tinygrad's public `Tensor.cat(*args, dim=0)` composition.
    ///
    /// Equal concatenating extents take the literal all-input
    /// `stack(...).flatten(dim, dim + 1)` route. Unequal extents instead pad
    /// every input with a source-typed zero and fold those full output-shaped
    /// tensors in source order with ADD. `concat` below remains the raw IR
    /// helper for internal callers that deliberately need direct `Op::Concat`.
    pub fn cat(
        &mut self,
        input: NodeId,
        args: impl Into<Vec<NodeId>>,
        dim: isize,
    ) -> Result<NodeId> {
        let plan = cat_plan(self, input, args.into(), dim)?;
        if plan.identity {
            return Ok(input);
        }
        let stack_axis = isize::try_from(plan.axis)
            .map_err(|_| Error::ShapeOverflow(plan.output_shape.clone()))?;
        let flatten_end = stack_axis
            .checked_add(1)
            .ok_or_else(|| Error::ShapeOverflow(plan.output_shape.clone()))?;
        let output = match plan.lowering {
            CatLowering::Stack => {
                let stacked = self.stack(plan.inputs, stack_axis)?;
                self.flatten(stacked, stack_axis, flatten_end)?
            }
            CatLowering::PadSum { paddings } => {
                let mut padded = plan
                    .inputs
                    .into_iter()
                    .zip(paddings)
                    .map(|(input, padding)| {
                        let dtype = self.dtype(input).expect("Cat input preflighted");
                        self.pad(input, padding, cat_zero(dtype))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter();
                let mut output = padded.next().expect("Cat plan has its receiver");
                for input in padded {
                    output = self.add(output, input)?;
                }
                output
            }
        };
        debug_assert_eq!(self.shape(output)?, &plan.output_shape);
        debug_assert_eq!(self.dtype(output)?, plan.output_dtype);
        Ok(output)
    }

    /// Convenience form of [`Graph::cat`] using tinygrad's default `dim=0`.
    pub fn cat_default(&mut self, input: NodeId, args: impl Into<Vec<NodeId>>) -> Result<NodeId> {
        self.cat(input, args, 0)
    }

    /// Concatenates at least two equally ranked tensors along `axis`.
    pub fn concat(&mut self, inputs: impl Into<Vec<NodeId>>, axis: usize) -> Result<NodeId> {
        let inputs = inputs.into();
        if inputs.len() < 2 {
            return Err(Error::InvalidConcat {
                axis,
                shapes: inputs
                    .iter()
                    .filter_map(|id| self.node(*id).ok().map(|n| n.shape.clone()))
                    .collect(),
            });
        }
        let first = self.node(inputs[0])?;
        if axis >= first.shape.rank() {
            return Err(Error::InvalidAxis {
                node: inputs[0],
                axis,
                rank: first.shape.rank(),
            });
        }
        let shape = first.shape.clone();
        let mut dtype = first.dtype;
        let mut total = 0usize;
        let shapes = inputs
            .iter()
            .map(|id| self.node(*id).map(|n| n.shape.clone()))
            .collect::<Result<Vec<_>>>()?;
        for (id, node_shape) in inputs.iter().zip(&shapes) {
            let node = self.node(*id)?;
            if node_shape.rank() != shape.rank()
                || node_shape
                    .dims()
                    .iter()
                    .enumerate()
                    .any(|(i, dim)| i != axis && *dim != shape.dims()[i])
            {
                return Err(Error::InvalidConcat { axis, shapes });
            }
            total = total
                .checked_add(node_shape.dims()[axis])
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            dtype = dtype.promote(node.dtype);
        }
        let mut dims = shape.dims().to_vec();
        dims[axis] = total;
        let output = Shape::new(dims);
        output.numel()?;
        Ok(self.push(Op::Concat { inputs, axis }, output, dtype))
    }

    pub(crate) fn scatter_positions(
        &mut self,
        input: NodeId,
        shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        if starts.len() != shape.rank()
            || steps.len() != shape.rank()
            || source.shape.rank() != shape.rank()
        {
            return Err(Error::InvalidMovementRank {
                op: "scatter",
                expected: shape.rank(),
                actual: starts.len().min(steps.len()).min(source.shape.rank()),
            });
        }
        source
            .shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        crate::movement_plan::StaticPositionMap::new(
            source.shape.clone(),
            shape.clone(),
            &starts,
            &steps,
        )
        .map_err(|error| match error {
            crate::movement_plan::MovementPlanError::Overflow => {
                Error::ShapeOverflow(shape.clone())
            }
            _ => Error::InvalidIndex,
        })?;
        Ok(self.push(
            Op::ScatterPositions {
                input,
                shape: shape.clone(),
                starts,
                steps,
            },
            shape,
            source.dtype,
        ))
    }

    pub(crate) fn scatter_positions_vjp(
        &mut self,
        cotangent: NodeId,
        input_shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    ) -> Result<NodeId> {
        let source = self.node(cotangent)?;
        if starts.len() != input_shape.rank()
            || steps.len() != input_shape.rank()
            || source.shape.rank() != input_shape.rank()
        {
            return Err(Error::InvalidMovementRank {
                op: "scatter vjp",
                expected: input_shape.rank(),
                actual: starts.len().min(steps.len()).min(source.shape.rank()),
            });
        }
        input_shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        source
            .shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
        crate::movement_plan::StaticPositionMap::new(
            input_shape.clone(),
            source.shape.clone(),
            &starts,
            &steps,
        )
        .map_err(|error| match error {
            crate::movement_plan::MovementPlanError::Overflow => {
                Error::ShapeOverflow(input_shape.clone())
            }
            _ => Error::InvalidIndex,
        })?;
        Ok(self.push(
            Op::ScatterPositionsVjp {
                cotangent,
                input_shape: input_shape.clone(),
                starts,
                steps,
            },
            input_shape,
            source.dtype,
        ))
    }

    /// Takes values from `input` at integer coordinates supplied by `index`.
    /// Index rank matches input rank and every non-axis index dimension must
    /// not exceed the corresponding input dimension. Negative indices are not
    /// accepted, matching tinygrad's gather contract.
    pub fn gather(&mut self, input: NodeId, index: NodeId, axis: usize) -> Result<NodeId> {
        let source = self.node(input)?;
        let index_node = self.node(index)?;
        validate_indexed("gather", source, index_node, axis)?;
        Ok(self.push(
            Op::Gather { input, index, axis },
            index_node.shape.clone(),
            source.dtype,
        ))
    }

    /// Signed-axis form of [`Self::gather`], matching tinygrad's public
    /// `gather(dim, index)` axis contract without changing the established
    /// unsigned Rust entry point.
    ///
    /// Axis normalization completes before the delegated gather validator can
    /// append a node. The existing gather route then validates index dtype,
    /// equal rank, and every non-axis extent before constructing the node.
    pub fn gather_signed(&mut self, input: NodeId, index: NodeId, axis: isize) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        let axis = normalize_axes(input, rank, Some(vec![axis])).map(|axes| axes[0])?;
        self.gather(input, index, axis)
    }

    /// Applies a checked, immutable mixed static index. Advanced indices are
    /// constant integer tensors represented by [`indexing::StaticIndex`];
    /// data-dependent boolean/nonzero indexing is intentionally not this API.
    pub fn static_index(
        &mut self,
        input: NodeId,
        specs: &[indexing::StaticIndex],
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let plan = indexing::StaticIndexPlan::new(source.shape.clone(), specs)?;
        Ok(self.push(
            Op::StaticIndex {
                input,
                plan: plan.clone(),
            },
            plan.output_shape().clone(),
            source.dtype,
        ))
    }

    pub(crate) fn static_index_grad(
        &mut self,
        cotangent: NodeId,
        input_shape: Shape,
        plan: indexing::StaticIndexPlan,
    ) -> Result<NodeId> {
        let source = self.node(cotangent)?;
        if source.shape != *plan.output_shape() {
            return Err(Error::InvalidIndex);
        }
        Ok(self.push(
            Op::StaticIndexGrad {
                cotangent,
                input_shape: input_shape.clone(),
                plan,
            },
            input_shape,
            source.dtype,
        ))
    }

    pub(crate) fn static_index_update_grad(
        &mut self,
        cotangent: NodeId,
        base_shape: Shape,
        value_shape: Shape,
        plan: indexing::StaticIndexPlan,
        wrt: super::StaticIndexUpdateWrt,
    ) -> Result<NodeId> {
        let source = self.node(cotangent)?;
        if source.dtype != DType::F32 || source.shape != base_shape {
            return Err(Error::NonDifferentiableIndexing(
                "static index update gradients require an F32 base cotangent",
            ));
        }
        if value_shape.broadcast_with(plan.output_shape()).as_ref() != Ok(plan.output_shape()) {
            return Err(Error::InvalidIndex);
        }
        let shape = match wrt {
            super::StaticIndexUpdateWrt::Base => base_shape.clone(),
            super::StaticIndexUpdateWrt::Value => value_shape.clone(),
        };
        Ok(self.push(
            Op::StaticIndexUpdateGrad {
                cotangent,
                base_shape,
                value_shape,
                plan,
                wrt,
            },
            shape,
            DType::F32,
        ))
    }

    /// Functional immutable replacement. RHS dtype must equal the snapshot
    /// dtype and broadcasts to the indexed result; duplicate target positions
    /// resolve in row-major update order with the final write winning.
    pub fn static_index_update(
        &mut self,
        base: NodeId,
        specs: &[indexing::StaticIndex],
        value: NodeId,
    ) -> Result<NodeId> {
        let base_node = self.node(base)?;
        let value_node = self.node(value)?;
        if value_node.dtype != base_node.dtype {
            return Err(Error::InvalidElementwiseDType {
                op: "static_index_update",
                actual: value_node.dtype,
            });
        }
        let plan = indexing::StaticIndexPlan::new(base_node.shape.clone(), specs)?;
        if value_node
            .shape
            .broadcast_with(plan.output_shape())
            .as_ref()
            != Ok(plan.output_shape())
        {
            return Err(Error::ShapeMismatch {
                op: "static_index_update",
                lhs: value_node.shape.clone(),
                rhs: plan.output_shape().clone(),
            });
        }
        Ok(self.push(
            Op::StaticIndexUpdate { base, value, plan },
            base_node.shape.clone(),
            base_node.dtype,
        ))
    }

    /// Replaces indexed base positions. Duplicate indices are deterministic:
    /// row-major later update coordinates win. Replacement scatter is
    /// deliberately non-differentiable; use [`Graph::scatter_add`] for a
    /// differentiable accumulation operation.
    pub fn scatter(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
    ) -> Result<NodeId> {
        self.indexed_scatter(base, index, updates, axis, false)
    }

    /// Source-literal public tinygrad `Tensor.scatter`. This intentionally
    /// differs from the legacy raw [`Self::scatter`] operation: replacement
    /// folds one-hot update lanes in source order, while scalar add/multiply
    /// delegate to the public scatter-reduce composition.
    pub fn scatter_tinygrad(
        &mut self,
        base: NodeId,
        dim: isize,
        index: NodeId,
        source: ScatterSource,
        mode: ScatterMode,
    ) -> Result<NodeId> {
        let plan = scatter_plan(self, base, dim, index, source, mode)?;
        let output = lower_scatter(self, base, index, &plan)?;
        debug_assert_eq!(
            self.shape(output).expect("scatter preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("scatter preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// Checked-in tinygrad's omitted `reduce=None` argument.
    pub fn scatter_tinygrad_default(
        &mut self,
        base: NodeId,
        dim: isize,
        index: NodeId,
        source: ScatterSource,
    ) -> Result<NodeId> {
        self.scatter_tinygrad(base, dim, index, source, ScatterMode::Replace)
    }

    /// Source-literal public `Tensor.scatter_reduce`. Unlike raw Scatter,
    /// this keeps invalid live indices as all-false one-hot lanes and reduces
    /// a synthetic update axis with Select identities.
    pub fn scatter_reduce(
        &mut self,
        base: NodeId,
        dim: isize,
        index: NodeId,
        src: NodeId,
        kind: ScatterReduceKind,
        include_self: bool,
    ) -> Result<NodeId> {
        let plan = scatter_reduce_plan(self, base, index, src, dim, kind, include_self)?;
        let output = lower_scatter_reduce(self, base, index, src, &plan)?;
        debug_assert_eq!(
            self.shape(output).expect("scatter_reduce preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("scatter_reduce preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// Checked-in tinygrad's omitted `include_self` argument.
    pub fn scatter_reduce_default(
        &mut self,
        base: NodeId,
        dim: isize,
        index: NodeId,
        src: NodeId,
        kind: ScatterReduceKind,
    ) -> Result<NodeId> {
        self.scatter_reduce(base, dim, index, src, kind, true)
    }

    /// Adds updates into indexed base positions. Duplicate coordinates are
    /// accumulated in row-major order and result dtype promotes base/updates.
    pub fn scatter_add(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
    ) -> Result<NodeId> {
        self.indexed_scatter(base, index, updates, axis, true)
    }

    fn indexed_scatter(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
        add: bool,
    ) -> Result<NodeId> {
        let base_node = self.node(base)?;
        let index_node = self.node(index)?;
        let update_node = self.node(updates)?;
        validate_indexed("scatter", base_node, index_node, axis)?;
        if update_node.shape.rank() != index_node.shape.rank()
            || update_node
                .shape
                .dims()
                .iter()
                .zip(index_node.shape.dims())
                .any(|(update, index)| update < index)
        {
            return Err(Error::InvalidUpdateShape {
                index: index_node.shape.clone(),
                updates: update_node.shape.clone(),
            });
        }
        Ok(self.push(
            Op::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            },
            base_node.shape.clone(),
            base_node.dtype.promote(update_node.dtype),
        ))
    }

    /// Fixed-shape form of tinygrad's `masked_select(size=N)`. The mask must
    /// be bool and broadcastable to input; matches use row-major order.
    pub fn masked_select(
        &mut self,
        input: NodeId,
        mask: NodeId,
        size: usize,
        fill: Scalar,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let mask_node = self.node(mask)?;
        if mask_node.dtype != DType::Bool {
            return Err(Error::InvalidLogicalDType {
                op: "masked_select",
                actual: mask_node.dtype,
            });
        }
        if mask_node.shape.broadcast_with(&source.shape).as_ref() != Ok(&source.shape) {
            return Err(Error::InvalidIndexedShape {
                op: "masked_select",
                input: source.shape.clone(),
                index: mask_node.shape.clone(),
            });
        }
        Ok(self.push(
            Op::MaskedSelect {
                input,
                mask,
                size,
                fill,
            },
            Shape::from([size]),
            source.dtype,
        ))
    }

    /// Checked-in tinygrad's public `Tensor.dot(rhs, dtype)` composition.
    ///
    /// This intentionally does not route through raw [`Graph::matmul`]: dot
    /// has a source-visible typed Sum accumulator and final storage cast.
    pub fn dot(&mut self, lhs: NodeId, rhs: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        let plan = source_dot_plan(self, lhs, rhs, dtype)?;
        let lhs = self.reshape(lhs, plan.lhs_shape.clone())?;
        let rhs = self.reshape(rhs, plan.rhs_reshape.clone())?;
        let rhs = self.transpose(rhs, -1, plan.rhs_axis)?;
        let product = self.mul(lhs, rhs)?;
        let reduced = self.reduce_with_dtypes(
            product,
            ReduceKind::Sum,
            Some(vec![-1]),
            false,
            plan.sum_dtypes,
        )?;
        let output = if plan.sum_dtypes.output == plan.output_dtype {
            reduced
        } else {
            self.cast(reduced, plan.output_dtype)?
        };
        debug_assert_eq!(
            self.shape(rhs).expect("source dot preflighted"),
            &plan.rhs_shape
        );
        debug_assert_eq!(
            self.dtype(product).expect("source dot preflighted"),
            plan.operand_dtype
        );
        debug_assert_eq!(
            self.shape(product).expect("source dot preflighted"),
            &plan.product_shape
        );
        debug_assert_eq!(
            self.shape(output).expect("source dot preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("source dot preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.dot(rhs)` default typed accumulation.
    pub fn dot_default(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.dot(lhs, rhs, None)
    }

    /// Source-literal tinygrad `Tensor.matmul(rhs, reverse, dtype)`.
    ///
    /// Unlike the legacy raw [`Self::matmul`] node, this is exactly the
    /// public `dot` composition: reshape/transpose, source-LUB Mul, typed
    /// Sum, and its final storage cast. `reverse` implements `__rmatmul__`.
    pub fn matmul_tinygrad(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        reverse: bool,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        if reverse {
            self.dot(rhs, lhs, dtype)
        } else {
            self.dot(lhs, rhs, dtype)
        }
    }

    /// Tinygrad's default `Tensor.matmul(rhs)` / `__matmul__` surface.
    pub fn matmul_tinygrad_default(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.matmul_tinygrad(lhs, rhs, false, None)
    }

    /// Tinygrad's reflected `__rmatmul__` surface.
    pub fn rmatmul_tinygrad_default(&mut self, rhs: NodeId, lhs: NodeId) -> Result<NodeId> {
        self.matmul_tinygrad(rhs, lhs, true, None)
    }

    fn lower_qr(&mut self, input: NodeId, plan: &QrPlan) -> Result<(NodeId, NodeId)> {
        // `eye(m, dtype=self.dtype).expand(batch + (m, m))` and the one
        // default-integer range are exactly the source setup. Both creation
        // helpers are lazy/scalar-backed rather than dense control payloads.
        let mut r = input;
        let q = self.eye(plan.m, Some(plan.m), plan.dtype)?;
        let mut q = self.expand(q, plan.q_shape.clone())?;
        let index = self.lazy_arange_default_int(
            0,
            i64::try_from(plan.m).map_err(|_| Error::ShapeOverflow(plan.q_shape.clone()))?,
            1,
        )?;
        let last_axis = plan.r_shape.rank() - 1;

        for i in 0..plan.stages {
            let i_scalar = Scalar::I(
                i64::try_from(i).map_err(|_| Error::ShapeOverflow(plan.r_shape.clone()))?,
            );
            // `at_i, x = idx.eq(i), (idx >= i).where(R[..., :, i], 0)`.
            let at_i = self.eq_scalar(index, i_scalar)?;
            let from_i = self.ge_scalar(index, i_scalar)?;
            let mut bounds = plan
                .r_shape
                .dims()
                .iter()
                .map(|&extent| (0, extent))
                .collect::<Vec<_>>();
            bounds[last_axis] = (i, i + 1);
            let column = self.shrink(r, bounds)?;
            let column = self.squeeze(column, Some(-1))?;
            let x = self.where_false_scalar(from_i, column, Scalar::I(0))?;
            let squared = self.square(x)?;
            let norm = self.sum_with_options(squared, Some(vec![-1]), true, None)?;
            let norm = self.sqrt(norm)?;
            let x_at_i = self.where_false_scalar(at_i, x, Scalar::I(0))?;
            let x0 = self.sum_with_options(x_at_i, Some(vec![-1]), true, None)?;
            let x0_nonzero = self.ne_scalar(x0, Scalar::I(0))?;
            let x0_sign = self.sign(x0)?;
            let sgn = self.where_false_scalar(x0_nonzero, x0_sign, Scalar::I(1))?;
            let active = self.ne_scalar(norm, Scalar::I(0))?;
            let signed_norm = self.mul(sgn, norm)?;
            let u0 = self.add(x0, signed_norm)?;
            let numerator = self.select(at_i, u0, x)?;
            let denominator = self.where_false_scalar(active, u0, Scalar::I(1))?;
            let v = self.div(numerator, denominator)?;
            let v = self.unsqueeze(v, -1)?;
            let signed_u0 = self.mul(sgn, u0)?;
            let safe_norm = self.where_false_scalar(active, norm, Scalar::I(1))?;
            let w_scale = self.div(signed_u0, safe_norm)?;
            let w_scale = self.where_false_scalar(active, w_scale, Scalar::I(0))?;
            let w_scale = self.unsqueeze(w_scale, -1)?;
            let w = self.mul(w_scale, v)?;

            // `R = R - w @ (v.T @ R)` followed by the source-ordered Q
            // update. `dot` supplies the exact typed accumulation/cast-back
            // contract for each rank-one product.
            let v_t = self.transpose(v, -2, -1)?;
            let r_projection = self.dot_default(v_t, r)?;
            let r_update = self.dot_default(w, r_projection)?;
            r = self.sub(r, r_update)?;
            let q_projection = self.dot_default(q, v)?;
            let w_t = self.transpose(w, -2, -1)?;
            let q_update = self.dot_default(q_projection, w_t)?;
            q = self.sub(q, q_update)?;
        }
        debug_assert_eq!(self.shape(q).expect("qr preflighted"), &plan.q_shape);
        debug_assert_eq!(self.shape(r).expect("qr preflighted"), &plan.r_shape);
        Ok((q, r))
    }

    /// Checked-in tinygrad's full-Q static Householder `Tensor.qr()`.
    pub fn qr(&mut self, input: NodeId) -> Result<(NodeId, NodeId)> {
        let plan = qr_plan(self, input)?;
        // Rehearse the whole literal composition on a private graph clone.
        // Every nested helper (including typed Dot) then validates all of its
        // movement, scalar, broadcast, reduction, and byte descriptors before
        // the real graph publishes its first node. The clone has no external
        // graph-visible effect and retains the same input NodeIds.
        let mut staged = self.clone();
        let (staged_q, staged_r) = staged.lower_qr(input, &plan)?;
        debug_assert_eq!(
            staged.shape(staged_q).expect("qr stage preflighted"),
            &plan.q_shape
        );
        debug_assert_eq!(
            staged.shape(staged_r).expect("qr stage preflighted"),
            &plan.r_shape
        );
        self.lower_qr(input, &plan)
    }

    fn lower_svd(&mut self, input: NodeId, plan: &SvdPlan) -> Result<(NodeId, NodeId, NodeId)> {
        let qr_input = if plan.transpose_input {
            self.transpose(input, -2, -1)?
        } else {
            input
        };
        let (q, r) = self.qr(qr_input)?;

        let square_bounds = |shape: &Shape, extent: usize| {
            let rank = shape.rank();
            shape
                .dims()
                .iter()
                .enumerate()
                .map(|(axis, &size)| {
                    if axis + 2 >= rank {
                        (0, extent)
                    } else {
                        (0, size)
                    }
                })
                .collect::<Vec<_>>()
        };
        let r_shape = self.shape(r)?.clone();
        let r_square_bounds = square_bounds(&r_shape, plan.num);
        let r_square = self.shrink(r, r_square_bounds)?;
        // Both barriers are source-visible. The scheduler may redirect an
        // eligible producer into their owned outputs, but may not erase either
        // Contiguous node identity.
        let mut u = self.contiguous(r_square)?;
        let eye = self.eye(plan.num, Some(plan.num), plan.dtype)?;
        let mut core_dims = plan.batch.clone();
        core_dims.extend([plan.num, plan.num]);
        let core_shape = Shape::new(core_dims);
        let mut v = self.expand(eye, core_shape.clone())?;
        v = self.contiguous(v)?;

        let h_i64 =
            i64::try_from(plan.h).map_err(|_| Error::ShapeOverflow(plan.input_shape.clone()))?;
        let num_i64 =
            i64::try_from(plan.num).map_err(|_| Error::ShapeOverflow(plan.input_shape.clone()))?;
        let first = self.lazy_arange_default_int(0, h_i64, 1)?;
        let second = self.lazy_arange_default_int(h_i64, num_i64, 1)?;
        let second = self.flip(second, [0])?;
        let permutation = self.cat(first, vec![second], 0)?;
        let columns = self.lazy_arange_default_int(0, num_i64, 1)?;
        let eye_num = self.eye(plan.num, Some(plan.num), plan.dtype)?;
        let eye_num = self.expand(eye_num, core_shape.clone())?;

        let mut state = SvdJacobiState { u, v, permutation };
        for _ in 0..plan.rounds {
            state = svd_jacobi_round(self, plan, state, columns, eye_num)?;
        }
        u = state.u;
        v = state.v;

        let singular = self.square(u)?;
        let singular = self.sum_with_options(singular, Some(vec![-2]), false, None)?;
        let singular = self.sqrt(singular)?;
        let (singular, indices) = self.sort(singular, -1, true)?;
        let expanded_indices = self.unsqueeze(indices, -2)?;
        let expanded_indices = self.expand(expanded_indices, core_shape.clone())?;
        u = self.gather_tinygrad(u, -1, expanded_indices)?;
        let nonzero = self.ne_scalar(singular, Scalar::I(0))?;
        let denominator = self.where_false_scalar(nonzero, singular, Scalar::I(1))?;
        let denominator = self.unsqueeze(denominator, -2)?;
        u = self.div(u, denominator)?;
        v = self.gather_tinygrad(v, -1, expanded_indices)?;

        let padding = plan
            .batch
            .iter()
            .map(|_| (0, 0))
            .chain(std::iter::repeat_n((0, plan.q_num - plan.num), 2))
            .collect::<Vec<_>>();
        let u_dtype = self.dtype(u)?;
        let mut q_shape = plan.batch.clone();
        q_shape.extend([plan.q_num, plan.q_num]);
        let q_shape = Shape::new(q_shape);
        let eye_q = self.eye(plan.q_num, Some(plan.q_num), u_dtype)?;
        let eye_q = self.expand(eye_q, q_shape)?;
        let eye_n = self.eye(plan.num, Some(plan.num), u_dtype)?;
        let eye_n = self.expand(eye_n, core_shape)?;
        let eye_n = self.pad(eye_n, padding.clone(), Scalar::I(0))?;
        let u = self.pad(u, padding, Scalar::I(0))?;
        let u = self.add(u, eye_q)?;
        let u = self.sub(u, eye_n)?;
        let mut u = self.dot_default(q, u)?;
        if !plan.full_matrices {
            let u_shape = self.shape(u)?.clone();
            let last = u_shape.rank() - 1;
            let bounds = u_shape
                .dims()
                .iter()
                .enumerate()
                .map(|(axis, &extent)| {
                    if axis == last {
                        (0, plan.num)
                    } else {
                        (0, extent)
                    }
                })
                .collect::<Vec<_>>();
            u = self.shrink(u, bounds)?;
        }
        let v_t = self.transpose(v, -2, -1)?;
        let outputs = if plan.m >= plan.n {
            (u, singular, v_t)
        } else {
            let u_t = self.transpose(u, -2, -1)?;
            (v, singular, u_t)
        };
        debug_assert_eq!(
            self.shape(outputs.0).expect("svd preflighted"),
            &plan.left_shape
        );
        debug_assert_eq!(
            self.shape(outputs.1).expect("svd preflighted"),
            &plan.singular_shape
        );
        debug_assert_eq!(
            self.shape(outputs.2).expect("svd preflighted"),
            &plan.right_shape
        );
        Ok(outputs)
    }

    /// Checked-in tinygrad's static Jacobi `Tensor.svd(full_matrices)`.
    ///
    /// This is a source-composed graph rather than an opaque SVD
    /// operation. The returned tuple is `(U, S, Vt)`. Its coupled Sort stage is
    /// currently CPU-interpreter-only; capture, strict native JIT, and device
    /// routes retain their existing fail-closed Sort boundary.
    pub fn svd(&mut self, input: NodeId, full_matrices: bool) -> Result<(NodeId, NodeId, NodeId)> {
        let plan = svd_plan(self, input, full_matrices)?;
        let mut staged = self.clone();
        let staged_outputs = staged.lower_svd(input, &plan)?;
        for (output, expected) in [
            (staged_outputs.0, &plan.left_shape),
            (staged_outputs.1, &plan.singular_shape),
            (staged_outputs.2, &plan.right_shape),
        ] {
            let shape = staged.shape(output)?;
            let dtype = staged.dtype(output)?;
            debug_assert_eq!(shape, expected);
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        }
        self.lower_svd(input, &plan)
    }

    /// Checked-in tinygrad's default `Tensor.svd()` form (`full_matrices=true`).
    pub fn svd_default(&mut self, input: NodeId) -> Result<(NodeId, NodeId, NodeId)> {
        self.svd(input, true)
    }

    fn lower_newton_schulz(
        &mut self,
        input: NodeId,
        plan: &NewtonSchulzPlan,
        params: &[i64],
        eps: f64,
    ) -> Result<NodeId> {
        // The source handles tall matrices through an exact transpose recurse
        // shell; the working matrix is therefore always no taller than wide.
        debug_assert_eq!(
            self.dtype(input).expect("newton_schulz preflighted"),
            plan.dtype
        );
        let working = if plan.transpose_input {
            self.transpose(input, -2, -1)?
        } else {
            input
        };
        // `G = self / (sqrt(sum(square(self), (-2,-1), keepdim=True)) + eps)`.
        let squared = self.square(working)?;
        let norm = self.sum_with_options(squared, Some(vec![-2, -1]), true, None)?;
        let norm = self.sqrt(norm)?;
        let denominator = self.add_scalar(norm, Scalar::F(eps))?;
        let mut g = self.div(working, denominator)?;

        for _ in 0..plan.iterations {
            // `functools.reduce` has no initializer: the checked plan only
            // reaches this loop with nonempty params. Preserve generator order
            // and keep each `(y @ y.T) @ x` expansion separate as in source.
            let base = g;
            let mut next = None;
            for (power, coefficient) in params.iter().copied().enumerate() {
                let mut x = base;
                for _ in 0..power {
                    let base_t = self.transpose(base, -2, -1)?;
                    let gram = self.dot_default(base, base_t)?;
                    x = self.dot_default(gram, x)?;
                }
                let term = self.scalar_mul(Scalar::I(coefficient), x)?;
                next = Some(match next {
                    Some(accumulator) => self.add(accumulator, term)?,
                    None => term,
                });
            }
            g = next.expect("positive Newton-Schulz step has checked params");
        }
        let output = if plan.transpose_input {
            self.transpose(g, -2, -1)?
        } else {
            g
        };
        debug_assert_eq!(
            self.shape(output).expect("newton_schulz preflighted"),
            &plan.input_shape
        );
        Ok(output)
    }

    /// Checked-in tinygrad's concrete static Newton--Schulz polynomial helper.
    pub fn newton_schulz(
        &mut self,
        input: NodeId,
        steps: isize,
        params: &[i64],
        eps: f64,
    ) -> Result<NodeId> {
        let plan = newton_schulz_plan(self, input, steps, params)?;
        // The staging graph proves every iteration's reshape, typed reduction,
        // scalar commitment, source Dot, broadcast, and byte extent before a
        // live constant or node is emitted.
        let mut staged = self.clone();
        let staged_output = staged.lower_newton_schulz(input, &plan, params, eps)?;
        debug_assert_eq!(
            staged.shape(staged_output).expect("newton_schulz staged"),
            &plan.input_shape
        );
        self.lower_newton_schulz(input, &plan, params, eps)
    }

    /// Source-default epsilon (`1e-7`) form. Steps and polynomial parameters
    /// are required by tinygrad and intentionally have no invented defaults.
    pub fn newton_schulz_default_eps(
        &mut self,
        input: NodeId,
        steps: isize,
        params: &[i64],
    ) -> Result<NodeId> {
        self.newton_schulz(input, steps, params, 1.0e-7)
    }

    pub fn matmul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let lhs_shape = &self.node(lhs)?.shape;
        let rhs_shape = &self.node(rhs)?.shape;
        let Some(shape) = matmul_shape(lhs_shape, rhs_shape) else {
            return Err(Error::InvalidMatmul {
                lhs: lhs_shape.clone(),
                rhs: rhs_shape.clone(),
            });
        };
        shape.numel()?;
        let dtype = self.node(lhs)?.dtype.promote(self.node(rhs)?.dtype);
        Ok(self.push(Op::Matmul { lhs, rhs }, shape, dtype))
    }

    /// Applies checked-in tinygrad's functional linear composition.
    ///
    /// Rank-one weights are elementwise multipliers; all other weights are
    /// passed unchanged to source-literal [`Self::dot_default`]. In
    /// particular this method does not transpose a conventional NN layout or
    /// publish raw Matmul.
    pub fn linear(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        let plan = linear_plan(self, input, weight, bias, dtype)?;
        let output = lower_linear(self, input, weight, bias, plan.dtype)?;
        debug_assert_eq!(
            self.shape(output).expect("linear preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("linear preflighted"),
            plan.output_dtype
        );
        debug_assert_eq!(
            self.shape(weight).expect("linear preflighted").rank() == 1,
            plan.rank_one_weight
        );
        Ok(output)
    }

    /// Adds a static dense Einstein summation node with NumPy/tinygrad-style
    /// subscript grammar, including ellipses and repeated-label diagonals.
    pub fn einsum(&mut self, equation: &str, inputs: &[NodeId]) -> Result<NodeId> {
        self.einsum_impl(equation, inputs, None)
    }

    /// Adds an Einstein summation with an explicit reduction/output dtype.
    pub fn einsum_with_dtype(
        &mut self,
        equation: &str,
        inputs: &[NodeId],
        dtype: DType,
    ) -> Result<NodeId> {
        self.einsum_impl(equation, inputs, Some(dtype))
    }

    fn einsum_impl(
        &mut self,
        equation: &str,
        inputs: &[NodeId],
        accumulation_dtype: Option<DType>,
    ) -> Result<NodeId> {
        let shapes = inputs
            .iter()
            .map(|id| Ok(self.node(*id)?.shape.clone()))
            .collect::<Result<Vec<_>>>()?;
        let plan = EinsumPlan::parse(equation, &shapes)?;
        let product_dtype = inputs.iter().try_fold(DType::Bool, |dtype, id| {
            Ok::<_, Error>(dtype.promote(self.node(*id)?.dtype))
        })?;
        if let Some(dtype) = accumulation_dtype {
            if !matches!(dtype, DType::F32 | DType::F64) {
                return Err(Error::InvalidEinsum {
                    equation: equation.to_owned(),
                    reason: "einsum dtype overrides currently support only f32 or f64",
                });
            }
            if product_dtype.is_float8() {
                return Err(Error::InvalidEinsum {
                    equation: equation.to_owned(),
                    reason: "einsum dtype overrides for float8 products are not implemented",
                });
            }
        }
        let dtype = accumulation_dtype.unwrap_or(product_dtype);
        Ok(self.push(
            Op::Einsum {
                inputs: inputs.to_vec(),
                plan: plan.clone(),
                product_dtype,
                accumulation_dtype,
            },
            plan.output_shape(),
            dtype,
        ))
    }

    pub(crate) fn einsum_grad(
        &mut self,
        upstream: NodeId,
        inputs: &[NodeId],
        plan: EinsumPlan,
        target: usize,
    ) -> Result<NodeId> {
        let target_id = *inputs.get(target).ok_or(Error::InvalidIndex)?;
        let target_node = self.node(target_id)?;
        let output_shape = plan.output_shape();
        if self.node(upstream)?.shape != output_shape {
            return Err(Error::ShapeMismatch {
                op: "einsum gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output_shape,
            });
        }
        if !target_node.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "einsum gradients require floating point target tensors",
            ));
        }
        Ok(self.push(
            Op::EinsumGrad {
                upstream,
                inputs: inputs.to_vec(),
                plan,
                target,
            },
            target_node.shape.clone(),
            target_node.dtype,
        ))
    }

    pub(crate) fn einsum_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        inputs: &[NodeId],
        plan: EinsumPlan,
        target: usize,
        wrt: usize,
    ) -> Result<NodeId> {
        let output = if wrt == inputs.len() {
            plan.output_shape()
        } else {
            self.node(*inputs.get(wrt).ok_or(Error::InvalidIndex)?)?
                .shape
                .clone()
        };
        Ok(self.push(
            Op::EinsumGradVjp {
                cotangent,
                upstream,
                inputs: inputs.to_vec(),
                plan,
                target,
                wrt,
            },
            output,
            self.node(cotangent)?.dtype,
        ))
    }

    /// NCHW/OIHW syntax adapter over the rank-generic compositional core.
    pub fn conv2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
    ) -> Result<NodeId> {
        let input_shape = self.shape(input)?.clone();
        let weight_shape = self.shape(weight)?.clone();
        if input_shape.rank() != 4 || weight_shape.rank() != 4 {
            return Err(Error::InvalidConv2d {
                input: input_shape,
                weight: weight_shape,
                reason: "input and weight must be rank 4",
            });
        }
        let groups = std::num::NonZeroUsize::new(options.groups).ok_or(Error::InvalidConv2d {
            input: input_shape.clone(),
            weight: weight_shape.clone(),
            reason: "groups, stride, and dilation must be positive",
        })?;
        let padding = options
            .padding
            .into_iter()
            .map(|value| {
                i64::try_from(value).map_err(|_| Error::ShapeOverflow(input_shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let window = SpatialWindow::new(
            weight_shape.dims()[2..].to_vec(),
            options.stride,
            options.dilation,
            [(padding[0], padding[1]), (padding[2], padding[3])],
        )
        .map_err(|_| Error::InvalidConv2d {
            input: input_shape.clone(),
            weight: weight_shape.clone(),
            reason: "groups, stride, and dilation must be positive",
        })?;
        self.convolution(
            input,
            weight,
            bias,
            ConvolutionSpec::new(window, groups, None),
        )
        .map_err(|error| match error {
            Error::InvalidConvolution { reason, .. } => Error::InvalidConv2d {
                input: input_shape,
                weight: weight_shape,
                reason,
            },
            error => error,
        })
    }
    pub fn conv_transpose2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
    ) -> Result<NodeId> {
        let input_shape = self.shape(input)?.clone();
        let weight_shape = self.shape(weight)?.clone();
        if input_shape.rank() != 4 || weight_shape.rank() != 4 {
            return Err(Error::InvalidConv2d {
                input: input_shape,
                weight: weight_shape,
                reason: "input and weight must be rank 4",
            });
        }
        let groups =
            std::num::NonZeroUsize::new(options.groups).ok_or_else(|| Error::InvalidConv2d {
                input: input_shape.clone(),
                weight: weight_shape.clone(),
                reason: "invalid transpose convolution geometry",
            })?;
        let padding = options
            .padding
            .into_iter()
            .map(|value| {
                i64::try_from(value).map_err(|_| Error::ShapeOverflow(input_shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let output_padding = options
            .output_padding
            .into_iter()
            .map(|value| {
                i64::try_from(value).map_err(|_| Error::ShapeOverflow(input_shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let window = SpatialWindow::new(
            weight_shape.dims()[2..].to_vec(),
            options.stride,
            options.dilation,
            [(padding[0], padding[1]), (padding[2], padding[3])],
        )
        .map_err(|_| Error::InvalidConv2d {
            input: input_shape.clone(),
            weight: weight_shape.clone(),
            reason: "invalid transpose convolution geometry",
        })?;
        let spec =
            TransposedConvolutionSpec::new(window, output_padding, groups).map_err(|_| {
                Error::InvalidConv2d {
                    input: input_shape.clone(),
                    weight: weight_shape.clone(),
                    reason: "invalid transpose convolution geometry",
                }
            })?;
        self.transposed_convolution(input, weight, bias, spec)
            .map_err(|error| match error {
                Error::InvalidConvolution { reason, .. } => Error::InvalidConv2d {
                    input: input_shape,
                    weight: weight_shape,
                    reason,
                },
                error => error,
            })
    }
    /// NCL/IOK syntax adapter over the rank-generic compositional core.
    pub fn conv_transpose1d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose1dOptions,
    ) -> Result<NodeId> {
        let input_shape = self.shape(input)?.clone();
        let weight_shape = self.shape(weight)?.clone();
        if input_shape.rank() != 3 || weight_shape.rank() != 3 {
            return Err(Error::InvalidConv2d {
                input: input_shape,
                weight: weight_shape,
                reason: "invalid 1d transpose convolution geometry",
            });
        }
        let groups =
            std::num::NonZeroUsize::new(options.groups).ok_or_else(|| Error::InvalidConv2d {
                input: input_shape.clone(),
                weight: weight_shape.clone(),
                reason: "invalid transpose convolution geometry",
            })?;
        let padding = options
            .padding
            .into_iter()
            .map(|value| {
                i64::try_from(value).map_err(|_| Error::ShapeOverflow(input_shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let output_padding = i64::try_from(options.output_padding)
            .map_err(|_| Error::ShapeOverflow(input_shape.clone()))?;
        let window = SpatialWindow::new(
            weight_shape.dims()[2..].to_vec(),
            [options.stride],
            [options.dilation],
            [(padding[0], padding[1])],
        )
        .map_err(|_| Error::InvalidConv2d {
            input: input_shape.clone(),
            weight: weight_shape.clone(),
            reason: "invalid transpose convolution geometry",
        })?;
        let spec =
            TransposedConvolutionSpec::new(window, [output_padding], groups).map_err(|_| {
                Error::InvalidConv2d {
                    input: input_shape.clone(),
                    weight: weight_shape.clone(),
                    reason: "invalid transpose convolution geometry",
                }
            })?;
        self.transposed_convolution(input, weight, bias, spec)
            .map_err(|error| match error {
                Error::InvalidConvolution { reason, .. } => Error::InvalidConv2d {
                    input: input_shape,
                    weight: weight_shape,
                    reason,
                },
                error => error,
            })
    }
    pub(crate) fn conv_transpose2d_grad(
        &mut self,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
        target: u8,
    ) -> Result<NodeId> {
        let output =
            conv_transpose2d_shape(&self.node(input)?.shape, &self.node(weight)?.shape, options)?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let node = match target {
            0 => input,
            1 => weight,
            2 => bias.ok_or(Error::NonDifferentiableIndexing("missing transpose bias"))?,
            _ => return Err(Error::InvalidIndex),
        };
        let n = self.node(node)?;
        if !n.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "transpose convolution gradients require floating point tensors",
            ));
        }
        Ok(self.push(
            Op::ConvTranspose2dGrad {
                upstream,
                input,
                weight,
                bias,
                options,
                target,
            },
            n.shape.clone(),
            n.dtype,
        ))
    }

    pub(crate) fn conv2d_grad(
        &mut self,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
        target: u8,
    ) -> Result<NodeId> {
        let output = conv2d_shape(&self.node(input)?.shape, &self.node(weight)?.shape, options)?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "conv2d gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let target_node = match target {
            0 => input,
            1 => weight,
            2 => bias.ok_or(Error::NonDifferentiableIndexing("missing conv2d bias"))?,
            _ => return Err(Error::InvalidIndex),
        };
        let target_data = self.node(target_node)?;
        if !target_data.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "conv2d gradients require floating point tensors",
            ));
        }
        Ok(self.push(
            Op::Conv2dGrad {
                upstream,
                input,
                weight,
                bias,
                options,
                target,
            },
            target_data.shape.clone(),
            target_data.dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv2d_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
        target: u8,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = conv_vjp_shape(self, upstream, input, weight, bias, wrt)?;
        Ok(self.push(
            Op::Conv2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                options,
                target,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_transpose2d_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
        target: u8,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = conv_vjp_shape(self, upstream, input, weight, bias, wrt)?;
        Ok(self.push(
            Op::ConvTranspose2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                options,
                target,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    pub(crate) fn matmul_grad(
        &mut self,
        upstream: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        lhs_gradient: bool,
    ) -> Result<NodeId> {
        let lhs_shape = self.node(lhs)?.shape.clone();
        let rhs_shape = self.node(rhs)?.shape.clone();
        let output = matmul_shape(&lhs_shape, &rhs_shape).ok_or_else(|| Error::InvalidMatmul {
            lhs: lhs_shape.clone(),
            rhs: rhs_shape.clone(),
        })?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "matmul gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let target = if lhs_gradient { lhs } else { rhs };
        let shape = self.node(target)?.shape.clone();
        let dtype = self.node(target)?.dtype;
        // Homogeneous F32/F64 raw Matmul has the same reshape/transpose,
        // product, and contraction geometry as the source Dot composition.
        // Build the requested local edge from those already scheduleable
        // primitives, then apply reverse mode's one canonical typed
        // unbroadcast projection. Narrow and mixed storage retain the
        // dedicated coordinate-map operation because changing their
        // recurrence/rounding boundary is not an equivalent lowering.
        if self.node(upstream)?.dtype == dtype
            && self.node(lhs)?.dtype == dtype
            && self.node(rhs)?.dtype == dtype
            && matches!(dtype, DType::F32 | DType::F64)
        {
            let mut candidate = self.clone();
            let plan = source_dot_plan(&candidate, lhs, rhs, None)?;
            if plan.output_shape != output
                || plan.operand_dtype != dtype
                || plan.sum_dtypes != ReductionDType::new(dtype, dtype)
            {
                return Err(Error::InvalidElementwiseDType {
                    op: "raw matmul gradient composition",
                    actual: dtype,
                });
            }
            let mut expanded_upstream_shape = plan.output_shape.dims().to_vec();
            expanded_upstream_shape.push(1);
            let expanded_upstream =
                candidate.reshape(upstream, Shape::new(expanded_upstream_shape))?;
            let expanded_upstream =
                candidate.expand(expanded_upstream, plan.product_shape.clone())?;
            let gradient = if lhs_gradient {
                let aligned_rhs = candidate.reshape(rhs, plan.rhs_reshape.clone())?;
                let aligned_rhs = candidate.transpose(aligned_rhs, -1, plan.rhs_axis)?;
                let local = candidate.mul(expanded_upstream, aligned_rhs)?;
                let local = candidate.unbroadcast(local, plan.lhs_shape)?;
                candidate.reshape(local, lhs_shape.clone())?
            } else {
                let aligned_lhs = candidate.reshape(lhs, plan.lhs_shape)?;
                let local = candidate.mul(expanded_upstream, aligned_lhs)?;
                let local = candidate.unbroadcast(local, plan.rhs_shape)?;
                let local = candidate.transpose(local, -1, plan.rhs_axis)?;
                candidate.reshape(local, rhs_shape.clone())?
            };
            if candidate.shape(gradient)? != &shape || candidate.dtype(gradient)? != dtype {
                return Err(Error::ShapeMismatch {
                    op: "raw matmul gradient composition",
                    lhs: candidate.shape(gradient)?.clone(),
                    rhs: shape,
                });
            }
            *self = candidate;
            return Ok(gradient);
        }
        Ok(self.push(
            Op::MatmulGrad {
                upstream,
                lhs,
                rhs,
                lhs_gradient,
            },
            shape,
            dtype,
        ))
    }

    pub(crate) fn matmul_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        lhs_gradient: bool,
        wrt: u8,
    ) -> Result<NodeId> {
        let output = match wrt {
            0 => matmul_shape(&self.node(lhs)?.shape, &self.node(rhs)?.shape).ok_or_else(|| {
                Error::InvalidMatmul {
                    lhs: self.node(lhs).unwrap().shape.clone(),
                    rhs: self.node(rhs).unwrap().shape.clone(),
                }
            })?,
            1 => self.node(lhs)?.shape.clone(),
            2 => self.node(rhs)?.shape.clone(),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(self.push(
            Op::MatmulGradVjp {
                cotangent,
                upstream,
                lhs,
                rhs,
                lhs_gradient,
                wrt,
            },
            output,
            self.node(cotangent)?.dtype,
        ))
    }

    pub fn shape(&self, id: NodeId) -> Result<&Shape> {
        Ok(&self.node(id)?.shape)
    }

    pub fn dtype(&self, id: NodeId) -> Result<DType> {
        Ok(self.node(id)?.dtype)
    }

    /// Read-only tinygrad `Tensor.nbytes()`: concrete element count times
    /// storage item size. This never appends a graph node.
    pub fn nbytes(&self, id: NodeId) -> Result<usize> {
        let node = self.node(id)?;
        node.shape
            .numel()?
            .checked_mul(node.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(node.shape.clone()))
    }

    /// Read-only tinygrad `Tensor.numel()`. This never appends a graph node.
    pub fn numel(&self, id: NodeId) -> Result<usize> {
        self.node(id)?.shape.numel()
    }

    /// Read-only tinygrad `Tensor.ndim`. RustGrad stores only concrete shapes,
    /// so this is the concrete descriptor rank.
    pub fn ndim(&self, id: NodeId) -> Result<usize> {
        Ok(self.node(id)?.shape.rank())
    }

    /// Read-only tinygrad `Tensor.max_shape` in RustGrad's concrete-shape
    /// model. There are no symbolic extents to substitute, so it is an owned
    /// copy of the concrete shape.
    pub fn max_shape(&self, id: NodeId) -> Result<Shape> {
        Ok(self.node(id)?.shape.clone())
    }

    /// Read-only tinygrad `Tensor.max_numel` for a concrete RustGrad shape.
    pub fn max_numel(&self, id: NodeId) -> Result<usize> {
        self.node(id)?.shape.numel()
    }

    /// Read-only checked-in tinygrad `Tensor.__len__`.
    ///
    /// Concrete non-scalar tensors return their leading extent, including
    /// zero. Scalars reject with the same source-visible 0-d length error;
    /// no descriptor product is needed, so overflow-shaped descriptors remain
    /// queryable when their leading extent is concrete.
    pub fn len_tinygrad(&self, id: NodeId) -> Result<usize> {
        self.node(id)?
            .shape
            .dims()
            .first()
            .copied()
            .ok_or(Error::InvalidTensorLen { node: id })
    }

    /// Read-only checked-in tinygrad `Tensor.__bool__`.
    ///
    /// Tensor truthiness is deliberately undefined for every valid tensor.
    /// Validate the graph reference first so an invalid node remains observable
    /// as `UnknownNode`; do not inspect descriptors or append a graph node.
    pub fn bool_tinygrad(&self, id: NodeId) -> Result<bool> {
        self.node(id)?;
        Err(Error::TensorBoolNotDefined)
    }

    /// Read-only tinygrad `Tensor.size()` without a dimension.
    ///
    /// The owned `Shape` keeps this query independent of the graph's internal
    /// node storage while preserving its concrete, ordered extents.
    pub fn size(&self, id: NodeId) -> Result<Shape> {
        Ok(self.node(id)?.shape.clone())
    }

    /// Read-only tinygrad `Tensor.size(dim)` with Python-style signed axes.
    pub fn size_dim(&self, id: NodeId, dim: isize) -> Result<usize> {
        let node = self.node(id)?;
        let rank = node.shape.rank() as isize;
        let axis = if dim < 0 {
            dim.checked_add(rank).ok_or(Error::InvalidAxis {
                node: id,
                axis: usize::MAX,
                rank: node.shape.rank(),
            })?
        } else {
            dim
        };
        if axis < 0 || axis >= rank {
            return Err(Error::InvalidAxis {
                node: id,
                axis: usize::try_from(axis).unwrap_or(usize::MAX),
                rank: node.shape.rank(),
            });
        }
        Ok(node.shape.dims()[axis as usize])
    }

    /// Read-only tinygrad `Tensor.element_size()`: storage bytes per element.
    pub fn element_size(&self, id: NodeId) -> Result<usize> {
        Ok(self.node(id)?.dtype.itemsize())
    }

    /// Source-literal `Tensor.sequential`: left-fold an ordered callable list.
    ///
    /// There is intentionally no clone/rollback wrapper here. Python's
    /// `functools.reduce` publishes effects from earlier callables before a
    /// later callable raises, and arbitrary user transforms may have their own
    /// graph side effects. An empty sequence returns the original identity.
    pub fn sequential(
        &mut self,
        input: NodeId,
        transforms: impl IntoIterator<Item = GraphSequentialTransform>,
    ) -> Result<NodeId> {
        self.node(input)?;
        transforms
            .into_iter()
            .try_fold(input, |current, transform| transform(self, current))
    }

    /// Returns the typed operation for inspection without exposing graph
    /// storage internals.
    pub fn op(&self, id: NodeId) -> Result<&Op> {
        Ok(&self.node(id)?.op)
    }

    pub fn trace(&self, output: NodeId) -> Result<CompileTrace> {
        self.node(output)?;
        let steps = self.nodes[..=output.index()]
            .iter()
            .enumerate()
            .map(|(id, node)| TraceStep {
                node: NodeId(id),
                operation: node.op.label(),
                shape: node.shape.clone(),
                dtype: node.dtype,
            })
            .collect();
        Ok(CompileTrace { output, steps })
    }

    pub(crate) fn push(&mut self, op: Op, shape: Shape, dtype: DType) -> NodeId {
        let requires_grad =
            self.grad_enabled && dtype.is_float() && self.op_inputs_require_grad(&op);
        self.push_with_grad(op, shape, dtype, requires_grad)
    }

    fn push_with_grad(
        &mut self,
        op: Op,
        shape: Shape,
        dtype: DType,
        requires_grad: bool,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            op,
            shape,
            dtype,
            requires_grad,
        });
        id
    }

    fn op_inputs_require_grad(&self, op: &Op) -> bool {
        let tracked = |id: NodeId| {
            self.nodes
                .get(id.index())
                .is_some_and(|node| node.requires_grad)
        };
        match op {
            // Sort's coupled values/indices producer deliberately retains its
            // historical leaf bit. Its values still participate in explicit
            // reverse transforms through `backward_inputs`; changing this bit
            // would be an unrelated public lifecycle-policy change.
            Op::Sort { .. } => false,
            _ => op.backward_inputs().into_iter().any(tracked),
        }
    }

    pub(crate) fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes.get(id.index()).ok_or(Error::UnknownNode(id))
    }

    pub(super) fn broadcast_shape(&self, lhs: NodeId, rhs: NodeId) -> Result<Shape> {
        let lhs = &self.node(lhs)?.shape;
        let rhs = &self.node(rhs)?.shape;
        lhs.broadcast_with(rhs)
    }
}
