// The operation declaration is the one source of truth for universal UOp
// structure. Each row records, in order: the payload-bearing enum variant,
// source arity, purity, private wire opcode/payload, stable top-level tag,
// historical decode gate, and optional wire subtag codec. Contextual dtype and
// topology checks, payload codecs, scheduling, and backend capability policy
// intentionally remain explicit at their own semantic boundaries.
macro_rules! uop_operation_schema {
    ($emit:ident) => {
        $emit! {
            Const(value: LiteralValue) => [zero, pure, Const, 0, always, none];
            VConst(value: LiteralValue) => [zero, pure, VConst, 1, always, none];
            DefineVar(value: VariableValue) => [zero, pure, DefineVar, 2, always, none];
            DefineGlobal(value: AddressValue) => [zero, pure, DefineGlobal, 3, always, none];
            DefineLocal(value: AddressValue) => [zero, pure, DefineLocal, 4, always, none];
            DefineRegister(value: AddressValue) => [zero, pure, DefineRegister, 5, always, none];
            Special(value: String) => [zero, pure, Special, 6, always, none];
            Range(value: u32) => [one, pure, Range, 7, always, none];
            EndRange => [one, impure, EndRange, 8, always, none];
            If => [one, impure, If, 9, always, none];
            EndIf => [one, impure, EndIf, 10, always, none];
            Unary(value: super::Unary) => [one, pure, Unary(wire: super::Unary), 11, always, unary];
            Binary(value: super::Binary) => [two, pure, Binary(wire: super::Binary), 12, always, binary];
            /// High-level ALU semantic retained by the portable interpreter.
            GraphUnary(value: crate::UnaryOp) => [one, pure, GraphUnary(wire: crate::UnaryOp), 13, always, graph_unary];
            GraphBinary(value: crate::BinaryOp) => [two, pure, GraphBinary(wire: crate::BinaryOp), 14, always, graph_binary];
            GraphCompare(value: crate::CompareOp) => [two, pure, GraphCompare(wire: crate::CompareOp), 15, always, compare];
            GraphLogical(value: crate::LogicalOp) => [graph_logical, pure, GraphLogical(wire: crate::LogicalOp), 16, always, logical];
            /// Live, two-input packed-U64 Threefry2x32 permutation.
            Threefry(value: ThreefryValue) => [zero, pure, Threefry, 39, v19, none];
            /// Complete static generalized-matmul semantic.
            Matmul(value: MatmulValue) => [zero, pure, Matmul, 30, v3, none];
            /// Narrow static F32 NCHW 1x1 convolution semantic.
            Conv2d(value: Box<crate::StaticConv2dPlan>) => [zero, pure, Conv2d, 35, v10, none];
            /// Complete materializing concat/gather/scatter semantic and ordered ABI.
            Movement(value: MovementValue) => [zero, pure, Movement, 31, v4, none];
            /// Captured random source semantic with an immutable stream reservation.
            Random(value: Box<crate::random::plan::RandomKernelPlan>) => [zero, pure, Random, 32, v9, none];
            /// Static inclusive prefix scan.
            PrefixScan(value: PrefixScanValue) => [zero, pure, PrefixScan, 36, v11, none];
            /// Stable CPU-static ordering with values and I32-index outputs.
            Sort(value: SortValue) => [zero, pure, Sort, 37, v16, none];
            /// Value-preserving CPU-static distribution validation boundary.
            TensorGuard(value: TensorGuardValue) => [zero, pure, TensorGuard, 38, v17, none];
            ReduceInit(value: ReductionValue) => [zero, pure, ReduceInit, 17, always, none];
            ReduceAccumulate => [two, pure, ReduceAccumulate, 18, always, none];
            ReduceFinalize => [one, pure, ReduceFinalize, 19, always, none];
            Ternary(value: super::Ternary) => [three, pure, Ternary(wire: super::Ternary), 20, always, ternary];
            Cast => [one, pure, Cast, 21, always, none];
            Bitcast => [one, pure, Bitcast, 22, always, none];
            Vectorize => [nonempty, pure, Vectorize, 23, always, none];
            Gep(value: u16) => [one, pure, Gep, 24, always, none];
            Index(value: IndexValue) => [index, pure, Index, 25, always, none];
            Load => [one, pure, Load, 26, always, none];
            Store => [two, impure, Store, 27, always, none];
            /// Immutable graph-adjacent assignment commit; never a pure kernel store.
            EffectStore(value: Box<crate::EffectPayload>) => [zero, impure, EffectStore, 33, effect, none];
            /// Orders an effect store after explicitly named predecessor effect IDs.
            After(value: Box<crate::EffectPayload>) => [one, impure, After, 34, effect, none];
            Barrier => [zero, impure, Barrier, 28, always, none];
            Sink => [any, impure, Sink, 29, always, none];
        }
    };
}

pub(crate) use uop_operation_schema;
