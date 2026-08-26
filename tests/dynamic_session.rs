use rustgrad::{CpuSession, DType, Error, Scalar, Shape, TensorData};

#[test]
fn dynamic_session_mask_pipeline_uses_exact_runtime_cardinality() {
    let mut session = CpuSession::new();
    let input = session
        .variable([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let mask = session
        .tensor_with_dtype(
            [1, 3],
            DType::Bool,
            [Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)],
        )
        .unwrap();
    let scalar = session.tensor([], [0.5]).unwrap();
    let selected = session.masked_select_dynamic(&input, &mask).unwrap();
    let negated = session.dynamic_neg(&selected).unwrap();
    let shifted = session.dynamic_add_scalar(&negated, &scalar).unwrap();
    let sum = session.dynamic_sum(&shifted).unwrap();
    let mean = session.dynamic_mean(&shifted).unwrap();

    assert_eq!(selected.dtype(), DType::F32);
    assert_eq!(
        session.realize_dynamic(&selected).unwrap(),
        TensorData::new([4], vec![1.0, 3.0, 4.0, 6.0]).unwrap()
    );
    assert_eq!(
        session.realize_dynamic(&shifted).unwrap(),
        TensorData::new([4], vec![-0.5, -2.5, -3.5, -5.5]).unwrap()
    );
    assert_eq!(
        session.realize_dynamic(&sum).unwrap().shape(),
        &Shape::from([])
    );
    assert_eq!(
        session.realize_dynamic(&sum).unwrap().to_vec_f64(),
        vec![-12.0]
    );
    assert_eq!(
        session.realize_dynamic(&mean).unwrap().to_vec_f64(),
        vec![-3.0]
    );

    session
        .set(
            &input,
            TensorData::new([2, 3], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap(),
        )
        .unwrap();
    assert_eq!(
        session.realize_dynamic(&shifted).unwrap().to_vec_f64(),
        vec![-1.5, -5.5, -7.5, -11.5]
    );
}

#[test]
fn dynamic_session_rejects_cross_handles_and_out_of_boundary_inputs_before_growth() {
    let mut first = CpuSession::new();
    let input = first.variable([2], [1.0, 2.0]).unwrap();
    let mask = first
        .tensor_with_dtype([2], DType::Bool, [Scalar::Bool(true), Scalar::Bool(false)])
        .unwrap();
    let selected = first.masked_select_dynamic(&input, &mask).unwrap();
    let nodes = first.graph().node_count();
    let integer = first
        .tensor_with_dtype([2], DType::I32, [Scalar::I(1), Scalar::I(2)])
        .unwrap();
    assert!(matches!(
        first.masked_select_dynamic(&integer, &mask),
        Err(Error::InvalidElementwiseDType { .. })
    ));
    assert_eq!(first.graph().node_count(), nodes + 1);

    let nonscalar = first.tensor([2], [1.0, 2.0]).unwrap();
    assert!(matches!(
        first.dynamic_mul_scalar(&selected, &nonscalar),
        Err(Error::InvalidIndex)
    ));

    let mut second = CpuSession::new();
    let second_input = second.variable([1], [1.0]).unwrap();
    assert!(matches!(
        first.dynamic_add_scalar(&selected, &second_input),
        Err(Error::SessionHandleMismatch { .. })
    ));
    assert!(matches!(
        second.realize_dynamic(&selected),
        Err(Error::SessionHandleMismatch { .. })
    ));
}

#[test]
fn dynamic_session_empty_pipeline_preserves_exact_empty_and_scalar_identities() {
    let mut session = CpuSession::new();
    let input = session.variable([0, 2], []).unwrap();
    let mask = session
        .tensor_with_dtype(
            [1, 2],
            DType::Bool,
            [Scalar::Bool(true), Scalar::Bool(true)],
        )
        .unwrap();
    let scalar = session.tensor([], [2.0]).unwrap();
    let selected = session.masked_select_dynamic(&input, &mask).unwrap();
    let squared = session.dynamic_square(&selected).unwrap();
    let scaled = session.dynamic_mul_scalar(&squared, &scalar).unwrap();
    let sum = session.dynamic_sum(&scaled).unwrap();
    let mean = session.dynamic_mean(&scaled).unwrap();

    assert_eq!(
        session.realize_dynamic(&scaled).unwrap().shape(),
        &Shape::from([0])
    );
    assert_eq!(
        session.realize_dynamic(&sum).unwrap().to_vec_f64(),
        vec![0.0]
    );
    assert!(session.realize_dynamic(&mean).unwrap().to_vec_f64()[0].is_nan());
}
