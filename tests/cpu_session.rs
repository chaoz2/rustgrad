use rustgrad::{CpuSession, DType, Error, SessionDevice, Shape, TensorData};

#[test]
fn documented_cpu_session_workflow_is_typed_traceable_and_rebindable() {
    let mut session = CpuSession::on(SessionDevice::Cpu).unwrap();
    let input = session.variable([2, 1], [1.0, 2.0]).unwrap();
    let scale = session.tensor([3], [10.0, 20.0, 30.0]).unwrap();
    let bias = session.tensor([3], [1.0, 1.0, 1.0]).unwrap();
    let product = session.mul(&input, &scale).unwrap();
    let output = session.add(&product, &bias).unwrap();
    let flattened = session.reshape(&output, [6]).unwrap();
    let loss = session.sum_all(&output).unwrap();
    let gradient = session.grad(&loss, &input).unwrap();

    let result = session.realize(&output).unwrap();
    assert_eq!(result.shape(), &Shape::from([2, 3]));
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(
        result.to_vec_f64(),
        vec![11.0, 21.0, 31.0, 21.0, 41.0, 61.0]
    );
    assert_eq!(
        session.realize(&flattened).unwrap().shape(),
        &Shape::from([6])
    );
    assert_eq!(session.realize(&loss).unwrap().to_vec_f64(), vec![186.0]);
    assert_eq!(
        session.realize(&gradient).unwrap().to_vec_f64(),
        vec![60.0, 60.0]
    );
    assert!(session.trace(&output).unwrap().to_string().contains("mul"));
    assert!(session.graph().node_count() >= 7);

    session
        .set(&input, TensorData::new([2, 1], vec![3.0, 4.0]).unwrap())
        .unwrap();
    assert_eq!(session.realize(&loss).unwrap().to_vec_f64(), vec![426.0]);
}

#[test]
fn session_rejects_cross_handles_unsupported_devices_and_bad_rebindings() {
    let mut first = CpuSession::new();
    let first_value = first.variable([1], [1.0]).unwrap();
    let mut second = CpuSession::new();
    let second_value = second.tensor([1], [2.0]).unwrap();

    assert!(matches!(
        first.add(&first_value, &second_value),
        Err(Error::SessionHandleMismatch { .. })
    ));
    assert!(matches!(
        CpuSession::on(SessionDevice::Cuda),
        Err(Error::UnsupportedSessionDevice { device: "cuda" })
    ));
    assert!(matches!(
        first.set(&first_value, TensorData::new([2], vec![1.0, 2.0]).unwrap()),
        Err(Error::InputShape { .. })
    ));
    assert!(matches!(
        first.set(
            &first_value,
            TensorData::from_scalars([1], DType::I32, [rustgrad::Scalar::I(1)]).unwrap(),
        ),
        Err(Error::InputDType { .. })
    ));
}
