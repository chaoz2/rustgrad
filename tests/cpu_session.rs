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
    assert!(matches!(
        first.execution_summary(&second_value, false),
        Err(Error::SessionHandleMismatch { .. })
    ));
}

#[test]
fn session_execution_summary_is_static_deterministic_and_non_mutating() {
    let mut session = CpuSession::new();
    let input = session.variable([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
    let squared = session.mul(&input, &input).unwrap();
    let output = session.sum_all(&squared).unwrap();
    let before = session.realize(&output).unwrap();
    let disabled = session.execution_summary(&output, false).unwrap();
    let enabled = session.execution_summary(&output, true).unwrap();
    assert_eq!(disabled, session.execution_summary(&output, false).unwrap());
    assert_ne!(disabled.identity, enabled.identity);
    assert_eq!(disabled.requested_outputs.len(), 1);
    assert_eq!(disabled.requested_outputs[0].shape, Shape::from([]));
    assert_eq!(before, session.realize(&output).unwrap());

    let empty = session.variable([0, 2], []).unwrap();
    let empty_output = session.relu(&empty).unwrap();
    let empty_summary = session.execution_summary(&empty_output, true).unwrap();
    assert_eq!(empty_summary.requested_outputs[0].bytes, 0);
    assert_eq!(empty_summary.zero_domain_item_count, 1);
}

#[test]
fn session_phase_b_classifier_and_static_movement_delegate_to_graph() {
    let mut session = CpuSession::new();
    let input = session.variable([2, 2], [1.0, 2.0, -1.0, 1.0]).unwrap();
    let weights = session
        .tensor([2, 3], [1.0, 0.0, -1.0, 0.0, 1.0, 1.0])
        .unwrap();
    let zero = session.tensor([1], [0.0]).unwrap();
    let one = session.tensor([1], [1.0]).unwrap();
    let logits = session.matmul(&input, &weights).unwrap();
    let shifted = session.sub(&logits, &zero).unwrap();
    let scaled = session.div(&shifted, &one).unwrap();
    let activated = session.relu(&scaled).unwrap();
    let probabilities = session.softmax(&activated, -1).unwrap();
    let classes = session.argmax(&probabilities, -1).unwrap();
    let loss = session.sum_all(&activated).unwrap();
    let input_gradient = session.grad(&loss, &input).unwrap();

    assert_eq!(activated.shape(), &Shape::from([2, 3]));
    assert_eq!(activated.dtype(), DType::F32);
    assert_eq!(
        session.realize(&classes).unwrap().to_vec_f64(),
        vec![1.0, 2.0]
    );
    let probability_values = session.realize(&probabilities).unwrap().to_vec_f64();
    assert!((probability_values[0] - 0.211_941_56).abs() < 1e-6);
    assert!((probability_values[1] - 0.576_116_88).abs() < 1e-6);
    assert_eq!(
        session.realize(&input_gradient).unwrap().to_vec_f64(),
        vec![0.0, 2.0, -1.0, 2.0]
    );
    assert!(
        session
            .trace(&probabilities)
            .unwrap()
            .to_string()
            .contains("exp")
    );

    let transposed = session.transpose(&activated, 0, 1).unwrap();
    let permuted = session.permute(&activated, [1, 0]).unwrap();
    assert_eq!(
        session.realize(&transposed).unwrap(),
        session.realize(&permuted).unwrap()
    );
    let shrunk = session.shrink(&activated, [(0, 2), (1, 3)]).unwrap();
    assert_eq!(
        session.realize(&shrunk).unwrap().to_vec_f64(),
        vec![2.0, 1.0, 1.0, 2.0]
    );
    let sliced = session
        .slice(
            &activated,
            [
                rustgrad::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                rustgrad::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
            ],
        )
        .unwrap();
    assert_eq!(
        session.realize(&sliced).unwrap().to_vec_f64(),
        vec![0.0, 1.0, 2.0, 1.0, 2.0, 1.0]
    );
    let concatenated = session.concat(&[&shrunk, &shrunk], 0).unwrap();
    assert_eq!(concatenated.shape(), &Shape::from([4, 2]));

    let index = session
        .tensor_with_dtype(
            [2, 2],
            DType::I32,
            [
                rustgrad::Scalar::I(2),
                rustgrad::Scalar::I(0),
                rustgrad::Scalar::I(1),
                rustgrad::Scalar::I(1),
            ],
        )
        .unwrap();
    let gathered = session.gather(&activated, &index, 1).unwrap();
    assert_eq!(
        session.realize(&gathered).unwrap().to_vec_f64(),
        vec![1.0, 1.0, 1.0, 1.0]
    );

    let scalar = session.tensor([], [2.0]).unwrap();
    let scalar_ratio = session.div(&scalar, &scalar).unwrap();
    assert_eq!(
        session.realize(&scalar_ratio).unwrap().to_vec_f64(),
        vec![1.0]
    );
    let empty = session.tensor([0, 2], []).unwrap();
    let empty_relu = session.relu(&empty).unwrap();
    assert!(session.realize(&empty_relu).unwrap().is_empty());
}

#[test]
fn session_phase_b_validates_axes_shapes_indices_and_cross_session_operands() {
    let mut session = CpuSession::new();
    let value = session.tensor([2, 2], [1.0; 4]).unwrap();
    assert!(matches!(
        session.softmax(&value, 2),
        Err(Error::InvalidReductionAxes { .. })
    ));
    assert!(matches!(
        session.transpose(&value, 0, 2),
        Err(Error::InvalidAxis { .. })
    ));
    assert!(matches!(
        session.shrink(&value, [(0, 3), (0, 1)]),
        Err(Error::InvalidBounds { .. })
    ));
    let bad_index = session
        .tensor_with_dtype([2, 2], DType::Bool, [rustgrad::Scalar::Bool(false); 4])
        .unwrap();
    assert!(matches!(
        session.gather(&value, &bad_index, 1),
        Err(Error::InvalidIndexDType { .. })
    ));
    let different = session.tensor([3, 2], [1.0; 6]).unwrap();
    assert!(matches!(
        session.concat(&[&value, &different], 1),
        Err(Error::InvalidConcat { .. })
    ));

    let mut other_session = CpuSession::new();
    let foreign = other_session.tensor([2, 2], [1.0; 4]).unwrap();
    assert!(matches!(
        session.concat(&[&value, &foreign], 0),
        Err(Error::SessionHandleMismatch { .. })
    ));
    assert!(matches!(
        session.gather(&value, &foreign, 0),
        Err(Error::SessionHandleMismatch { .. })
    ));
}
