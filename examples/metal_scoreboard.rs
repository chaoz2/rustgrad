//! Record exact host-observed facts for one persistent Metal inference session.

use rustgrad::nn::Linear;
use rustgrad::runtime::metal::{
    MetalInferencePlan, MetalRuntime, MetalScoreboardContext, MetalSessionScoreboard,
};
use rustgrad::{CapturedInference, DType, Graph, TensorData};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = MetalRuntime::load()?.device(0)?;
    let model = Linear::new_static(4, 2, true, 7)?;
    let mut graph = Graph::new();
    let features = graph.input_dtype("features", [1, 4], DType::F32);
    let scores = model.forward(&mut graph, features)?;
    let captured = CapturedInference::from_module_graph(&model, &graph, &[scores])?;
    let plan = MetalInferencePlan::new(captured, device.renderer(64)?)?;
    println!("capture: {:?}", plan.execution_plan());
    println!("Metal plan: {:?}", plan.summary());

    let mut scoreboard = MetalSessionScoreboard::new(
        MetalScoreboardContext::new(
            "linear-1x4",
            env!("CARGO_PKG_VERSION"),
            "live Metal device; host wall clock and host API counters",
        )?,
        &plan,
    );
    let mut session = plan.prepare(device)?;
    scoreboard.bind(&session)?;
    let features = TensorData::new([1, 4], vec![1.0, 2.0, 3.0, 4.0])?;
    for _ in 0..3 {
        let run = session.run(&BTreeMap::from([("features".into(), features.clone())]))?;
        scoreboard.record(&run)?;
    }
    let report = scoreboard.report()?;
    report.write_json("metal-scoreboard.json")?;
    println!("scoreboard: {report:?}");
    Ok(())
}
