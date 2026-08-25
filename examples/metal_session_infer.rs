//! Strict static elementwise CpuSession inference on a caller-owned Metal device.

use rustgrad::runtime::metal::{MetalDiscovery, MetalRenderer, MetalRuntime};
use rustgrad::{CpuSession, TensorData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = MetalRuntime::load()?;
    let MetalDiscovery::Devices(mut devices) = runtime.discover()? else {
        return Err("Metal framework loaded but this process sees no device".into());
    };
    let device = devices.remove(0);
    let renderer = MetalRenderer::new(64, device.info().capabilities.clone())?;

    let mut session = CpuSession::new();
    let input = session.variable([2, 2], [1.0, 2.0, 3.0, 4.0])?;
    let bias = session.constant(TensorData::new([2], vec![10.0, 20.0])?)?;
    let output = session.add(&input, &bias)?;

    let cpu = session.realize(&output)?;
    let first = session.realize_metal(&output, device.clone(), renderer.clone())?;
    let second = session.realize_metal(&output, device.clone(), renderer)?;
    assert_eq!(cpu, *first.output());
    assert_eq!(first.trace, second.trace);
    println!(
        "strict Metal parity: cache_entries={}, keys={:?}, trace={:016x}",
        device.cache().len(),
        first.cache_keys,
        first.trace.logical_identity,
    );
    Ok(())
}
