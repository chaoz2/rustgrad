use rustgrad::{
    FuzzComparison, FuzzConfig, FuzzFailureArtifact, regression_cases, replay_failure,
    run_campaign, run_case,
};
use std::{env, fs, path::PathBuf, process::ExitCode};

fn parse_u64(value: Option<String>, default: u64) -> Result<u64, String> {
    value.map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|_| format!("invalid integer {value}"))
    })
}

fn run(mut args: impl Iterator<Item = String>) -> Result<bool, String> {
    match args.next().as_deref() {
        Some("run") | None => {
            let seed = parse_u64(args.next(), 0)?;
            let cases = parse_u64(args.next(), 64)?;
            let native = args.next().as_deref() != Some("interpreter-only");
            if args.next().is_some() {
                return Err("usage: semantic_fuzz run [seed] [cases] [interpreter-only]".into());
            }
            let report = run_campaign(FuzzConfig {
                seed,
                cases,
                native,
            })?;
            println!(
                "seed={} generated={} interpreter_matches={} native_matches={} native_unsupported={} failures={}",
                report.seed,
                report.generated,
                report.interpreter_matches,
                report.native_matches,
                report.native_unsupported,
                report.failures.len()
            );
            for failure in &report.failures {
                let path = format!("rustgrad-failure-{:016x}.rgfz", failure.identity);
                fs::write(
                    &path,
                    failure.to_bytes().map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
                println!("failure={path}");
            }
            Ok(report.failures.is_empty())
        }
        Some("replay") => {
            let path = args
                .next()
                .ok_or("usage: semantic_fuzz replay <artifact>")?;
            if args.next().is_some() {
                return Err("usage: semantic_fuzz replay <artifact>".into());
            }
            let artifact = FuzzFailureArtifact::from_bytes(
                &fs::read(path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            let reproduced = replay_failure(&artifact)?;
            println!(
                "identity={:016x} reproduced={reproduced}",
                artifact.identity
            );
            Ok(reproduced)
        }
        Some("corpus") => {
            let paths = args.collect::<Vec<_>>();
            if paths.is_empty() {
                return Err("usage: semantic_fuzz corpus <artifact>...".into());
            }
            let mut reproduced = 0usize;
            for path in &paths {
                let artifact = FuzzFailureArtifact::from_bytes(
                    &fs::read(path).map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
                if replay_failure(&artifact)? {
                    reproduced += 1;
                }
            }
            println!("artifacts={} reproduced={reproduced}", paths.len());
            Ok(reproduced == paths.len())
        }
        Some("regressions") => {
            let directory = PathBuf::from(args.next().unwrap_or_else(|| ".".into()));
            if args.next().is_some() {
                return Err("usage: semantic_fuzz regressions [directory]".into());
            }
            fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
            let mut failures = 0usize;
            for (index, case) in regression_cases().iter().enumerate() {
                for comparison in run_case(0xfeed, index as u64, case, false)? {
                    if let FuzzComparison::Failure(failure) = comparison {
                        let path =
                            directory.join(format!("failure-{:016x}.rgfz", failure.identity));
                        fs::write(
                            &path,
                            failure.to_bytes().map_err(|error| error.to_string())?,
                        )
                        .map_err(|error| error.to_string())?;
                        println!("failure={}", path.display());
                        failures += 1;
                    }
                }
            }
            println!(
                "regressions={} failures={failures}",
                regression_cases().len()
            );
            Ok(true)
        }
        Some(_) => Err("usage: semantic_fuzz <run|replay|corpus|regressions>".into()),
    }
}

fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("semantic_fuzz: {error}");
            ExitCode::from(2)
        }
    }
}
