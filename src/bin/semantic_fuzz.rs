use rustgrad::fuzz::{
    FuzzCorpusMode, FuzzCorpusState, FuzzReplayStatus, read_failure_artifact,
    reconcile_regression_corpus, write_failure_artifact_atomic,
};
use rustgrad::{FuzzConfig, replay_failure, run_campaign};
use std::{env, path::PathBuf, process::ExitCode};

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
            let native = match args.next().as_deref() {
                None => true,
                Some("interpreter-only") => false,
                Some(token) => {
                    return Err(format!(
                        "unknown run mode {token}; usage: semantic_fuzz run [seed] [cases] [interpreter-only]"
                    ));
                }
            };
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
                write_failure_artifact_atomic(PathBuf::from(".").as_path(), failure)?;
                println!("failure=failure-{:016x}.rgfz", failure.identity);
            }
            Ok(report.failures.is_empty())
        }
        Some("replay") => {
            let path = args
                .next()
                .ok_or("usage: semantic_fuzz replay <artifact>")?;
            if path.starts_with('-') || args.next().is_some() {
                return Err("usage: semantic_fuzz replay <artifact>".into());
            }
            let artifact = read_failure_artifact(PathBuf::from(path).as_path())?;
            let status = replay_failure(&artifact)?;
            println!(
                "identity={:016x} status={}",
                artifact.identity,
                replay_status_name(&status)
            );
            Ok(status == FuzzReplayStatus::Reproduced)
        }
        Some("corpus") => {
            let paths = args.collect::<Vec<_>>();
            if paths.is_empty() || paths.iter().any(|path| path.starts_with('-')) {
                return Err("usage: semantic_fuzz corpus <artifact>...".into());
            }
            let mut reproduced = 0usize;
            let mut resolved = 0usize;
            let mut changed = 0usize;
            let mut unsupported = 0usize;
            for path in &paths {
                let artifact = read_failure_artifact(PathBuf::from(path).as_path())?;
                match replay_failure(&artifact)? {
                    FuzzReplayStatus::Reproduced => reproduced += 1,
                    FuzzReplayStatus::Resolved => resolved += 1,
                    FuzzReplayStatus::Changed => changed += 1,
                    FuzzReplayStatus::Unsupported { .. } => unsupported += 1,
                }
            }
            println!(
                "artifacts={} reproduced={reproduced} resolved={resolved} changed={changed} unsupported={unsupported}",
                paths.len()
            );
            Ok(reproduced == paths.len())
        }
        Some("regressions") => {
            let mut directory = None;
            let mut write = false;
            let mut prune = false;
            for token in args {
                match token.as_str() {
                    "--write" if !write => write = true,
                    "--prune-resolved" if !prune => prune = true,
                    _ if token.starts_with('-') || directory.is_some() => {
                        return Err("usage: semantic_fuzz regressions [directory] [--write] [--prune-resolved]".into());
                    }
                    _ => directory = Some(PathBuf::from(token)),
                }
            }
            if prune && !write {
                return Err("--prune-resolved requires --write".into());
            }
            let mode = if prune {
                FuzzCorpusMode::WriteAndPruneResolved
            } else if write {
                FuzzCorpusMode::Write
            } else {
                FuzzCorpusMode::Check
            };
            let directory = directory.unwrap_or_else(|| PathBuf::from("."));
            let report = reconcile_regression_corpus(&directory, mode)?;
            println!(
                "regressions={} inventoried={} current_failures={} unresolved={} reproduced={} new={} changed={} resolved={} unsupported={} written={} pruned={}",
                report.regressions,
                report.inventoried,
                report.current_failures,
                report.unresolved,
                report.reproduced,
                report.new,
                report.changed,
                report.resolved,
                report.unsupported,
                report.written,
                report.pruned,
            );
            for record in &report.records {
                let state = match record.state {
                    FuzzCorpusState::Reproduced => "reproduced",
                    FuzzCorpusState::New => "new",
                    FuzzCorpusState::Changed => "changed",
                    FuzzCorpusState::Resolved => "resolved",
                    FuzzCorpusState::Unsupported => "unsupported",
                };
                if let Some(previous) = record.previous_identity {
                    println!(
                        "identity={:016x} previous_identity={previous:016x} state={state}",
                        record.identity
                    );
                } else {
                    println!("identity={:016x} state={state}", record.identity);
                }
            }
            Ok(report.is_clean())
        }
        Some(_) => Err("usage: semantic_fuzz <run|replay|corpus|regressions>".into()),
    }
}

fn replay_status_name(status: &FuzzReplayStatus) -> &'static str {
    match status {
        FuzzReplayStatus::Reproduced => "reproduced",
        FuzzReplayStatus::Resolved => "resolved",
        FuzzReplayStatus::Changed => "changed",
        FuzzReplayStatus::Unsupported { .. } => "unsupported",
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

#[cfg(test)]
mod tests {
    use super::run;

    fn arguments(values: &[&str]) -> impl Iterator<Item = String> {
        values.iter().map(|value| (*value).to_string())
    }

    #[test]
    fn rejects_unknown_modes_and_tokens() {
        assert!(run(arguments(&["unknown"])).is_err());
        assert!(run(arguments(&["run", "7", "1", "typo"])).is_err());
        assert!(run(arguments(&["run", "7", "1", "interpreter-only", "typo"])).is_err());
        assert!(run(arguments(&["corpus", "--typo"])).is_err());
        assert!(run(arguments(&["replay", "--typo"])).is_err());
        assert!(run(arguments(&["regressions", "--typo"])).is_err());
        assert!(run(arguments(&["regressions", "--write", "--write"])).is_err());
        assert!(run(arguments(&["regressions", "--prune-resolved"])).is_err());
    }
}
