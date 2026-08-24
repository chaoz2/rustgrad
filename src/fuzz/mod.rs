//! Deterministic semantic differential and persisted-failure tooling.
//!
//! The generator is bounded and constructor-driven: generated cases are valid
//! static RustGrad programs, never arbitrary Rust or malformed IR. Campaigns
//! compare the CPU oracle with graph-independent captured interpreter replay
//! and explicit strict-native replay. Unsupported native contracts remain
//! visible in [`FuzzCampaign`] and are never counted as matches.
//!
//! ```
//! let report = rustgrad::run_campaign(rustgrad::FuzzConfig {
//!     seed: 7,
//!     cases: 8,
//!     native: false,
//! })?;
//! assert!(report.failures.is_empty());
//! assert_eq!(report.interpreter_matches, 8);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod artifact;
mod case;
mod corpus;
mod execute;
mod fixtures;
mod generate;
mod minimize;

pub use artifact::{
    FuzzArtifactError, FuzzComparisonPolicy, FuzzFailureArtifact, FuzzOutcome, FuzzPath,
    MAX_FUZZ_ARTIFACT_FILE_BYTES,
};
pub use case::{FuzzBinaryOp, FuzzCase, FuzzReduction, FuzzTensor};
pub use corpus::{
    FuzzCorpusMode, FuzzCorpusRecord, FuzzCorpusReport, FuzzCorpusState, read_failure_artifact,
    reconcile_regression_corpus, write_failure_artifact_atomic,
};
pub use execute::{
    FuzzCampaign, FuzzComparison, FuzzConfig, FuzzReplayStatus, replay_failure, run_campaign,
    run_case,
};
pub use fixtures::regression_cases;
pub use generate::generate_case;
pub use minimize::minimize_case;

#[cfg(test)]
mod tests;
