//! Deterministic machine-readable projection of the compatibility ledger.

use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;

pub(crate) const SOURCE_PATH: &str = "docs/COMPATIBILITY.md";
pub(crate) const MANIFEST_PATH: &str = "docs/compatibility.json";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CompatibilityManifest {
    schema_version: u32,
    source: &'static str,
    source_fingerprint: String,
    entries: Vec<CompatibilityEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CompatibilityEntry {
    section: String,
    subject_header: String,
    subject: String,
    status: CompatibilityStatus,
    fields: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CompatibilityStatus {
    Verified,
    Partial,
    NotStarted,
    Excluded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ManifestError {
    line: usize,
    reason: String,
}

impl ManifestError {
    fn new(line: usize, reason: impl Into<String>) -> Self {
        Self {
            line,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "compatibility ledger line {}: {}",
            self.line, self.reason
        )
    }
}

impl std::error::Error for ManifestError {}

pub(crate) fn parse(source: &str) -> Result<CompatibilityManifest, ManifestError> {
    let mut section = String::new();
    let mut table: Option<(Vec<String>, usize)> = None;
    let mut entries = Vec::new();

    for (line_index, raw_line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if let Some(title) = line.strip_prefix("## ") {
            section = title.trim().to_owned();
            table = None;
            continue;
        }

        if !line.starts_with('|') {
            if !line.is_empty() {
                table = None;
            }
            continue;
        }

        let cells = table_cells(line);
        if cells.is_empty() {
            continue;
        }
        if table_separator(&cells) {
            continue;
        }

        if let Some(status_index) = cells.iter().position(|cell| cell == "Status") {
            if section.is_empty() {
                return Err(ManifestError::new(
                    line_number,
                    "status table is not owned by a level-two section",
                ));
            }
            table = Some((cells, status_index));
            continue;
        }

        let Some((headers, status_index)) = &table else {
            continue;
        };
        if cells.len() != headers.len() {
            return Err(ManifestError::new(
                line_number,
                format!(
                    "table row has {} cells but its header has {}",
                    cells.len(),
                    headers.len()
                ),
            ));
        }

        let status = parse_status(&cells[*status_index]).ok_or_else(|| {
            ManifestError::new(
                line_number,
                format!("unsupported status `{}`", cells[*status_index]),
            )
        })?;
        let subject_index = if *status_index == 0 { 1 } else { 0 };
        let subject = cells
            .get(subject_index)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ManifestError::new(line_number, "status row has no subject"))?;
        let mut fields = BTreeMap::new();
        for (index, (header, value)) in headers.iter().zip(&cells).enumerate() {
            if index != *status_index && index != subject_index {
                fields.insert(header.clone(), value.clone());
            }
        }
        entries.push(CompatibilityEntry {
            section: section.clone(),
            subject_header: headers[subject_index].clone(),
            subject: subject.clone(),
            status,
            fields,
        });
    }

    if entries.is_empty() {
        return Err(ManifestError::new(0, "no compatibility entries found"));
    }

    Ok(CompatibilityManifest {
        schema_version: 1,
        source: SOURCE_PATH,
        source_fingerprint: format!("fnv1a64:{:016x}", fnv1a64(source.as_bytes())),
        entries,
    })
}

pub(crate) fn render(source: &str) -> Result<String, ManifestError> {
    let manifest = parse(source)?;
    let mut json = serde_json::to_string_pretty(&manifest)
        .map_err(|error| ManifestError::new(0, format!("JSON encoding failed: {error}")))?;
    json.push('\n');
    Ok(json)
}

fn table_cells(line: &str) -> Vec<String> {
    let inner = line
        .strip_prefix('|')
        .and_then(|line| line.strip_suffix('|'))
        .unwrap_or(line);
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut in_code = false;
    let mut escaped = false;
    for character in inner.chars() {
        if character == '`' && !escaped {
            in_code = !in_code;
            cell.push(character);
        } else if character == '|' && !in_code && !escaped {
            cells.push(cell.trim().to_owned());
            cell.clear();
        } else {
            cell.push(character);
        }
        escaped = character == '\\' && !escaped;
        if character != '\\' {
            escaped = false;
        }
    }
    cells.push(cell.trim().to_owned());
    cells
}

fn table_separator(cells: &[String]) -> bool {
    cells.iter().all(|cell| {
        let stripped = cell.trim_matches(':');
        stripped.len() >= 3 && stripped.bytes().all(|byte| byte == b'-')
    })
}

fn parse_status(cell: &str) -> Option<CompatibilityStatus> {
    let marker = cell.split_whitespace().next()?;
    match marker {
        "✅" => Some(CompatibilityStatus::Verified),
        "🚧" => Some(CompatibilityStatus::Partial),
        "⬜" => Some(CompatibilityStatus::NotStarted),
        "🚫" => Some(CompatibilityStatus::Excluded),
        _ => None,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn parses_every_supported_status_and_rejects_unknown_markers() {
        let source = "\
## Surface\n\
| Capability | Status | Acceptance requirement |\n\
|---|---:|---|\n\
| Dense | ✅ | exact |\n\
| Runtime | 🚧 | bounded |\n\
| Backend | ⬜ | pending |\n\
| Legacy | 🚫 | approved |\n";
        let manifest = parse(source).expect("valid manifest");
        assert_eq!(manifest.entries.len(), 4);
        assert_eq!(manifest.entries[0].status, CompatibilityStatus::Verified);
        assert_eq!(manifest.entries[1].status, CompatibilityStatus::Partial);
        assert_eq!(manifest.entries[2].status, CompatibilityStatus::NotStarted);
        assert_eq!(manifest.entries[3].status, CompatibilityStatus::Excluded);

        let invalid = source.replace("🚧", "◐");
        assert!(
            parse(&invalid)
                .unwrap_err()
                .to_string()
                .contains("unsupported status")
        );
    }

    #[test]
    fn checked_in_manifest_matches_the_ledger() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let source = fs::read_to_string(root.join(SOURCE_PATH)).expect("compatibility ledger");
        let expected = render(&source).expect("valid compatibility ledger");
        let actual = fs::read_to_string(root.join(MANIFEST_PATH)).expect(
            "generated compatibility manifest; run `cargo run --bin compatibility_manifest -- --write`",
        );
        assert_eq!(
            actual, expected,
            "compatibility manifest is stale; run `cargo run --bin compatibility_manifest -- --write`"
        );
    }
}
