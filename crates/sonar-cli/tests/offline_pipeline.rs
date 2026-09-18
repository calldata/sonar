//! Offline, network-free end-to-end tests for the `simulate` and `decode`
//! pipelines.
//!
//! Unlike `e2e_simulation` / `e2e_cli_output_streams` (which are `#[ignore]` and
//! require mainnet RPC), these run in CI with no network: a local account-cache
//! directory (`_meta.json` + per-account JSON) puts the CLI into offline mode,
//! so the whole parse → load → (mutate/prepare) → execute → render path is
//! exercised against fixed local state. They guard behavior parity of that
//! pipeline regardless of how the handlers are wired internally.

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use tempfile::TempDir;

/// A deterministic legacy SOL-transfer transaction (1 signature, 3 accounts:
/// fee-payer, recipient, system program), transferring 10_000_000 lamports.
const TRANSFER_TX: &str = "AYXl4tu2q/qsjwA+woUaYKC+uPuAozXJHsgxsZLux/8uXuN2z8P1tLt0wHkQImIfxXBjg3dT8ryk8D5BA6g+/QABAAEDiojj3XQJ8ZX9UtstPLpdcspnCb8dlBIb83SIAbQPb1wCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAgIAAQwCAAAAgJaYAAAAAAA=";

const PAYER: &str = "AKnL4NNf3DGWZJS6cPknBuEGnVsV4A4m5tgebLHaRSZ9";
const RECIPIENT: &str = "8qbHbw2BbbTHBW1sbeqakYXVKRQM8Ne7pLK7m6CVfeR";
const SYSTEM_PROGRAM: &str = "11111111111111111111111111111111";

/// Build a cache directory that puts the CLI into offline mode. The presence of
/// `_meta.json` is what flips the loader offline; the per-account JSON files are
/// read by the local-dir source so no RPC is needed.
fn offline_cache_dir() -> TempDir {
    let dir = TempDir::new().expect("create temp cache dir");
    let system_account = |lamports: u64| {
        format!(
            r#"{{"lamports":{lamports},"data":["","base64"],"owner":"{SYSTEM_PROGRAM}","executable":false,"rentEpoch":0}}"#
        )
    };
    std::fs::write(dir.path().join(format!("{PAYER}.json")), system_account(1_000_000_000))
        .unwrap();
    std::fs::write(dir.path().join(format!("{RECIPIENT}.json")), system_account(1)).unwrap();
    // Existence alone enables offline mode; contents are irrelevant for a raw-tx
    // input (the signature-cache path is never taken).
    std::fs::write(dir.path().join("_meta.json"), "{}").unwrap();
    dir
}

fn sonar() -> Command {
    cargo_bin_cmd!("sonar")
}

#[test]
fn simulate_offline_executes_and_reports_success() {
    let dir = offline_cache_dir();
    let assert = sonar()
        .args(["simulate", TRANSFER_TX, "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("SUCCESS"), "expected a success banner, got:\n{stdout}");
    assert!(stdout.contains(SYSTEM_PROGRAM), "expected the executed program in output:\n{stdout}");
}

#[test]
fn simulate_offline_json_is_structured_and_successful() {
    let dir = offline_cache_dir();
    let assert = sonar()
        .args(["simulate", TRANSFER_TX, "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1", "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("simulate --json emits JSON");
    assert!(json.get("transaction").is_some(), "missing transaction section: {json}");
    let simulation = json.get("simulation").expect("missing simulation section");
    assert!(simulation.get("status").is_some(), "missing simulation.status: {simulation}");
    assert!(
        simulation.get("compute_units_consumed").is_some(),
        "missing simulation.compute_units_consumed: {simulation}"
    );
}

#[test]
fn simulate_offline_bundle_executes_all() {
    let dir = offline_cache_dir();
    let assert = sonar()
        .args(["simulate", TRANSFER_TX, TRANSFER_TX, "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("Bundle"), "expected a bundle banner, got:\n{stdout}");
    assert!(stdout.contains("2/2"), "expected both bundle txs to run, got:\n{stdout}");
}

#[test]
fn decode_offline_renders_decoded_transfer() {
    let dir = offline_cache_dir();
    let assert = sonar()
        .args(["decode", TRANSFER_TX, "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("Decoded Instructions"), "expected decode header:\n{stdout}");
    assert!(stdout.contains("Transfer"), "system transfer should be decoded:\n{stdout}");
    assert!(stdout.contains(PAYER), "payer should appear in account list:\n{stdout}");
    assert!(stdout.contains(RECIPIENT), "recipient should appear in account list:\n{stdout}");
}

// ---------------------------------------------------------------------------
// Transaction v1 (SIMD-0385)
// ---------------------------------------------------------------------------

/// A v1 transaction produced by the reference implementation
/// (`solana-message`/`solana-transaction` 5.0 via `wincode`): one signature, a
/// system transfer of 1_000_000 lamports, and a header config requesting a
/// 5_000-lamport priority fee, a 200_000 compute unit limit, a 64 MiB loaded
/// accounts data size limit, and a 32 KiB heap.
///
/// The loaded accounts data size request matters: an unset bit 3 means a
/// zero-byte budget (SIMD-0385), which no transaction can execute within.
fn v1_transfer_tx() -> &'static str {
    // The reference-implementation fixture, shared by every crate that
    // exercises the v1 wire format (the file keeps a trailing newline).
    include_str!("../../sonar-sim/tests/fixtures/v1_transfer.b64").trim()
}

const V1_PAYER: &str = "GmaDrppBC7P5ARKV8g3djiwP89vz1jLK23V2GBjuAEGB";
const V1_RECIPIENT: &str = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin";

/// [`v1_transfer_tx`] with the last byte of its signature array flipped.
///
/// Derived instead of copied: an earlier copy was a different transaction
/// entirely (236 bytes, mask `0x17`), so the two would have drifted apart.
fn v1_tampered_signature_tx() -> String {
    let mut bytes = BASE64_STANDARD.decode(v1_transfer_tx()).expect("fixture is base64");
    let last = bytes.last_mut().expect("fixture is not empty");
    *last ^= 0x0f;
    BASE64_STANDARD.encode(bytes)
}

/// [`v1_transfer_tx`] with its compute-unit-limit request removed (mask `0x1b`).
///
/// The SIMD says an absent request is a zero limit and that `ComputeBudget`
/// instructions do not configure v1, so nothing in this transaction can stand in
/// for the missing request.
fn v1_tx_without_a_compute_unit_limit() -> String {
    let mut bytes = BASE64_STANDARD.decode(v1_transfer_tx()).expect("fixture is base64");

    let mask = u32::from_le_bytes(bytes[4..8].try_into().expect("mask is four bytes")) & !0b100;
    bytes[4..8].copy_from_slice(&mask.to_le_bytes());

    // Config values follow the address table in mask-bit order, so the four bytes
    // that bit 2 owned sit right after the eight-byte priority fee.
    let num_addresses = usize::from(bytes[41]);
    let config_start = 42 + num_addresses * 32;
    bytes.drain(config_start + 8..config_start + 12);

    BASE64_STANDARD.encode(bytes)
}

/// Offline account cache for the v1 fixture's payer and recipient.
fn v1_offline_cache_dir() -> TempDir {
    let dir = TempDir::new().expect("create temp cache dir");
    let system_account = |lamports: u64| {
        format!(
            r#"{{"lamports":{lamports},"data":["","base64"],"owner":"{SYSTEM_PROGRAM}","executable":false,"rentEpoch":0}}"#
        )
    };
    std::fs::write(dir.path().join(format!("{V1_PAYER}.json")), system_account(1_000_000_000))
        .unwrap();
    std::fs::write(dir.path().join(format!("{V1_RECIPIENT}.json")), system_account(1)).unwrap();
    std::fs::write(dir.path().join("_meta.json"), "{}").unwrap();
    dir
}

#[test]
fn simulate_v1_offline_executes_the_transaction() {
    let dir = v1_offline_cache_dir();
    let assert = sonar()
        .args(["simulate", v1_transfer_tx(), "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("SUCCESS"), "expected a success banner, got:\n{stdout}");
    // The v1 config is surfaced next to the summary for v1 transactions.
    assert!(
        stdout.contains("v1 config: priority fee: 5,000 lamports"),
        "expected v1 config:\n{stdout}"
    );
    // The "CU used / limit" denominator is the header's request, not the legacy
    // 200,000 default and not a `ComputeBudget` instruction.
    assert!(stdout.contains("/ 200,000 ("), "expected the header limit:\n{stdout}");
}

#[test]
fn bundle_mutations_report_the_post_mutation_size() {
    // A bundle reaches the mutation path through `parse_bundle` rather than the
    // single-transaction one. Both have to rebuild the v1 view, or the report
    // shows the mutated instruction list next to the pre-mutation size.
    let dir = v1_offline_cache_dir();
    let tx = v1_transfer_tx();
    let assert = sonar()
        .args(["simulate", tx, tx, "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1", "--json"])
        .args(["--insert-ix", "1=program=MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr data=0x6869"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("json output");

    // 240 bytes for the fixture, plus the memo program's address in the address
    // table (32), the instruction header (4) and its data (2).
    for transaction in json["transactions"].as_array().expect("transaction list") {
        assert_eq!(transaction["transaction"]["size_bytes"], 240 + 32 + 4 + 2, "{stdout}");
    }
}

#[test]
fn simulate_v1_without_a_compute_unit_limit_reports_a_zero_limit() {
    let dir = v1_offline_cache_dir();
    let assert = sonar()
        .args(["simulate", &v1_tx_without_a_compute_unit_limit(), "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1"])
        .assert();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();

    // Zero compute units cannot execute anything, so the run fails — and the
    // denominator has to say 0 rather than claiming the legacy default.
    assert!(stdout.contains("/ 0 ("), "expected a zero limit:\n{stdout}");
    assert!(!stdout.contains("/ 200,000 ("), "denominator fell back:\n{stdout}");
}

#[test]
fn simulate_v1_offline_json_reports_v1_format_and_config() {
    let dir = v1_offline_cache_dir();
    let assert = sonar()
        .args(["simulate", v1_transfer_tx(), "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1", "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("simulate --json emits JSON");

    let transaction = json.get("transaction").expect("missing transaction section");
    assert_eq!(transaction["version"], "v1");
    assert_eq!(transaction["size_bytes"], 240);
    assert_eq!(transaction["v1_config"]["config_mask"], "0x0000001f");
    assert_eq!(transaction["v1_config"]["priority_fee"], 5_000);
    assert_eq!(transaction["v1_config"]["compute_unit_limit"], 200_000);
    assert_eq!(transaction["v1_config"]["heap_size"], 32_768);
    assert_eq!(transaction["v1_config"]["effective_heap_size"], 32_768);
    // No address lookup tables exist in v1.
    assert_eq!(transaction["lookups"].as_array().map(Vec::len), Some(0));

    let simulation = json.get("simulation").expect("missing simulation section");
    assert!(simulation.get("status").is_some(), "missing simulation.status: {simulation}");
}

#[test]
fn decode_v1_offline_renders_transfer_and_config() {
    let dir = v1_offline_cache_dir();
    let assert = sonar()
        .args(["decode", v1_transfer_tx(), "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("Decoded Instructions"), "expected decode header:\n{stdout}");
    assert!(stdout.contains("Transfer"), "system transfer should be decoded:\n{stdout}");
    assert!(stdout.contains(V1_PAYER), "payer should appear in account list:\n{stdout}");
    assert!(stdout.contains(V1_RECIPIENT), "recipient should appear in account list:\n{stdout}");
    assert!(stdout.contains("Transaction Config (v1)"), "expected v1 config block:\n{stdout}");
}

#[test]
fn decode_v1_offline_json_reports_240_byte_v1_transaction() {
    let dir = v1_offline_cache_dir();
    let assert = sonar()
        .args(["decode", v1_transfer_tx(), "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1", "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("decode --json emits JSON");
    // `decode --json` emits the transaction section at the top level.
    assert_eq!(json["version"], "v1");
    assert_eq!(json["size_bytes"], 240);
    assert_eq!(json["v1_config"]["config_mask"], "0x0000001f");
}

#[test]
fn simulate_v1_offline_rejects_signature_check_against_v1_payload() {
    // `--check-sig` verifies v1 signatures against the v1 payload. A tampered
    // signature must fail even though the VM-level check is skipped for v1.
    let tampered = v1_tampered_signature_tx();

    let dir = v1_offline_cache_dir();
    let assert = sonar()
        .args(["simulate", &tampered, "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1", "--check-sig"])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(stderr.contains("signature 0 is invalid"), "expected a v1 signature error:\n{stderr}");
}

#[test]
fn decode_v1_offline_accepts_base58_input() {
    // The same transaction in base58 must parse identically: v1 detection
    // happens on the decoded bytes, not on the input encoding. (The reported
    // `encoding` is the canonical re-encoding the pipeline parses, as for
    // legacy/v0 inputs.)
    const V1_TRANSFER_TX_BASE58: &str = "BwVkU2eX13rdzY1PNsXLV5w5S7fHEMdMQviQrpJhac1YzWXeJnpEgWHpUQB3FZTysRLa2Zc9MjwfHaLSoGJmDGpLSDwi6r1DHtva7eLWB6pojvx4MNkmX4P7ot2KebpP1DnypRxb6CaRNkXY3YaChGzPeGTat9sZcCNRVoygUoHRubp8JmsfZvFACJbWPKsyyN2CG87S3xharHHkXywa3yuBbgBXshNYDUpie5pzVL8Wu8tdKeKGxghs7NMKaaHPSZKNmxJrMvwiaBBbL7AK3k3QvikxYfiJwSxYb7sXqhvz9TKwDhHAF4FNtRQ3qCHFLp6Ycqa7";

    let dir = v1_offline_cache_dir();
    let assert = sonar()
        .args(["decode", V1_TRANSFER_TX_BASE58, "--cache-dir"])
        .arg(dir.path())
        .args(["--rpc-url", "http://localhost:1", "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("decode --json emits JSON");
    assert_eq!(json["version"], "v1");
    assert_eq!(json["size_bytes"], 240);
    assert_eq!(json["v1_config"]["priority_fee"], 5_000);
}
