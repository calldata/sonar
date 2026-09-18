//! SIMD-0385 "Transaction V1" message format.
//!
//! A v1 transaction drops address lookup tables and compute budget
//! *instructions* in favor of a self-describing binary layout with an explicit
//! configuration mask. On the wire it is:
//!
//! ```text
//! VersionByte (u8 = 129)
//! LegacyHeader (num_required_signatures, num_readonly_signed_accounts,
//!               num_readonly_unsigned_accounts)         -- 3 bytes, no padding
//! TransactionConfigMask (u32, little-endian)
//! LifetimeSpecifier ([u8; 32])
//! NumInstructions (u8)
//! NumAddresses (u8)
//! Addresses ([[u8; 32]; NumAddresses])
//! ConfigValues ([[u8; 4]; popcount of the mask, in bit order)
//! InstructionHeaders ([(u8, u8, u16); NumInstructions]) -- 4 bytes each, no padding
//! InstructionPayloads (account indices then data, per instruction, in order)
//! Signatures ([[u8; 64]; num_required_signatures])      -- no length prefix
//! ```
//!
//! Unlike legacy/v0, the signature array is a fixed-length trailing array whose
//! length comes from the message header, and trailing bytes are forbidden.
//!
//! Signatures cover the serialization of everything *before* the signature
//! array, which is exactly [`V1Transaction::signed_payload`].
//!
//! Parsing here is intentionally strict: every sanitization rule listed in the
//! SIMD (and enforced by the reference implementation in `solana-message`) is
//! checked, so a transaction that parses is one the cluster would accept. Where
//! the SIMD leaves room, this parser picks the stricter reading — unknown
//! config-mask bits are rejected rather than silently dropped, because a dropped
//! bit would change the signed payload.

use std::collections::HashSet;
use std::fmt;

use solana_hash::Hash;
use solana_message::MessageHeader;
use solana_message::VersionedMessage;
use solana_message::compiled_instruction::CompiledInstruction;
use solana_message::v1::{Message as V1Message, TransactionConfig, TransactionConfigMask};
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_transaction::versioned::VersionedTransaction;

use crate::error::{Result, SonarSimError};

/// Version byte that distinguishes v1 from legacy/v0 transaction formats.
pub const V1_PREFIX: u8 = 0x81;

/// Maximum serialized size of a v1 transaction.
pub const MAX_TRANSACTION_SIZE: usize = 4096;

/// Maximum number of signatures in a v1 transaction.
pub const MAX_SIGNATURES: u8 = 12;

/// Maximum number of addresses in a v1 message.
pub const MAX_ADDRESSES: u8 = 64;

/// Maximum number of instructions in a v1 message.
pub const MAX_INSTRUCTIONS: u8 = 64;

/// Minimum requested heap size (32 KiB).
pub const MIN_HEAP_SIZE: u32 = 32 * 1024;

/// Maximum requested heap size (256 KiB).
pub const MAX_HEAP_SIZE: u32 = 256 * 1024;

/// Size of an Ed25519 signature.
pub const SIGNATURE_SIZE: usize = 64;

/// Bytes preceding the address table: version byte + header + mask + lifetime
/// specifier + instruction/address counts.
const FIXED_HEADER_SIZE: usize = 1 + 3 + 4 + 32 + 1 + 1;

// ── TransactionConfigMask bits ──

/// Priority fee (bits 0-1, both required): total lamports, 8 bytes little-endian.
pub const MASK_PRIORITY_FEE: u32 = 0b11;
/// Compute unit limit (bit 2): 4 bytes little-endian.
pub const MASK_COMPUTE_UNIT_LIMIT: u32 = 0b100;
/// Loaded accounts data size limit (bit 3): 4 bytes little-endian.
pub const MASK_LOADED_ACCOUNTS_DATA_SIZE: u32 = 0b1000;
/// Requested heap size (bit 4): 4 bytes little-endian.
pub const MASK_HEAP_SIZE: u32 = 0b1_0000;
/// All bits defined by SIMD-0385.
pub const MASK_KNOWN_BITS: u32 =
    MASK_PRIORITY_FEE | MASK_COMPUTE_UNIT_LIMIT | MASK_LOADED_ACCOUNTS_DATA_SIZE | MASK_HEAP_SIZE;

/// Config requests carried in the transaction header instead of by
/// `ComputeBudget` instructions.
///
/// `None` means the request was not present in the mask; the SIMD defines the
/// effective default for each absent field (0 lamports, 0 compute units, 0
/// loaded-accounts-data bytes, 32 KiB heap).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V1Config {
    /// Total lamports offered as a priority fee.
    pub priority_fee: Option<u64>,
    /// Requested compute unit limit.
    pub compute_unit_limit: Option<u32>,
    /// Requested limit on loaded accounts data size, in bytes.
    pub loaded_accounts_data_size_limit: Option<u32>,
    /// Requested heap size in bytes; must be a 1 KiB multiple in `[32 KiB, 256 KiB]`.
    pub heap_size: Option<u32>,
}

impl From<TransactionConfig> for V1Config {
    /// Read back the config requests the upstream message carries.
    ///
    /// The inverse of what [`V1Transaction::to_v1_message`] writes, so a message
    /// that has been mutated (or built by a caller) is authoritative for its own
    /// config rather than the view it was parsed from.
    fn from(config: TransactionConfig) -> Self {
        Self {
            priority_fee: config.priority_fee,
            compute_unit_limit: config.compute_unit_limit,
            loaded_accounts_data_size_limit: config.loaded_accounts_data_size_limit,
            heap_size: config.heap_size,
        }
    }
}

impl V1Config {
    /// Whether no config request is present.
    pub fn is_empty(&self) -> bool {
        self.priority_fee.is_none()
            && self.compute_unit_limit.is_none()
            && self.loaded_accounts_data_size_limit.is_none()
            && self.heap_size.is_none()
    }

    /// The config mask that encodes exactly these requests.
    pub fn mask(&self) -> u32 {
        let mut mask = 0;
        if self.priority_fee.is_some() {
            mask |= MASK_PRIORITY_FEE;
        }
        if self.compute_unit_limit.is_some() {
            mask |= MASK_COMPUTE_UNIT_LIMIT;
        }
        if self.loaded_accounts_data_size_limit.is_some() {
            mask |= MASK_LOADED_ACCOUNTS_DATA_SIZE;
        }
        if self.heap_size.is_some() {
            mask |= MASK_HEAP_SIZE;
        }
        mask
    }

    /// Effective heap size: 32 KiB when the request is absent.
    pub fn effective_heap_size(&self) -> u32 {
        self.heap_size.unwrap_or(MIN_HEAP_SIZE)
    }

    /// Effective compute unit limit: 0 when the request is absent.
    pub fn effective_compute_unit_limit(&self) -> u32 {
        self.compute_unit_limit.unwrap_or(0)
    }

    /// Effective loaded accounts data size limit: 0 when the request is absent.
    pub fn effective_loaded_accounts_data_size_limit(&self) -> u32 {
        self.loaded_accounts_data_size_limit.unwrap_or(0)
    }
}

impl fmt::Display for V1Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        if let Some(fee) = self.priority_fee {
            parts.push(format!("priority_fee={fee} lamports"));
        }
        if let Some(limit) = self.compute_unit_limit {
            parts.push(format!("compute_unit_limit={limit}"));
        }
        if let Some(limit) = self.loaded_accounts_data_size_limit {
            parts.push(format!("loaded_accounts_data_size_limit={limit}"));
        }
        if let Some(heap) = self.heap_size {
            parts.push(format!("heap_size={heap}"));
        }
        if parts.is_empty() { f.write_str("none") } else { f.write_str(&parts.join(", ")) }
    }
}

/// One instruction of a v1 message: the program and account operands are
/// indices into the message's address list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V1Instruction {
    /// Index into the message's address list of the program to invoke.
    pub program_account_index: u8,
    /// Indices into the message's address list of the accounts passed to the program.
    pub account_indexes: Vec<u8>,
    /// Instruction data.
    pub data: Vec<u8>,
}

/// A fully parsed v1 transaction message plus its signatures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V1Transaction {
    /// Number of signatures the transaction requires.
    pub num_required_signatures: u8,
    /// Number of signing accounts loaded as read-only.
    pub num_readonly_signed_accounts: u8,
    /// Number of non-signing accounts loaded as read-only.
    pub num_readonly_unsigned_accounts: u8,
    /// Decoded config requests; [`V1Transaction::config_mask`] is derived from them.
    pub config: V1Config,
    /// Recent blockhash (renamed `lifetime_specifier` in v1).
    pub lifetime_specifier: Hash,
    /// Every address the transaction references, in message order.
    pub addresses: Vec<Pubkey>,
    /// Instructions, in execution order.
    pub instructions: Vec<V1Instruction>,
    /// Ed25519 signatures; `signatures[i]` belongs to `addresses[i]`.
    pub signatures: Vec<Signature>,
}

impl V1Transaction {
    /// The config mask these requests encode, byte-identical to the wire mask.
    ///
    /// `parse` reads the wire mask and rejects anything the requests cannot
    /// encode, so deriving it here is not a lossy re-encoding: the mask is a 1:1
    /// function of the `Option` pattern in [`V1Transaction::config`].
    pub fn config_mask(&self) -> u32 {
        self.config.mask()
    }

    /// The message header, which has the same meaning as in legacy/v0.
    pub fn header(&self) -> MessageHeader {
        MessageHeader {
            num_required_signatures: self.num_required_signatures,
            num_readonly_signed_accounts: self.num_readonly_signed_accounts,
            num_readonly_unsigned_accounts: self.num_readonly_unsigned_accounts,
        }
    }

    /// The fee payer: the first writable signing address.
    pub fn fee_payer(&self) -> Option<&Pubkey> {
        self.addresses.first()
    }

    /// Whether `bytes` begins with the v1 version byte.
    ///
    /// There is no ambiguity with legacy/v0: those formats begin with a
    /// single-byte `short_vec` signature count, which always has its high bit
    /// clear (`< 128`), while every versioned message begins with 0x80 or above.
    pub fn is_v1_bytes(bytes: &[u8]) -> bool {
        bytes.first() == Some(&V1_PREFIX)
    }

    /// Parse a complete serialized v1 transaction, enforcing every SIMD-0385
    /// sanitization rule.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let parser = Parser::new(bytes);
        parser.parse()
    }

    /// Serialize back to the exact wire layout.
    ///
    /// Round-trips byte-for-byte with [`V1Transaction::parse`] for any
    /// transaction this parser accepts.
    pub fn serialize(&self) -> Vec<u8> {
        // Every cast below narrows, and `parse` guarantees each value fits: it caps
        // the lists and reads the lengths out of the same width of field. These
        // assertions only fire for a view built by hand, where the truncated bytes
        // would otherwise go out silently.
        debug_assert!(
            self.instructions.len() <= MAX_INSTRUCTIONS as usize,
            "v1 instruction count {} exceeds the {MAX_INSTRUCTIONS} the format can encode",
            self.instructions.len()
        );
        debug_assert!(
            self.addresses.len() <= MAX_ADDRESSES as usize,
            "v1 address count {} exceeds the {MAX_ADDRESSES} the format can encode",
            self.addresses.len()
        );
        for instruction in &self.instructions {
            debug_assert!(
                instruction.account_indexes.len() <= u8::MAX as usize,
                "v1 instruction account count {} exceeds the u8 the format can encode",
                instruction.account_indexes.len()
            );
            debug_assert!(
                instruction.data.len() <= u16::MAX as usize,
                "v1 instruction data length {} exceeds the u16 the format can encode",
                instruction.data.len()
            );
        }

        let mut out =
            Vec::with_capacity(FIXED_HEADER_SIZE + self.addresses.len() * 32 + self.config_size());

        out.push(V1_PREFIX);
        out.push(self.num_required_signatures);
        out.push(self.num_readonly_signed_accounts);
        out.push(self.num_readonly_unsigned_accounts);
        out.extend_from_slice(&self.config_mask().to_le_bytes());
        out.extend_from_slice(self.lifetime_specifier.as_ref());
        out.push(self.instructions.len() as u8);
        out.push(self.addresses.len() as u8);

        for address in &self.addresses {
            out.extend_from_slice(address.as_ref());
        }

        for value in self.config_slots() {
            out.extend_from_slice(&value.to_le_bytes());
        }

        for instruction in &self.instructions {
            out.push(instruction.program_account_index);
            out.push(instruction.account_indexes.len() as u8);
            out.extend_from_slice(&(instruction.data.len() as u16).to_le_bytes());
        }

        for instruction in &self.instructions {
            out.extend_from_slice(&instruction.account_indexes);
            out.extend_from_slice(&instruction.data);
        }

        for signature in &self.signatures {
            out.extend_from_slice(signature.as_ref());
        }

        out
    }

    /// The bytes signed by every signature: the serialization of the message
    /// (everything before the trailing signature array).
    pub fn signed_payload(&self) -> Vec<u8> {
        let mut bytes = self.serialize();
        bytes.truncate(bytes.len() - self.signatures.len() * SIGNATURE_SIZE);
        bytes
    }

    /// Verify every signature against its signing address, returning an error
    /// naming the first failure.
    ///
    /// v1 signatures cover the v1 payload, which is why this cannot be delegated
    /// to a legacy/v0 verifier.
    pub fn verify_signatures(&self) -> Result<()> {
        let payload = self.signed_payload();
        for (index, signature) in self.signatures.iter().enumerate() {
            let Some(address) = self.addresses.get(index) else {
                return Err(SonarSimError::TransactionParse {
                    reason: format!(
                        "v1 transaction has {} signatures but only {} addresses",
                        self.signatures.len(),
                        self.addresses.len()
                    ),
                });
            };
            if signature == &Signature::default() {
                return Err(SonarSimError::TransactionParse {
                    reason: format!("v1 transaction signature {index} is missing"),
                });
            }
            if !signature.verify(address.as_ref(), &payload) {
                return Err(SonarSimError::TransactionParse {
                    reason: format!(
                        "v1 transaction signature {index} is invalid for signer {address}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// The message as the Solana crates model it, in the native v1 form.
    ///
    /// Execution goes through this value rather than through a legacy lowering,
    /// so the backend sees a real v1 message: the config requests in the header
    /// (compute unit limit, heap size, loaded accounts data size, priority fee)
    /// take effect exactly like the equivalent `ComputeBudget` instructions do
    /// for legacy/v0.
    ///
    /// Signatures are carried over verbatim; they cover the v1 payload, so
    /// signature verification must use [`V1Transaction::verify_signatures`].
    pub fn to_versioned_transaction(&self) -> VersionedTransaction {
        VersionedTransaction {
            signatures: self.signatures.clone(),
            message: VersionedMessage::V1(self.to_v1_message()),
        }
    }

    /// Convert the parsed message into the upstream representation.
    ///
    /// The reconstruction is exact: the mask upstream derives from these config
    /// requests is the mask the transaction carried on the wire (see
    /// `upstream_config_mask_matches_the_wire_mask`).
    pub fn to_v1_message(&self) -> V1Message {
        let mut config = TransactionConfig::empty();
        if let Some(fee) = self.config.priority_fee {
            config = config.with_priority_fee(fee);
        }
        if let Some(limit) = self.config.compute_unit_limit {
            config = config.with_compute_unit_limit(limit);
        }
        if let Some(limit) = self.config.loaded_accounts_data_size_limit {
            config = config.with_loaded_accounts_data_size_limit(limit);
        }
        if let Some(size) = self.config.heap_size {
            config = config.with_heap_size(size);
        }

        V1Message {
            header: self.header(),
            config,
            lifetime_specifier: self.lifetime_specifier,
            account_keys: self.addresses.clone(),
            instructions: self
                .instructions
                .iter()
                .map(|instruction| CompiledInstruction {
                    program_id_index: instruction.program_account_index,
                    accounts: instruction.account_indexes.clone(),
                    data: instruction.data.clone(),
                })
                .collect(),
        }
    }

    /// The config mask upstream derives from the parsed config requests.
    pub fn upstream_config_mask(&self) -> u32 {
        TransactionConfigMask::from(&self.to_v1_message().config).0
    }

    /// Rebuild the v1 view of a transaction that has since changed.
    ///
    /// Transaction mutations (instruction and account patches) are applied to
    /// the executable message, which would otherwise leave the stored v1 view
    /// stale — wrong size, wrong instruction list, wrong header config. Everything
    /// derived from the message is re-read from `tx` — including the config and its
    /// mask, so a future config mutation cannot leave a stale view behind — and the
    /// result is put back through [`V1Transaction::parse`], so that exactly one
    /// validator, the v1 parser, decides whether it is still well formed.
    pub fn rebuild_from_executable(&self, tx: &VersionedTransaction) -> Result<Self> {
        let VersionedMessage::V1(message) = &tx.message else {
            return Err(SonarSimError::Internal {
                reason: "v1 transactions execute as native v1 messages, but this transaction \
                         is no longer a v1 message"
                    .into(),
            });
        };

        let rebuilt = V1Transaction {
            num_required_signatures: message.header.num_required_signatures,
            num_readonly_signed_accounts: message.header.num_readonly_signed_accounts,
            num_readonly_unsigned_accounts: message.header.num_readonly_unsigned_accounts,
            config: V1Config::from(message.config),
            lifetime_specifier: message.lifetime_specifier,
            addresses: message.account_keys.clone(),
            instructions: message
                .instructions
                .iter()
                .map(|instruction| V1Instruction {
                    program_account_index: instruction.program_id_index,
                    account_indexes: instruction.accounts.clone(),
                    data: instruction.data.clone(),
                })
                .collect(),
            signatures: tx.signatures.clone(),
        };

        V1Transaction::parse(&rebuilt.serialize())
    }

    /// Bytes occupied by the config values that follow the address table.
    fn config_size(&self) -> usize {
        self.config_slots().iter().map(|_| std::mem::size_of::<u32>()).sum::<usize>()
    }

    /// Config values in wire order (bit order of the mask), as 4-byte
    /// little-endian slots. The priority fee spans two consecutive slots.
    fn config_slots(&self) -> Vec<u32> {
        let mut slots = Vec::new();
        if let Some(fee) = self.config.priority_fee {
            let bytes = fee.to_le_bytes();
            slots.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            slots.push(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]));
        }
        if let Some(limit) = self.config.compute_unit_limit {
            slots.push(limit);
        }
        if let Some(limit) = self.config.loaded_accounts_data_size_limit {
            slots.push(limit);
        }
        if let Some(heap) = self.config.heap_size {
            slots.push(heap);
        }
        slots
    }
}

// ── Parsing ──

/// Bounds-checked reader over the serialized transaction.
struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8]> {
        if self.remaining() < len {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction is truncated: need {len} more byte(s) for {what} at offset {}, \
                     but only {} remain",
                    self.pos,
                    self.remaining()
                ),
            });
        }
        let slice = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }

    fn u16_le(&mut self, what: &str) -> Result<u16> {
        let bytes = self.take(2, what)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32_le(&mut self, what: &str) -> Result<u32> {
        let bytes = self.take(4, what)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64_le(&mut self, what: &str) -> Result<u64> {
        let bytes = self.take(8, what)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(array))
    }

    fn hash(&mut self, what: &str) -> Result<Hash> {
        let bytes = self.take(32, what)?;
        let mut array = [0u8; 32];
        array.copy_from_slice(bytes);
        Ok(Hash::new_from_array(array))
    }

    fn pubkey(&mut self, what: &str) -> Result<Pubkey> {
        let bytes = self.take(32, what)?;
        let mut array = [0u8; 32];
        array.copy_from_slice(bytes);
        Ok(Pubkey::new_from_array(array))
    }

    fn signature(&mut self, what: &str) -> Result<Signature> {
        let bytes = self.take(SIGNATURE_SIZE, what)?;
        let mut array = [0u8; SIGNATURE_SIZE];
        array.copy_from_slice(bytes);
        Ok(Signature::from(array))
    }

    fn parse(mut self) -> Result<V1Transaction> {
        if self.bytes.len() > MAX_TRANSACTION_SIZE {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction is {} bytes, which exceeds the {MAX_TRANSACTION_SIZE}-byte limit",
                    self.bytes.len()
                ),
            });
        }

        let version_byte = self.u8("version byte")?;
        if version_byte != V1_PREFIX {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "expected v1 version byte {V1_PREFIX} (0x{V1_PREFIX:02x}), found {version_byte}"
                ),
            });
        }

        let num_required_signatures = self.u8("num_required_signatures")?;
        let num_readonly_signed_accounts = self.u8("num_readonly_signed_accounts")?;
        let num_readonly_unsigned_accounts = self.u8("num_readonly_unsigned_accounts")?;

        if num_required_signatures == 0 {
            return Err(SonarSimError::TransactionParse {
                reason:
                    "v1 transaction header declares no required signatures; the fee payer must \
                         be a signer"
                        .into(),
            });
        }
        if num_readonly_signed_accounts >= num_required_signatures {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction header is invalid: num_readonly_signed_accounts \
                     ({num_readonly_signed_accounts}) must be less than num_required_signatures \
                     ({num_required_signatures}), otherwise the fee payer would be read-only"
                ),
            });
        }
        if num_required_signatures > MAX_SIGNATURES {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction requests {num_required_signatures} signatures, which exceeds \
                     the limit of {MAX_SIGNATURES}"
                ),
            });
        }

        let config_mask = self.u32_le("transaction config mask")?;
        validate_config_mask(config_mask)?;

        let lifetime_specifier = self.hash("lifetime specifier")?;

        let num_instructions = self.u8("num_instructions")?;
        let num_addresses = self.u8("num_addresses")?;

        if num_instructions > MAX_INSTRUCTIONS {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction has {num_instructions} instructions, which exceeds the limit \
                     of {MAX_INSTRUCTIONS}"
                ),
            });
        }
        if num_addresses > MAX_ADDRESSES {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction references {num_addresses} addresses, which exceeds the limit \
                     of {MAX_ADDRESSES}"
                ),
            });
        }
        let min_addresses =
            usize::from(num_required_signatures) + usize::from(num_readonly_unsigned_accounts);
        if usize::from(num_addresses) < min_addresses {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction declares {num_addresses} addresses, fewer than the \
                     {min_addresses} required by its header \
                     (num_required_signatures + num_readonly_unsigned_accounts)"
                ),
            });
        }

        let num_addresses = usize::from(num_addresses);
        let mut addresses = Vec::with_capacity(num_addresses);
        for index in 0..num_addresses {
            addresses.push(self.pubkey(&format!("address {index}"))?);
        }
        let mut seen = HashSet::with_capacity(num_addresses);
        for address in &addresses {
            if !seen.insert(*address) {
                return Err(SonarSimError::TransactionParse {
                    reason: format!("v1 transaction contains duplicate address {address}"),
                });
            }
        }

        // Config values follow the address table, one 4-byte slot per set bit,
        // in bit order. The priority fee spans two slots (8-byte u64).
        let priority_fee = if config_mask & MASK_PRIORITY_FEE == MASK_PRIORITY_FEE {
            Some(self.u64_le("priority fee")?)
        } else {
            None
        };
        let compute_unit_limit = if config_mask & MASK_COMPUTE_UNIT_LIMIT != 0 {
            Some(self.u32_le("compute unit limit")?)
        } else {
            None
        };
        let loaded_accounts_data_size_limit = if config_mask & MASK_LOADED_ACCOUNTS_DATA_SIZE != 0 {
            Some(self.u32_le("loaded accounts data size limit")?)
        } else {
            None
        };
        let heap_size = if config_mask & MASK_HEAP_SIZE != 0 {
            let heap_size = self.u32_le("heap size")?;
            validate_heap_size(heap_size)?;
            Some(heap_size)
        } else {
            None
        };
        let config = V1Config {
            priority_fee,
            compute_unit_limit,
            loaded_accounts_data_size_limit,
            heap_size,
        };

        let instructions = self.parse_instructions(num_instructions, num_addresses)?;

        let num_signatures = usize::from(num_required_signatures);
        let expected_signature_bytes = num_signatures * SIGNATURE_SIZE;
        if self.remaining() < expected_signature_bytes {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction is truncated: header requires {num_signatures} signature(s) \
                     ({expected_signature_bytes} bytes) but only {} byte(s) remain",
                    self.remaining()
                ),
            });
        }
        let mut signatures = Vec::with_capacity(num_signatures);
        for index in 0..num_signatures {
            signatures.push(self.signature(&format!("signature {index}"))?);
        }
        if self.remaining() > 0 {
            return Err(SonarSimError::TransactionParse {
                reason: format!(
                    "v1 transaction has {} trailing byte(s) after the signature array",
                    self.remaining()
                ),
            });
        }

        Ok(V1Transaction {
            num_required_signatures,
            num_readonly_signed_accounts,
            num_readonly_unsigned_accounts,
            config,
            lifetime_specifier,
            addresses,
            instructions,
            signatures,
        })
    }

    /// Read all instruction headers, then their payloads.
    ///
    /// Headers are read first (they are contiguous on the wire) so the sizes are
    /// known before any payload allocation or index validation.
    fn parse_instructions(
        &mut self,
        num_instructions: u8,
        num_addresses: usize,
    ) -> Result<Vec<V1Instruction>> {
        let mut headers = Vec::with_capacity(usize::from(num_instructions));
        for index in 0..num_instructions {
            let program_account_index = self.u8(&format!("instruction {index} program index"))?;
            let num_accounts = usize::from(self.u8(&format!("instruction {index} account count"))?);
            let data_len = usize::from(self.u16_le(&format!("instruction {index} data length"))?);

            validate_account_index(
                program_account_index,
                num_addresses,
                usize::from(index),
                "program",
            )?;

            let payload_len = num_accounts.checked_add(data_len).ok_or_else(|| {
                SonarSimError::TransactionParse {
                    reason: format!("instruction {index} payload length overflows"),
                }
            })?;
            if payload_len > self.remaining() {
                return Err(SonarSimError::TransactionParse {
                    reason: format!(
                        "v1 transaction is truncated: instruction {index} declares \
                         {num_accounts} account index(es) and {data_len} data byte(s), but only {} \
                         byte(s) remain (later instruction headers, payloads, and the signature \
                         array are all still to come)",
                        self.remaining()
                    ),
                });
            }

            headers.push((program_account_index, num_accounts, data_len));
        }

        let mut instructions = Vec::with_capacity(headers.len());
        for (index, (program_account_index, num_accounts, data_len)) in
            headers.into_iter().enumerate()
        {
            let mut account_indexes = Vec::with_capacity(num_accounts);
            for position in 0..num_accounts {
                let account_index =
                    self.u8(&format!("instruction {index} account index {position}"))?;
                validate_account_index(account_index, num_addresses, index, "account")?;
                account_indexes.push(account_index);
            }
            let data = self.take(data_len, &format!("instruction {index} data"))?.to_vec();
            instructions.push(V1Instruction { program_account_index, account_indexes, data });
        }

        Ok(instructions)
    }
}

/// Reject an instruction index that is out of range, or a program index that
/// names the fee payer (which the format forbids).
fn validate_account_index(
    index: u8,
    num_addresses: usize,
    instruction_index: usize,
    role: &str,
) -> Result<()> {
    if usize::from(index) >= num_addresses {
        return Err(SonarSimError::TransactionParse {
            reason: format!(
                "v1 instruction {instruction_index} {role} index {index} is out of range \
                 (transaction has {num_addresses} addresses)"
            ),
        });
    }
    if role == "program" && index == 0 {
        return Err(SonarSimError::TransactionParse {
            reason: format!(
                "v1 instruction {instruction_index} names the fee payer (address 0) as its \
                 program, which is not allowed"
            ),
        });
    }
    Ok(())
}

/// Reject config masks this version cannot interpret.
///
/// Unknown bits are rejected rather than ignored: each set bit consumes a
/// 4-byte config slot, and the mask itself is part of the signed payload, so a
/// transaction carrying bits this version does not understand cannot be safely
/// interpreted or re-serialized.
fn validate_config_mask(config_mask: u32) -> Result<()> {
    if config_mask & !MASK_KNOWN_BITS != 0 {
        return Err(SonarSimError::TransactionParse {
            reason: format!(
                "v1 transaction config mask 0x{config_mask:08x} sets unsupported bits \
                 (known bits are 0x{MASK_KNOWN_BITS:08x})"
            ),
        });
    }

    let priority_fee_bits = config_mask & MASK_PRIORITY_FEE;
    if priority_fee_bits != 0 && priority_fee_bits != MASK_PRIORITY_FEE {
        return Err(SonarSimError::TransactionParse {
            reason: format!(
                "v1 transaction config mask 0x{config_mask:08x} sets only one of the two \
                 priority-fee bits; both bits 0 and 1 must be set together"
            ),
        });
    }

    Ok(())
}

/// A requested heap size must be a 1 KiB multiple within the allowed range.
fn validate_heap_size(heap_size: u32) -> Result<()> {
    if heap_size % 1024 != 0 || !(MIN_HEAP_SIZE..=MAX_HEAP_SIZE).contains(&heap_size) {
        return Err(SonarSimError::TransactionParse {
            reason: format!(
                "v1 transaction requests a heap size of {heap_size} bytes, which must be a \
                 multiple of 1024 within [{MIN_HEAP_SIZE}, {MAX_HEAP_SIZE}]"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use solana_transaction::versioned::VersionedTransaction;
    use std::str::FromStr;

    /// Fixtures are produced by the reference implementation (the standalone
    /// `solana-message` / `solana-transaction` 5.0 release, via `wincode`), so these
    /// tests pin Sonar's hand-written parser to authoritative bytes. This workspace
    /// resolves the 4.4.1 line instead, which is what
    /// `agrees_with_upstream_on_every_fixture` cross-checks each fixture against.
    ///
    /// A: one signature, one instruction, every config field configured
    /// (mask `0b11111`).
    ///
    /// This is the shared v1 transfer fixture (`tests/fixtures/v1_transfer.b64`),
    /// which the CLI and pipeline tests use too, so it lives in one file. The
    /// assertions below pin its exact contents.
    const FIXTURE_A: &str = include_str!("../tests/fixtures/v1_transfer.b64");
    /// B: two instructions (transfer + memo), no config (mask `0`).
    const FIXTURE_B: &str = "gQEAAgAAAAAFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQIEZr5+Myx6RTMyvZ0Kf32wVfXF7xoGraZtmLOftoEMRzqFDy1uAqR6+CTQmradxC1wyyjL+iSft+5XudJWwSdi7wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABUpTWpkpIQZNJOhxYNo4fHw1td28kruB5B+oQEEFRI0CAgwAAwEQAAABAgAAACoAAAAAAAAAAHNvbmFyIHYxIGZpeHR1cmV+VcSdX9m5IvKsIdMVrYWKpAhaSeQ+zZtsr94Mo5omXDh4UwpoE3TDQGZgydxyD0Qg3NGp+WslzojLjBWTFHII";
    /// C: two signatures (one read-only), compute unit limit +
    /// loaded-accounts-data size configured (mask `0b1100`).
    const FIXTURE_C: &str = "gQIBAgwAAAAJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQIFkaKKC3Q4FZOk2UaVeSCJJq/IrYLIg5t2RDWbnrqaSzrQSrIydCu0qzoTaL1GFeTm0CJKtxoBa6+FIKMyyXeHN4UPLW4CpHr4JNCatp3ELXDLKMv6JJ+37le50lbBJ2LvAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFSlNamSkhBk0k6HFg2jh8fDW13bySu4HkH6hAQQVEjcBcFQAAAAAEAwIMAAQBDQAAAgIAAAAHAAAAAAAAAAFzZWNvbmQgc2lnbmVyY7TQyv/p4M4K4P4bj1kLhD8zy7gZfxRDOnTpPzVJu58I9RrQkQTB3PKaBoNSHOpwyoLRXCPS0AYDDtFCa1LSDMfriX6pEGEfvNOQUaS5+cQVrKC0JZCFAnzKeEvWhwAoxf70gd7nAURL3DH2eBQx8B6INgPKtOmWw3n3CcZ2ywk=";
    /// A real mainnet v1 transaction, so the parser is also exercised against
    /// production data instead of only generated fixtures.
    ///
    /// Signature `5N7KXKgV496MGXUnfuLHixBSCEGiLgBXJa8HK3RETixzYVAjwB5fwQoAxS9Tj2Zz5UqyFwCa3sG5b1Ae1DwrGBwv`,
    /// slot 447950910: 2231 bytes, one signature,
    /// 63 addresses, two instructions (a 1_000_000-lamport system transfer and a
    /// program call taking 61 accounts). The chain charged 5176 lamports for it —
    /// 5000 base for one signature plus exactly the 176-lamport priority fee
    /// this parser reads out of the config mask.
    const MAINNET_TX: &str = include_str!("../tests/fixtures/mainnet_v1_transaction.b64");

    const PAYER_A: &str = "GmaDrppBC7P5ARKV8g3djiwP89vz1jLK23V2GBjuAEGB";
    const RECIPIENT_A: &str = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin";
    const PAYER_B: &str = "7v54NWdBtkjuAFJrLGsS2SXnuk8nKam81mZJeeYxVFi9";
    const PAYER_C: &str = "AoVsGaj8MSJ6xwKxfFxo9iZWH3enC8RRTXKH2fx2F8os";
    const SIGNER_C: &str = "F25s3DdjXdCxYBhh2z8FBusVEMT4b9bGNFVKJi3wFoF4";
    const SYSTEM_PROGRAM: &str = "11111111111111111111111111111111";
    const MEMO_PROGRAM: &str = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr";

    fn bytes(fixture: &str) -> Vec<u8> {
        // `include_str!` fixtures keep their trailing newline.
        BASE64.decode(fixture.trim()).expect("fixture is valid base64")
    }

    fn parse(fixture: &str) -> V1Transaction {
        V1Transaction::parse(&bytes(fixture)).expect("fixture parses")
    }

    fn pubkey(value: &str) -> Pubkey {
        Pubkey::from_str(value).expect("valid pubkey")
    }

    #[test]
    fn parses_single_instruction_transaction_with_config() {
        let tx = parse(FIXTURE_A);

        assert_eq!(tx.num_required_signatures, 1);
        assert_eq!(tx.num_readonly_signed_accounts, 0);
        assert_eq!(tx.num_readonly_unsigned_accounts, 1);
        assert_eq!(tx.instructions.len(), 1);
        assert_eq!(tx.addresses.len(), 3);
        assert_eq!(tx.signatures.len(), 1);
        assert_eq!(tx.addresses[0], pubkey(PAYER_A));
        assert_eq!(tx.addresses[1], pubkey(RECIPIENT_A));
        assert_eq!(tx.addresses[2], pubkey(SYSTEM_PROGRAM));
        assert_eq!(tx.fee_payer(), Some(&pubkey(PAYER_A)));

        assert_eq!(tx.config_mask(), 0b11111);
        assert_eq!(
            tx.config,
            V1Config {
                priority_fee: Some(5_000),
                compute_unit_limit: Some(200_000),
                loaded_accounts_data_size_limit: Some(64 * 1024 * 1024),
                heap_size: Some(32 * 1024),
            }
        );

        let instruction = &tx.instructions[0];
        assert_eq!(instruction.program_account_index, 2);
        assert_eq!(instruction.account_indexes, vec![0, 1]);
        let mut expected_data = 2u32.to_le_bytes().to_vec();
        expected_data.extend_from_slice(&1_000_000u64.to_le_bytes());
        assert_eq!(instruction.data, expected_data);
    }

    #[test]
    fn parses_two_instructions_without_config() {
        let tx = parse(FIXTURE_B);

        assert_eq!(tx.addresses[0], pubkey(PAYER_B));
        assert_eq!(tx.addresses[2], pubkey(SYSTEM_PROGRAM));
        assert_eq!(tx.addresses[3], pubkey(MEMO_PROGRAM));
        assert_eq!(tx.config_mask(), 0);
        assert!(tx.config.is_empty());
        assert_eq!(tx.instructions.len(), 2);
        assert_eq!(tx.instructions[0].program_account_index, 2);
        assert_eq!(tx.instructions[0].account_indexes, vec![0, 1]);
        assert_eq!(tx.instructions[1].program_account_index, 3);
        assert_eq!(tx.instructions[1].account_indexes, vec![0]);
        assert_eq!(tx.instructions[1].data, b"sonar v1 fixture".to_vec());
    }

    /// SIMD-0385: an unset field means the *minimum* value, and for the loaded
    /// accounts data size limit the minimum is zero — deliberately unlike the
    /// 64 MiB default legacy/v0 transactions get. A transaction that leaves bit
    /// 3 unset therefore asks for a zero-byte budget and cannot load anything,
    /// which is why executable v1 transactions (including the mainnet fixture)
    /// always request it.
    #[test]
    fn unset_config_fields_use_the_documented_minimums() {
        let tx = parse(FIXTURE_B);
        assert_eq!(tx.config_mask(), 0);
        assert_eq!(tx.config.priority_fee, None);
        assert_eq!(tx.config.compute_unit_limit, None);
        assert_eq!(tx.config.loaded_accounts_data_size_limit, None);
        assert_eq!(tx.config.heap_size, None);

        assert_eq!(tx.config.effective_compute_unit_limit(), 0);
        assert_eq!(tx.config.effective_loaded_accounts_data_size_limit(), 0);
        assert_eq!(tx.config.effective_heap_size(), MIN_HEAP_SIZE);
    }

    #[test]
    fn parses_readonly_signer_and_loaded_data_size_limit() {
        let tx = parse(FIXTURE_C);

        assert_eq!(tx.num_required_signatures, 2);
        assert_eq!(tx.num_readonly_signed_accounts, 1);
        assert_eq!(tx.num_readonly_unsigned_accounts, 2);
        assert_eq!(tx.addresses[0], pubkey(PAYER_C));
        assert_eq!(tx.addresses[1], pubkey(SIGNER_C));
        assert_eq!(tx.signatures.len(), 2);
        assert_eq!(
            tx.config,
            V1Config {
                priority_fee: None,
                compute_unit_limit: Some(1_400_000),
                loaded_accounts_data_size_limit: Some(64 * 1024 * 1024),
                heap_size: None,
            }
        );
        assert_eq!(tx.config.effective_heap_size(), MIN_HEAP_SIZE);
    }

    #[test]
    fn round_trips_reference_fixtures_byte_for_byte() {
        for fixture in [FIXTURE_A, FIXTURE_B, FIXTURE_C] {
            let original = bytes(fixture);
            let parsed = V1Transaction::parse(&original).expect("parses");
            assert_eq!(parsed.serialize(), original, "round trip must be exact");
        }
    }

    #[test]
    fn signed_payload_excludes_signature_array() {
        let tx = parse(FIXTURE_A);
        let payload = tx.signed_payload();
        let serialized = tx.serialize();
        assert_eq!(payload.len(), serialized.len() - SIGNATURE_SIZE);
        assert_eq!(payload, serialized[..payload.len()].to_vec());
        assert_eq!(payload[0], V1_PREFIX);
    }

    #[test]
    fn verifies_reference_fixture_signatures() {
        parse(FIXTURE_A).verify_signatures().expect("fixture A is correctly signed");
        parse(FIXTURE_B).verify_signatures().expect("fixture B is correctly signed");
        parse(FIXTURE_C).verify_signatures().expect("fixture C is correctly signed");
    }

    /// The Solana crates gained native v1 support in 4.x, so this parser can be
    /// checked against upstream rather than only against the fixtures it was
    /// built from: both must produce the same message and the same bytes.
    #[test]
    fn agrees_with_upstream_on_every_fixture() {
        for (label, fixture) in
            [("A", FIXTURE_A), ("B", FIXTURE_B), ("C", FIXTURE_C), ("mainnet", MAINNET_TX)]
        {
            let raw = bytes(fixture);
            assert!(V1Transaction::is_v1_bytes(&raw), "{label}: detected as v1");

            let upstream: VersionedTransaction =
                wincode::deserialize(&raw).unwrap_or_else(|e| panic!("{label}: upstream: {e}"));
            assert_eq!(wincode::serialize(&upstream).unwrap(), raw, "{label}: wire bytes");

            let ours = parse(fixture);
            // The executable form must serialize to exactly the input bytes, so
            // what the backend runs is byte-for-byte what the cluster accepted.
            assert_eq!(
                wincode::serialize(&ours.to_versioned_transaction()).unwrap(),
                raw,
                "{label}: executable form wire bytes"
            );
            let VersionedMessage::V1(message) = &upstream.message else {
                panic!("{label}: upstream did not read a v1 message");
            };

            assert_eq!(message.header, ours.header(), "{label}: header");
            assert_eq!(message.account_keys, ours.addresses, "{label}: addresses");
            assert_eq!(message.lifetime_specifier, ours.lifetime_specifier, "{label}: lifetime");
            assert_eq!(upstream.signatures, ours.signatures, "{label}: signatures");
            assert_eq!(
                message.config.priority_fee, ours.config.priority_fee,
                "{label}: priority fee"
            );
            assert_eq!(
                message.config.compute_unit_limit, ours.config.compute_unit_limit,
                "{label}: compute unit limit"
            );
            assert_eq!(
                message.config.loaded_accounts_data_size_limit,
                ours.config.loaded_accounts_data_size_limit,
                "{label}: loaded accounts data size"
            );
            assert_eq!(message.config.heap_size, ours.config.heap_size, "{label}: heap size");
            assert_eq!(message.instructions.len(), ours.instructions.len(), "{label}: ix count");

            for (upstream_ix, our_ix) in message.instructions.iter().zip(&ours.instructions) {
                assert_eq!(
                    upstream_ix.program_id_index, our_ix.program_account_index,
                    "{label}: program index"
                );
                assert_eq!(upstream_ix.accounts, our_ix.account_indexes, "{label}: ix accounts");
                assert_eq!(upstream_ix.data, our_ix.data, "{label}: ix data");
            }
        }
    }

    #[test]
    fn parses_mainnet_transaction() {
        let tx = parse(MAINNET_TX);

        assert_eq!(tx.num_required_signatures, 1);
        assert_eq!(tx.addresses.len(), 63);
        assert_eq!(tx.instructions.len(), 2);
        assert_eq!(tx.serialize().len(), 2231);

        assert_eq!(tx.config_mask(), 0b1111);
        assert_eq!(tx.config.priority_fee, Some(176));
        assert_eq!(tx.config.compute_unit_limit, Some(84_501));
        assert_eq!(tx.config.loaded_accounts_data_size_limit, Some(67_108_864));
        assert_eq!(tx.config.heap_size, None);
        assert_eq!(tx.config.effective_heap_size(), MIN_HEAP_SIZE);

        // A system transfer, then a program call whose program sits at a high
        // account index — the shape that motivated the format.
        assert_eq!(tx.instructions[0].program_account_index, 10);
        assert_eq!(tx.instructions[1].program_account_index, 47);
        assert_eq!(tx.instructions[1].account_indexes.len(), 61);

        assert!(tx.instructions.iter().all(|instruction| {
            instruction.account_indexes.iter().all(|index| (*index as usize) < tx.addresses.len())
        }));
        tx.verify_signatures().expect("mainnet signature verifies");
    }

    #[test]
    fn rejects_tampered_signature() {
        let mut tx = parse(FIXTURE_A);
        tx.signatures[0] = Signature::from([9u8; SIGNATURE_SIZE]);
        let err = tx.verify_signatures().unwrap_err();
        assert!(err.to_string().contains("signature 0 is invalid"), "{err}");
    }

    #[test]
    fn rejects_zeroed_signature() {
        let mut tx = parse(FIXTURE_A);
        tx.signatures[0] = Signature::default();
        let err = tx.verify_signatures().unwrap_err();
        assert!(err.to_string().contains("signature 0 is missing"), "{err}");
    }

    #[test]
    fn executes_as_a_native_v1_message() {
        let tx = parse(FIXTURE_B);
        let message = tx.to_v1_message();

        assert_eq!(message.header, tx.header());
        assert_eq!(message.account_keys, tx.addresses);
        assert_eq!(message.lifetime_specifier, tx.lifetime_specifier);
        assert_eq!(message.instructions.len(), 2);
        assert_eq!(message.instructions[0].program_id_index, 2);
        assert_eq!(message.instructions[0].accounts, vec![0, 1]);
        assert_eq!(message.instructions[1].program_id_index, 3);
        assert_eq!(message.instructions[1].data, b"sonar v1 fixture".to_vec());

        let versioned = tx.to_versioned_transaction();
        assert!(matches!(versioned.message, VersionedMessage::V1(_)));
        assert_eq!(versioned.signatures, tx.signatures);

        // The executable message has to carry the config requests: that is what
        // the backend reads to honor the header.
        assert_eq!(message.config.priority_fee, tx.config.priority_fee);
        assert_eq!(message.config.compute_unit_limit, tx.config.compute_unit_limit);
        assert_eq!(message.config.heap_size, tx.config.heap_size);
    }

    /// The config requests are re-encoded into the upstream representation for
    /// execution; the mask upstream derives from them must therefore be the mask
    /// that was on the wire.
    /// A rebuilt view takes its config from the message, not from the view it was
    /// parsed from: a config-touching mutation must not leave a stale copy behind.
    #[test]
    fn rebuild_re_reads_the_config_from_the_message() {
        let tx = parse(FIXTURE_A);
        let mut executable = tx.to_versioned_transaction();
        let VersionedMessage::V1(message) = &mut executable.message else {
            panic!("fixture A executes as a native v1 message");
        };
        message.config.compute_unit_limit = Some(99_999);
        // Dropping a request has to drop its mask bit as well.
        message.config.heap_size = None;

        let rebuilt = tx.rebuild_from_executable(&executable).expect("rebuilds");

        assert_eq!(rebuilt.config.compute_unit_limit, Some(99_999));
        assert_eq!(rebuilt.config.heap_size, None);
        assert_eq!(rebuilt.config_mask(), tx.config_mask() & !MASK_HEAP_SIZE);
    }

    /// Serialize a message with the reference implementation's writer.
    fn reference_wire_bytes(message: &V1Message) -> Vec<u8> {
        let tx = VersionedTransaction {
            signatures: vec![
                Signature::default();
                usize::from(message.header.num_required_signatures)
            ],
            message: VersionedMessage::V1(message.clone()),
        };
        wincode::serialize(&tx).expect("the reference implementation serializes the message")
    }

    /// Assert that Sonar and the reference implementation both reject `bytes`.
    ///
    /// Rejecting a transaction the cluster would accept is the dangerous kind of
    /// parser bug, so every rule beyond the wire layout is checked against the
    /// implementation rather than only against the fixtures this parser accepts.
    fn assert_rejected_by_both(label: &str, bytes: &[u8]) {
        assert!(V1Transaction::parse(bytes).is_err(), "{label}: Sonar accepted it");
        let reference_rejects = match wincode::deserialize::<VersionedTransaction>(bytes) {
            Ok(tx) => tx.sanitize().is_err(),
            Err(_) => true,
        };
        assert!(reference_rejects, "{label}: the reference implementation accepted it");
    }

    /// Sonar's rejection rules are the reference implementation's rules.
    #[test]
    fn rejections_agree_with_the_reference_implementation() {
        // Control: the unmutated fixture is accepted by both, so the cases below
        // fail for the mutation and not for the way they are built.
        let pristine = parse(FIXTURE_A).to_v1_message();
        let pristine_bytes = reference_wire_bytes(&pristine);
        assert!(V1Transaction::parse(&pristine_bytes).is_ok());
        assert!(
            wincode::deserialize::<VersionedTransaction>(&pristine_bytes)
                .expect("reference parses the fixture")
                .sanitize()
                .is_ok()
        );

        type Mutation = fn(&mut V1Message);
        let cases: [(&str, Mutation); 9] = [
            ("program index 0 (the fee payer)", |m| m.instructions[0].program_id_index = 0),
            ("program index out of range", |m| m.instructions[0].program_id_index = 60),
            ("instruction account index out of range", |m| m.instructions[0].accounts[0] = 60),
            ("duplicate address", |m| {
                let first = m.account_keys[0];
                m.account_keys[1] = first;
            }),
            ("heap below the minimum", |m| m.config.heap_size = Some(1024)),
            ("heap above the maximum", |m| m.config.heap_size = Some(512 * 1024)),
            ("heap not a multiple of 1024", |m| m.config.heap_size = Some(33 * 1024 + 1)),
            ("no required signatures", |m| m.header.num_required_signatures = 0),
            ("read-only fee payer", |m| m.header.num_readonly_signed_accounts = 1),
        ];
        for (label, mutate) in cases {
            let mut message = parse(FIXTURE_A).to_v1_message();
            mutate(&mut message);
            assert_rejected_by_both(label, &reference_wire_bytes(&message));
        }

        // The mask is derived from the typed config when the reference
        // implementation serializes, so mask-level rules are checked on raw bytes.
        for (label, mask) in
            [("only one priority-fee bit", 0b0_0001u32), ("unknown mask bit", 0b10_0000)]
        {
            let mut bytes = bytes(FIXTURE_A);
            bytes[4..8].copy_from_slice(&mask.to_le_bytes());
            assert_rejected_by_both(label, &bytes);
        }
    }

    /// A v1 message built with the reference types, for boundary cases.
    ///
    /// Every instruction targets address 1, so callers need at least two: address 0
    /// is the fee payer, which may not be used as a program.
    fn reference_message(addresses: usize, instructions: usize, heap: Option<u32>) -> V1Message {
        assert!(addresses >= 2, "address 0 is the fee payer, address 1 the program");
        let mut message = parse(FIXTURE_A).to_v1_message();
        message.header.num_required_signatures = 1;
        message.header.num_readonly_signed_accounts = 0;
        message.header.num_readonly_unsigned_accounts = 0;
        message.account_keys = (0..addresses).map(|_| Pubkey::new_unique()).collect();
        message.instructions = (0..instructions)
            .map(|_| CompiledInstruction {
                program_id_index: 1,
                accounts: vec![1],
                data: vec![0u8; 4],
            })
            .collect();
        message.config.heap_size = heap;
        message
    }

    /// Values exactly at a documented limit are accepted, so the parser cannot
    /// turn a cluster-valid transaction away over an off-by-one.
    #[test]
    fn accepts_values_exactly_at_the_format_limits() {
        let cases: [(&str, V1Message); 5] = [
            ("12 addresses and 12 signatures", {
                let mut message = reference_message(12, 1, None);
                message.header.num_required_signatures = MAX_SIGNATURES;
                message
            }),
            ("64 addresses", reference_message(MAX_ADDRESSES as usize, 1, None)),
            ("64 instructions", reference_message(2, MAX_INSTRUCTIONS as usize, None)),
            ("minimum heap size", reference_message(2, 1, Some(MIN_HEAP_SIZE))),
            ("maximum heap size", reference_message(2, 1, Some(MAX_HEAP_SIZE))),
        ];
        for (label, message) in cases {
            let bytes = reference_wire_bytes(&message);
            let parsed =
                V1Transaction::parse(&bytes).unwrap_or_else(|err| panic!("{label}: {err}"));
            assert_eq!(parsed.serialize(), bytes, "{label}: round trip");
            assert!(
                wincode::deserialize::<VersionedTransaction>(&bytes)
                    .expect("reference parses it")
                    .sanitize()
                    .is_ok(),
                "{label}: the reference implementation rejected it"
            );
        }
    }

    /// The 4096-byte cap is the SIMD's, so a transaction one byte over it is
    /// rejected and one exactly at it is not.
    #[test]
    fn transaction_size_limit_is_inclusive() {
        let mut message = reference_message(2, 1, None);
        message.instructions[0].data = Vec::new();
        let empty_size = reference_wire_bytes(&message).len();
        // Payload length is the only degree of freedom here: one account index plus
        // the instruction data.
        message.instructions[0].data = vec![0u8; MAX_TRANSACTION_SIZE - empty_size];

        let bytes = reference_wire_bytes(&message);
        assert_eq!(bytes.len(), MAX_TRANSACTION_SIZE);
        assert!(V1Transaction::parse(&bytes).is_ok(), "exactly at the limit");

        message.instructions[0].data = vec![0u8; MAX_TRANSACTION_SIZE - empty_size + 1];
        let oversized = reference_wire_bytes(&message);
        assert_eq!(oversized.len(), MAX_TRANSACTION_SIZE + 1);
        let err = V1Transaction::parse(&oversized).expect_err("one byte over the limit");
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[test]
    fn upstream_config_mask_matches_the_wire_mask() {
        for (label, fixture) in
            [("A", FIXTURE_A), ("B", FIXTURE_B), ("C", FIXTURE_C), ("mainnet", MAINNET_TX)]
        {
            let tx = parse(fixture);
            assert_eq!(tx.upstream_config_mask(), tx.config_mask(), "{label}");
        }
    }

    #[test]
    fn detects_v1_by_version_byte() {
        assert!(V1Transaction::is_v1_bytes(&bytes(FIXTURE_A)));
        // Legacy/v0 begin with a `short_vec` signature count below 0x80.
        assert!(!V1Transaction::is_v1_bytes(&[0x01, 0x00, 0x01]));
        assert!(!V1Transaction::is_v1_bytes(&[0x80]));
        assert!(!V1Transaction::is_v1_bytes(&[]));
    }

    // ── Rejections ──

    fn rejection(parse_result: Result<V1Transaction>, expected: &str) {
        let err = parse_result.expect_err("must be rejected");
        let message = err.to_string();
        assert!(message.contains(expected), "expected {expected:?} in {message:?}");
    }

    fn serialize_mutated(mutate: impl FnOnce(&mut V1Transaction)) -> Vec<u8> {
        let mut tx = parse(FIXTURE_A);
        mutate(&mut tx);
        tx.serialize()
    }

    /// A fixture with its wire config mask replaced.
    ///
    /// The mask is derived from the config on serialization, so a mask the config
    /// cannot encode only exists in raw bytes.
    fn raw_with_config_mask(mask: u32) -> Vec<u8> {
        let mut bytes = bytes(FIXTURE_A);
        bytes[4..8].copy_from_slice(&mask.to_le_bytes());
        bytes
    }

    #[test]
    fn rejects_wrong_version_byte() {
        let mut raw = bytes(FIXTURE_A);
        raw[0] = 0x80;
        rejection(V1Transaction::parse(&raw), "expected v1 version byte");
    }

    #[test]
    fn rejects_truncated_transaction() {
        let mut raw = bytes(FIXTURE_A);
        raw.truncate(raw.len() - 1);
        rejection(V1Transaction::parse(&raw), "truncated");
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut raw = bytes(FIXTURE_A);
        raw.push(0x00);
        rejection(V1Transaction::parse(&raw), "trailing byte");
    }

    #[test]
    fn rejects_oversized_transaction() {
        let raw = vec![V1_PREFIX; MAX_TRANSACTION_SIZE + 1];
        rejection(V1Transaction::parse(&raw), "exceeds the 4096-byte limit");
    }

    #[test]
    fn rejects_duplicate_addresses() {
        let raw = serialize_mutated(|tx| tx.addresses[1] = tx.addresses[0]);
        rejection(V1Transaction::parse(&raw), "duplicate address");
    }

    #[test]
    fn rejects_out_of_range_instruction_account_index() {
        let raw = serialize_mutated(|tx| tx.instructions[0].account_indexes = vec![0, 9]);
        rejection(V1Transaction::parse(&raw), "account index 9 is out of range");
    }

    #[test]
    fn rejects_fee_payer_as_program() {
        let raw = serialize_mutated(|tx| tx.instructions[0].program_account_index = 0);
        rejection(V1Transaction::parse(&raw), "names the fee payer");
    }

    #[test]
    fn rejects_readonly_fee_payer_header() {
        let raw = serialize_mutated(|tx| {
            tx.num_required_signatures = 1;
            tx.num_readonly_signed_accounts = 1;
        });
        rejection(V1Transaction::parse(&raw), "num_readonly_signed_accounts");
    }

    #[test]
    fn rejects_more_signatures_than_allowed() {
        let raw = serialize_mutated(|tx| tx.num_required_signatures = MAX_SIGNATURES + 1);
        rejection(V1Transaction::parse(&raw), "exceeds the limit of 12");
    }

    #[test]
    fn rejects_unknown_config_mask_bits() {
        rejection(V1Transaction::parse(&raw_with_config_mask(0b10_0000)), "unsupported bits");
    }

    #[test]
    fn rejects_partial_priority_fee_bits() {
        rejection(
            V1Transaction::parse(&raw_with_config_mask(0b1)),
            "only one of the two priority-fee bits",
        );
    }

    #[test]
    fn rejects_heap_size_out_of_range() {
        let raw = serialize_mutated(|tx| tx.config.heap_size = Some(1024));
        rejection(V1Transaction::parse(&raw), "multiple of 1024");
    }

    #[test]
    fn rejects_misaligned_heap_size() {
        let raw = serialize_mutated(|tx| tx.config.heap_size = Some(33 * 1024 + 1));
        rejection(V1Transaction::parse(&raw), "multiple of 1024");
    }

    #[test]
    fn rejects_missing_signatures() {
        // Declaring two signers while the input carries one signature's worth of
        // trailing bytes fails the signature-array length check.
        let raw = serialize_mutated(|tx| tx.num_required_signatures = 2);
        rejection(V1Transaction::parse(&raw), "truncated");
    }

    #[test]
    fn config_mask_round_trips_every_known_combination() {
        for mask in 0..=MASK_KNOWN_BITS {
            let priority_fee_bits = mask & MASK_PRIORITY_FEE;
            if priority_fee_bits != 0 && priority_fee_bits != MASK_PRIORITY_FEE {
                continue;
            }
            let config = {
                let mut tx = parse(FIXTURE_A);
                tx.config = V1Config {
                    priority_fee: (priority_fee_bits == MASK_PRIORITY_FEE).then_some(7),
                    compute_unit_limit: (mask & MASK_COMPUTE_UNIT_LIMIT != 0).then_some(11),
                    loaded_accounts_data_size_limit: (mask & MASK_LOADED_ACCOUNTS_DATA_SIZE != 0)
                        .then_some(13),
                    heap_size: (mask & MASK_HEAP_SIZE != 0).then_some(MIN_HEAP_SIZE),
                };
                tx
            };
            // The requests are the source of truth, and the wire mask the derived
            // value: both directions have to agree for every combination.
            assert_eq!(config.config_mask(), mask);
            assert_eq!(u32::from_le_bytes(config.serialize()[4..8].try_into().unwrap()), mask);
            let reparsed = V1Transaction::parse(&config.serialize()).expect("mask combination");
            assert_eq!(reparsed.config_mask(), mask);
            assert_eq!(reparsed.config, config.config);
            assert_eq!(reparsed.serialize(), config.serialize());
        }
    }
}
