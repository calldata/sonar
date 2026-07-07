//! The single seam for writing the program's own output to the terminal.
//!
//! `docs/output-conventions.md` defines the contract: **stdout** carries the
//! primary, machine-consumable result (exactly one JSON document under
//! `--json`); **stderr** carries human-facing diagnostics, progress, and
//! confirmations. This module is the only place that writes to those streams
//! directly, so the crate-root `clippy::print_stdout` / `clippy::print_stderr`
//! deny turns a stray `println!` in a handler into a compile error rather than a
//! silently corrupted stdout stream.
//!
//! Use [`outln!`] for primary output and [`errln!`] for diagnostics. Diagnostics
//! that are logging (warnings/errors) should still go through the `log` crate;
//! `errln!` is for direct, unconditional stderr lines (confirmations, hints).

use std::io::Write;

/// Write a line to stdout — the command's primary, machine-consumable result.
/// Prefer the [`outln!`] macro over calling this directly.
pub(crate) fn write_primary(args: std::fmt::Arguments<'_>) {
    let mut out = std::io::stdout().lock();
    let _ = out.write_fmt(args);
    let _ = out.write_all(b"\n");
}

/// Write a line to stderr — a human-facing diagnostic, status, or confirmation
/// that must never contaminate stdout. Prefer the [`errln!`] macro.
pub(crate) fn write_diagnostic(args: std::fmt::Arguments<'_>) {
    let mut err = std::io::stderr().lock();
    let _ = err.write_fmt(args);
    let _ = err.write_all(b"\n");
}

/// Write a line to stdout — the command's primary, machine-consumable result.
/// The stdout counterpart of `println!`, routed through [`write_primary`].
macro_rules! outln {
    ($($arg:tt)*) => {
        $crate::output::stream::write_primary(std::format_args!($($arg)*))
    };
}

/// Write a line to stderr — a human-facing diagnostic or confirmation. The
/// stderr counterpart of `eprintln!`, routed through [`write_diagnostic`].
macro_rules! errln {
    ($($arg:tt)*) => {
        $crate::output::stream::write_diagnostic(std::format_args!($($arg)*))
    };
}

pub(crate) use {errln, outln};
