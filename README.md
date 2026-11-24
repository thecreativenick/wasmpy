# wasmpy

A WebAssembly 2.0 interpreter in pure Python that aims to pass the official core spec test
suite (excluding SIMD), with a dashboard of pass/fail counts per `.wast` file.

## Status

Early scaffold. Done so far:

- Binary decoder: header, strict LEB128 (rejects overlong and too-large encodings), and the
  type, function, export and code sections. Section order is checked; other sections are kept raw.
- Interpreter for the i32 instruction set and structured control flow: block, loop, if/else,
  br, br_if, br_table, return, call, select, locals, with traps for divide by zero, signed
  overflow and call-stack exhaustion.
- Tests build modules byte by byte with a tiny assembler (no external tools needed).

## Planned milestones

1. i64, f32 and f64 with exact IEEE semantics (NaN bit patterns, rounding, trunc_sat).
2. Memory, loads and stores, globals, tables, call_indirect, imports and exports, start, data and elem.
3. The validator (the type-checking algorithm from the spec appendix).
4. A text-format (`.wat`) parser and `.wast` script runner: assert_return, assert_trap,
   assert_invalid, assert_malformed, and the rest.
5. Reference types, bulk memory, multi-value; the localhost results dashboard.

## Success bar

Every assertion in the official core `.wast` files passes, except SIMD, with none skipped.

## Development

```bash
pip install -e .[dev]
pytest
```
