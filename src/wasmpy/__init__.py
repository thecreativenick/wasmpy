"""wasmpy: a WebAssembly interpreter in Python, aiming at the official spec test suite."""

from .binary import DecodeError, FuncType, Function, Module, decode_module
from .interp import Instance, Trap

__all__ = ["decode_module", "DecodeError", "Module", "FuncType", "Function", "Instance", "Trap"]
