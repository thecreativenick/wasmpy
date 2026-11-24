"""A straightforward WebAssembly interpreter for the i32 / control-flow core.

Values on the operand stack are Python ints holding the unsigned 32-bit bit
pattern; signed operations convert on the fly. Block/else/end positions are
found once per function before it first runs.

Next milestones: i64/f32/f64, memory and loads/stores, globals, tables and
call_indirect, imports, validation, and the .wast script runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from .binary import DecodeError, Module, Reader

MASK = 0xFFFFFFFF
MAX_CALL_DEPTH = 300


class Trap(Exception):
    pass


def signed(x: int) -> int:
    return x - (1 << 32) if x & 0x80000000 else x


def _clz(x: int) -> int:
    return 32 - x.bit_length()


def _ctz(x: int) -> int:
    return 32 if x == 0 else (x & -x).bit_length() - 1


def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


BINOPS = {
    0x6A: lambda a, b: (a + b) & MASK,
    0x6B: lambda a, b: (a - b) & MASK,
    0x6C: lambda a, b: (a * b) & MASK,
    0x71: lambda a, b: a & b,
    0x72: lambda a, b: a | b,
    0x73: lambda a, b: a ^ b,
    0x74: lambda a, b: (a << (b & 31)) & MASK,
    0x75: lambda a, b: (signed(a) >> (b & 31)) & MASK,
    0x76: lambda a, b: a >> (b & 31),
    0x77: lambda a, b: ((a << (b & 31)) | (a >> ((32 - (b & 31)) & 31))) & MASK,
    0x78: lambda a, b: ((a >> (b & 31)) | (a << ((32 - (b & 31)) & 31))) & MASK,
    # comparisons
    0x46: lambda a, b: int(a == b),
    0x47: lambda a, b: int(a != b),
    0x48: lambda a, b: int(signed(a) < signed(b)),
    0x49: lambda a, b: int(a < b),
    0x4A: lambda a, b: int(signed(a) > signed(b)),
    0x4B: lambda a, b: int(a > b),
    0x4C: lambda a, b: int(signed(a) <= signed(b)),
    0x4D: lambda a, b: int(a <= b),
    0x4E: lambda a, b: int(signed(a) >= signed(b)),
    0x4F: lambda a, b: int(a >= b),
}
UNOPS = {
    0x45: lambda a: int(a == 0),
    0x67: _clz,
    0x68: _ctz,
    0x69: lambda a: bin(a).count("1"),
}


@dataclass
class Label:
    is_loop: bool
    height: int       # operand stack height below the block's params
    arity: int        # values a branch to this label carries
    target: int       # pc to continue at after a branch


class Instance:
    def __init__(self, module: Module):
        self.module = module
        self._blocks: dict[int, dict[int, tuple[int | None, int]]] = {}
        self._depth = 0

    # ---------------------------------------------------------------- API
    def invoke(self, export: str, *args: int) -> list[int]:
        kind, idx = self.module.exports[export]
        if kind != "func":
            raise TypeError(f"{export} is a {kind}, not a function")
        results = self.call(idx, [a & MASK for a in args])
        return [signed(r) for r in results]

    def call(self, func_index: int, args: list[int]) -> list[int]:
        if self._depth >= MAX_CALL_DEPTH:
            raise Trap("call stack exhausted")
        self._depth += 1
        try:
            return self._run(func_index, args)
        finally:
            self._depth -= 1

    # ------------------------------------------------------- block layout
    def _block_type(self, r: Reader) -> tuple[int, int]:
        """Read a blocktype; returns (param count, result count)."""
        b = r.data[r.pos]
        if b == 0x40:
            r.pos += 1
            return 0, 0
        if b in (0x7F, 0x7E, 0x7D, 0x7C, 0x70, 0x6F):
            r.pos += 1
            return 0, 1
        ft = self.module.types[r.s33()]
        return len(ft.params), len(ft.results)

    def _layout(self, func_index: int) -> dict[int, tuple[int | None, int]]:
        """Map each block/loop/if opcode position -> (else position or None, end position)."""
        if func_index in self._blocks:
            return self._blocks[func_index]
        code = self.module.functions[func_index].body
        r = Reader(code)
        open_: list[list] = []
        out: dict[int, tuple[int | None, int]] = {}
        while not r.eof():
            pc = r.pos
            op = r.byte()
            if op in (0x02, 0x03, 0x04):
                self._block_type(r)
                open_.append([pc, None])
            elif op == 0x05:
                if not open_:
                    raise DecodeError("else without if")
                open_[-1][1] = pc
            elif op == 0x0B:
                if open_:
                    start, else_pc = open_.pop()
                    out[start] = (else_pc, pc)
            elif op in (0x0C, 0x0D, 0x10, 0x20, 0x21, 0x22, 0x23, 0x24):
                r.u32()
            elif op == 0x0E:
                for _ in range(r.u32() + 1):
                    r.u32()
            elif op == 0x41:
                r.s32()
            elif op == 0x42:
                r.s64()
        if open_:
            raise DecodeError("unexpected end of function body")
        self._blocks[func_index] = out
        return out

    # ---------------------------------------------------------- execution
    def _run(self, func_index: int, args: list[int]) -> list[int]:
        mod = self.module
        func = mod.functions[func_index]
        ftype = mod.types[func.type_index]
        if len(args) != len(ftype.params):
            raise TypeError(f"expected {len(ftype.params)} arguments, got {len(args)}")
        locals_ = list(args) + [0] * len(func.locals)
        layout = self._layout(func_index)
        code = func.body
        r = Reader(code)
        stack: list[int] = []
        labels: list[Label] = []
        nresults = len(ftype.results)

        def branch(depth: int) -> None:
            label = labels[-1 - depth]
            carried = stack[len(stack) - label.arity:] if label.arity else []
            del stack[label.height:]
            stack.extend(carried)
            if label.is_loop:
                del labels[len(labels) - depth:]
            else:
                del labels[len(labels) - 1 - depth:]
            r.pos = label.target

        while True:
            pc = r.pos
            op = r.byte()
            if op in BINOPS:
                b = stack.pop()
                a = stack.pop()
                stack.append(BINOPS[op](a, b))
            elif op in UNOPS:
                stack.append(UNOPS[op](stack.pop()))
            elif op == 0x20:
                stack.append(locals_[r.u32()])
            elif op == 0x21:
                locals_[r.u32()] = stack.pop()
            elif op == 0x22:
                locals_[r.u32()] = stack[-1]
            elif op == 0x41:
                stack.append(r.s32() & MASK)
            elif op in (0x6D, 0x6E, 0x6F, 0x70):
                b = stack.pop()
                a = stack.pop()
                if b == 0:
                    raise Trap("integer divide by zero")
                if op == 0x6D:
                    if signed(a) == -(1 << 31) and signed(b) == -1:
                        raise Trap("integer overflow")
                    stack.append(_trunc_div(signed(a), signed(b)) & MASK)
                elif op == 0x6E:
                    stack.append(a // b)
                elif op == 0x6F:
                    sa, sb = signed(a), signed(b)
                    stack.append((sa - sb * _trunc_div(sa, sb)) & MASK)
                else:
                    stack.append(a % b)
            elif op in (0x02, 0x03):
                nparams, nres = self._block_type(r)
                _, end = layout[pc]
                if op == 0x02:
                    labels.append(Label(False, len(stack) - nparams, nres, end + 1))
                else:
                    labels.append(Label(True, len(stack) - nparams, nparams, r.pos))
            elif op == 0x04:
                nparams, nres = self._block_type(r)
                else_pc, end = layout[pc]
                cond = stack.pop()
                if cond:
                    labels.append(Label(False, len(stack) - nparams, nres, end + 1))
                elif else_pc is not None:
                    labels.append(Label(False, len(stack) - nparams, nres, end + 1))
                    r.pos = else_pc + 1
                else:
                    r.pos = end + 1
            elif op == 0x05:
                # reached the end of a then-branch: skip the else-branch
                r.pos = labels[-1].target - 1  # the matching end, which pops the label
            elif op == 0x0B:
                if labels:
                    labels.pop()
                else:
                    return stack[len(stack) - nresults:] if nresults else []
            elif op == 0x0C:
                branch(r.u32())
            elif op == 0x0D:
                depth = r.u32()
                if stack.pop():
                    branch(depth)
            elif op == 0x0E:
                targets = [r.u32() for _ in range(r.u32())]
                default = r.u32()
                i = stack.pop()
                depth = targets[i] if i < len(targets) else default
                if depth == len(labels):  # branch to the function body = return
                    return stack[len(stack) - nresults:] if nresults else []
                branch(depth)
            elif op == 0x0F:
                return stack[len(stack) - nresults:] if nresults else []
            elif op == 0x10:
                callee = r.u32()
                ct = mod.types[mod.functions[callee].type_index]
                n = len(ct.params)
                call_args = stack[len(stack) - n:] if n else []
                del stack[len(stack) - n:]
                stack.extend(self.call(callee, call_args))
            elif op == 0x1A:
                stack.pop()
            elif op == 0x1B:
                c = stack.pop()
                b = stack.pop()
                a = stack.pop()
                stack.append(a if c else b)
            elif op == 0x01:
                pass
            elif op == 0x00:
                raise Trap("unreachable")
            else:
                raise NotImplementedError(f"opcode 0x{op:02x} at {pc}")
