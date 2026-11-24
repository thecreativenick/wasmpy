"""Tests build modules byte by byte with a tiny assembler, so no external tools are needed.

Once the text-format parser exists, these move to .wat sources and the
official .wast suites take over.
"""

import pytest

from wasmpy import DecodeError, Instance, Trap, decode_module

I32 = 0x7F
END, IF, ELSE, BLOCK, LOOP, BR, BR_IF, BR_TABLE, RETURN, CALL = 0x0B, 0x04, 0x05, 0x02, 0x03, 0x0C, 0x0D, 0x0E, 0x0F, 0x10
LGET, LSET, LTEE, CONST, DROP = 0x20, 0x21, 0x22, 0x41, 0x1A
EQZ, LT_S, ADD, SUB, MUL, DIV_S, REM_S, SHR_S, ROTL, CLZ = 0x45, 0x48, 0x6A, 0x6B, 0x6C, 0x6D, 0x6F, 0x75, 0x77, 0x67


def uleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def sleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        done = (n == 0 and not b & 0x40) or (n == -1 and b & 0x40)
        out.append(b if done else b | 0x80)
        if done:
            return bytes(out)


def vec(items):
    return uleb(len(items)) + b"".join(items)


def section(sid, payload):
    return bytes([sid]) + uleb(len(payload)) + payload


def code(*parts):
    """Flatten ints / bytes into instruction bytes; ('c', n) is i32.const n."""
    out = bytearray()
    for p in parts:
        if isinstance(p, tuple):
            out += bytes([CONST]) + sleb(p[1])
        elif isinstance(p, int):
            out.append(p)
        else:
            out += p
    return bytes(out)


def build(funcs, exports):
    """funcs: list of (params, results, n_extra_i32_locals, body). Each gets its own type."""
    types = [b"\x60" + vec([bytes([t]) for t in p]) + vec([bytes([t]) for t in r]) for p, r, _, _ in funcs]
    func_sec = vec([uleb(i) for i in range(len(funcs))])
    bodies = []
    for _, _, nlocals, body in funcs:
        locals_ = vec([uleb(nlocals) + bytes([I32])]) if nlocals else vec([])
        b = locals_ + body
        bodies.append(uleb(len(b)) + b)
    exp = vec([uleb(len(n.encode())) + n.encode() + b"\x00" + uleb(i) for n, i in exports])
    return (b"\0asm\x01\0\0\0" + section(1, vec(types)) + section(3, func_sec)
            + section(7, exp) + section(10, vec(bodies)))


def instance(funcs, exports):
    return Instance(decode_module(build(funcs, exports)))


def test_add_and_wraparound():
    inst = instance([([I32, I32], [I32], 0, code(LGET, 0, LGET, 1, ADD, END))], [("add", 0)])
    assert inst.invoke("add", 2, 3) == [5]
    assert inst.invoke("add", 0x7FFFFFFF, 1) == [-(2 ** 31)]
    assert inst.invoke("add", -5, 3) == [-2]


def test_recursive_factorial_with_if_else():
    body = code(LGET, 0, EQZ, IF, I32, ("c", 1), ELSE,
                LGET, 0, LGET, 0, ("c", 1), SUB, CALL, 0, MUL, END, END)
    inst = instance([([I32], [I32], 0, body)], [("fac", 0)])
    assert [inst.invoke("fac", n)[0] for n in range(6)] == [1, 1, 2, 6, 24, 120]
    assert inst.invoke("fac", 13) == [1932053504]  # 13! mod 2^32, as a signed i32


def test_loop_with_br_if():
    # sum = 0; while n != 0 { sum += n; n -= 1 } return sum
    body = code(BLOCK, 0x40, LOOP, 0x40,
                LGET, 0, EQZ, BR_IF, 1,
                LGET, 1, LGET, 0, ADD, LSET, 1,
                LGET, 0, ("c", 1), SUB, LSET, 0,
                BR, 0, END, END, LGET, 1, END)
    inst = instance([([I32], [I32], 1, body)], [("sum", 0)])
    assert inst.invoke("sum", 100) == [5050]
    assert inst.invoke("sum", 0) == [0]


def test_br_table_switch():
    # switch (x) { case 0: 100; case 1: 200; default: 300 }
    body = code(BLOCK, 0x40, BLOCK, 0x40, BLOCK, 0x40,
                LGET, 0, BR_TABLE, 2, 0, 1, 2, END,
                ("c", 100), RETURN, END, ("c", 200), RETURN, END, ("c", 300), END)
    inst = instance([([I32], [I32], 0, body)], [("switch", 0)])
    assert [inst.invoke("switch", x)[0] for x in (0, 1, 2, 7, -1)] == [100, 200, 300, 300, 300]


def test_block_result_and_tee():
    # block (result i32): 7 -> tee local0; br_if not taken; drop; 9   then + local0 = 16
    body = code(BLOCK, I32, ("c", 7), LTEE, 0, ("c", 0), BR_IF, 0, DROP, ("c", 9), END,
                LGET, 0, ADD, END)
    inst = instance([([], [I32], 1, body)], [("f", 0)])
    assert inst.invoke("f") == [16]


def test_signed_ops():
    div = instance([([I32, I32], [I32], 0, code(LGET, 0, LGET, 1, DIV_S, END)),
                    ([I32, I32], [I32], 0, code(LGET, 0, LGET, 1, REM_S, END)),
                    ([I32, I32], [I32], 0, code(LGET, 0, LGET, 1, SHR_S, END)),
                    ([I32, I32], [I32], 0, code(LGET, 0, LGET, 1, ROTL, END)),
                    ([I32], [I32], 0, code(LGET, 0, CLZ, END))],
                   [("div", 0), ("rem", 1), ("shr", 2), ("rotl", 3), ("clz", 4)])
    assert div.invoke("div", -7, 2) == [-3]       # truncates toward zero
    assert div.invoke("rem", -7, 2) == [-1]       # sign follows the dividend
    assert div.invoke("shr", -8, 1) == [-4]
    assert div.invoke("rotl", 0x80000001, 1) == [3]
    assert div.invoke("clz", 1) == [31] and div.invoke("clz", 0) == [32]
    with pytest.raises(Trap, match="divide by zero"):
        div.invoke("div", 1, 0)
    with pytest.raises(Trap, match="overflow"):
        div.invoke("div", -(2 ** 31), -1)


def test_unbounded_recursion_traps():
    inst = instance([([], [], 0, code(CALL, 0, END))], [("f", 0)])
    with pytest.raises(Trap, match="call stack exhausted"):
        inst.invoke("f")


@pytest.mark.parametrize("data,msg", [
    (b"\0asx\x01\0\0\0", "magic"),
    (b"\0asm\x02\0\0\0", "version"),
    (b"\0asm\x01\0\0\0" + section(1, b"\x01\x60\x00\x00")[:-1], "unexpected end|size"),
    (b"\0asm\x01\0\0\0" + b"\x01\x85\x80\x80\x80\x10" + b"\x00", "too large|too long"),
    (b"\0asm\x01\0\0\0" + section(3, vec([uleb(0)])), "inconsistent"),
    (b"\0asm\x01\0\0\0" + section(3, vec([])) + section(1, vec([])), "unexpected section"),
])
def test_malformed_modules(data, msg):
    with pytest.raises(DecodeError, match=msg):
        decode_module(data)
