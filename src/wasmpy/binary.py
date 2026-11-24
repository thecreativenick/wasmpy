"""WebAssembly binary-format decoder (core spec, chapter 5): module header, LEB128,
and the type / function / export / code sections.

Other sections (import, table, memory, global, start, element, data) are
recorded as raw bytes for now and decoded in later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAGIC = b"\0asm"
VERSION = b"\x01\0\0\0"
VALTYPES = {0x7F: "i32", 0x7E: "i64", 0x7D: "f32", 0x7C: "f64", 0x70: "funcref", 0x6F: "externref"}
SECTION_NAMES = {0: "custom", 1: "type", 2: "import", 3: "function", 4: "table", 5: "memory",
                 6: "global", 7: "export", 8: "start", 9: "element", 10: "code", 11: "data",
                 12: "datacount"}
SECTION_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 10, 11]  # datacount sits before code


class DecodeError(ValueError):
    pass


class Reader:
    def __init__(self, data: bytes, pos: int = 0, end: int | None = None):
        self.data, self.pos = data, pos
        self.end = len(data) if end is None else end

    def eof(self) -> bool:
        return self.pos >= self.end

    def byte(self) -> int:
        if self.pos >= self.end:
            raise DecodeError("unexpected end")
        b = self.data[self.pos]
        self.pos += 1
        return b

    def bytes(self, n: int) -> bytes:
        if self.pos + n > self.end:
            raise DecodeError("unexpected end")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return self._leb(32, signed=False)

    def s32(self) -> int:
        return self._leb(32, signed=True)

    def s33(self) -> int:
        return self._leb(33, signed=True)

    def s64(self) -> int:
        return self._leb(64, signed=True)

    def _leb(self, bits: int, signed: bool) -> int:
        result = shift = 0
        max_bytes = (bits + 6) // 7
        for i in range(max_bytes):
            b = self.byte()
            result |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                if i == max_bytes - 1:  # unused bits of the last byte must be a sign/zero extension
                    used = bits - 7 * (max_bytes - 1)
                    extra = b >> used if not signed else (b & 0x7F) >> (used - 1)
                    allowed = (0,) if not signed else (0, (0x7F >> (used - 1)))
                    if extra not in allowed:
                        raise DecodeError("integer too large")
                if signed and b & 0x40:
                    result -= 1 << shift
                return result
        raise DecodeError("integer representation too long")

    def name(self) -> str:
        try:
            return self.bytes(self.u32()).decode("utf-8")
        except UnicodeDecodeError as e:
            raise DecodeError("malformed UTF-8 encoding") from e

    def valtype(self) -> str:
        b = self.byte()
        if b not in VALTYPES:
            raise DecodeError(f"invalid value type 0x{b:02x}")
        return VALTYPES[b]


@dataclass(frozen=True)
class FuncType:
    params: tuple[str, ...]
    results: tuple[str, ...]


@dataclass
class Function:
    type_index: int
    locals: list[str] = field(default_factory=list)
    body: bytes = b""          # instruction bytes, ending with the final 0x0B (end)


@dataclass
class Module:
    types: list[FuncType] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    exports: dict[str, tuple[str, int]] = field(default_factory=dict)  # name -> (kind, index)
    raw_sections: dict[str, list[bytes]] = field(default_factory=dict)


def decode_module(data: bytes) -> Module:
    if data[:4] != MAGIC:
        raise DecodeError("magic header not detected")
    if data[4:8] != VERSION:
        raise DecodeError("unknown binary version")
    r = Reader(data, 8)
    mod = Module()
    func_types: list[int] = []
    last_rank = -1
    while not r.eof():
        sid = r.byte()
        size = r.u32()
        start = r.pos
        if start + size > len(data):
            raise DecodeError("section size mismatch")
        s = Reader(data, start, start + size)
        if sid != 0:
            if sid not in SECTION_ORDER or SECTION_ORDER.index(sid) <= last_rank:
                raise DecodeError(f"unexpected section id {sid}")
            last_rank = SECTION_ORDER.index(sid)
        if sid == 1:
            for _ in range(s.u32()):
                if s.byte() != 0x60:
                    raise DecodeError("malformed function type")
                params = tuple(s.valtype() for _ in range(s.u32()))
                results = tuple(s.valtype() for _ in range(s.u32()))
                mod.types.append(FuncType(params, results))
        elif sid == 3:
            func_types = [s.u32() for _ in range(s.u32())]
        elif sid == 7:
            kinds = {0: "func", 1: "table", 2: "memory", 3: "global"}
            for _ in range(s.u32()):
                name = s.name()
                kind = s.byte()
                if kind not in kinds:
                    raise DecodeError("malformed export kind")
                if name in mod.exports:
                    raise DecodeError("duplicate export name")
                mod.exports[name] = (kinds[kind], s.u32())
        elif sid == 10:
            count = s.u32()
            if count != len(func_types):
                raise DecodeError("function and code section have inconsistent lengths")
            for ti in func_types:
                body_size = s.u32()
                body_end = s.pos + body_size
                b = Reader(data, s.pos, body_end)
                local_types: list[str] = []
                for _ in range(b.u32()):
                    n = b.u32()
                    t = b.valtype()
                    if len(local_types) + n > 50_000:
                        raise DecodeError("too many locals")
                    local_types += [t] * n
                mod.functions.append(Function(ti, local_types, data[b.pos : body_end]))
                s.pos = body_end
        else:
            mod.raw_sections.setdefault(SECTION_NAMES.get(sid, "custom"), []).append(data[start : start + size])
            s.pos = s.end
        if s.pos != s.end:
            raise DecodeError("section size mismatch")
        r.pos = start + size
    if func_types and not mod.functions:
        raise DecodeError("function and code section have inconsistent lengths")
    return mod
