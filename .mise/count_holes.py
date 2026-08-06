#!/usr/bin/env python3
"""Count magenta (background) pixels per frame. Exit 1 if any hole pixel
appears anywhere in any frame (the eval camera looks steeply down from
altitude, so no sky is ever legitimately in frame)."""
import struct, sys, zlib

def load_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, w, h, idat = 8, 0, 0, b""
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos : pos + 8])
        chunk = data[pos + 8 : pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, color = struct.unpack(">IIBB", chunk[:10])
            assert depth == 8 and color in (2, 6), (path, depth, color)
            bpp = 3 if color == 2 else 4
        elif typ == b"IDAT":
            idat += chunk
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * bpp
    rows, prev = [], bytearray(stride)
    p = 0
    for _ in range(h):
        f = raw[p]; p += 1
        line = bytearray(raw[p : p + stride]); p += stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 1: line[i] = (line[i] + a) & 0xFF
            elif f == 2: line[i] = (line[i] + b) & 0xFF
            elif f == 3: line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif f == 4:
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        rows.append(bytes(line)); prev = line
    return w, h, bpp, rows

fail = False
for path in sys.argv[1:]:
    w, h, bpp, rows = load_png(path)
    holes = 0
    for y in range(h):
        row = rows[y]
        for x in range(w):
            r, g, b = row[x * bpp], row[x * bpp + 1], row[x * bpp + 2]
            if r > 200 and b > 200 and g < 60:
                holes += 1
    print(f"{path}: {holes} hole px")
    if holes > 0:
        fail = True
sys.exit(1 if fail else 0)
