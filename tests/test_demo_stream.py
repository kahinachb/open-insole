"""Serial capture and left/right pairing of the live demo."""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.stream import CAPTURE_MAGIC, CHUNK_HEADER, CaptureSource, Pairer, codec, raw_counts  # noqa: E402


def frame(identifier: int, forces=(1000, 1100, 1200, 1300)) -> bytes:
    body = bytes([identifier, identifier + 1, 0]) + struct.pack(">HHHH", *forces)
    return body + bytes(codec.TAILLE_TRAME - len(body))


def packet(first: int) -> bytes:
    data = b"".join(frame(first + 2 * index) for index in range(4))
    return data + struct.pack(">I", 0)


def test_pairer_pairs_in_arrival_order_and_drops_a_lone_foot():
    pairer = Pairer()
    assert pairer.push("L", 0.0, {"a": 1}) == []
    assert pairer.push("R", 0.004, {"b": 2}) == [(0.004, {"a": 1, "b": 2})]
    for index in range(20):
        pairer.push("L", float(index), {"a": index})
    assert pairer.dropped == 20 - 8


def test_bare_and_recorded_captures_give_the_same_counts(tmp_path):
    stream = (packet(1) + packet(9)) * 10
    bare = tmp_path / "dataSemelle.bin"
    bare.write_bytes(stream)
    recorded = tmp_path / "session.bin"
    recorded.write_bytes(CAPTURE_MAGIC + CHUNK_HEADER.pack(0.0, len(stream)) + stream)
    for path in (bare, recorded):
        counts = raw_counts(CaptureSource(path, speed=0), seconds=1e9)
        assert sorted(counts, key=int) == ["1", "3", "5", "7", "9", "11", "13", "15"]
        assert counts["13"][2] == [1200] * 10
