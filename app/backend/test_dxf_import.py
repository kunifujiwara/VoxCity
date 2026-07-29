import io

import ezdxf
import pytest

from backend.dxf_import import parse_dxf, ParsedDxf


def _to_bytes(doc) -> bytes:
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def _doc(units: int | None = None):
    doc = ezdxf.new()
    if units is not None:
        doc.header["$INSUNITS"] = units
    return doc


def test_parses_line_as_two_point_polyline():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 5), dxfattribs={"layer": "walls"})
    parsed = parse_dxf(_to_bytes(doc))
    assert isinstance(parsed, ParsedDxf)
    layer = next(l for l in parsed.layers if l.name == "walls")
    assert layer.polylines == [[[0.0, 0.0], [10.0, 5.0]]]


def test_parses_open_and_closed_lwpolyline():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (1, 0), (1, 1)], close=False, dxfattribs={"layer": "a"})
    msp.add_lwpolyline([(0, 0), (2, 0), (2, 2)], close=True, dxfattribs={"layer": "a"})
    parsed = parse_dxf(_to_bytes(doc))
    a = next(l for l in parsed.layers if l.name == "a")
    assert a.polylines[0] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert a.polylines[1] == [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]


def test_parses_old_style_polyline_with_vertices():
    doc = _doc()
    msp = doc.modelspace()
    pl = msp.add_polyline2d([(0, 0), (3, 0), (3, 3)], dxfattribs={"layer": "window"})
    pl.close(True)
    parsed = parse_dxf(_to_bytes(doc))
    w = next(l for l in parsed.layers if l.name == "window")
    assert w.polylines[0] == [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 0.0]]


def test_groups_by_layer_in_first_seen_order():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "second"})
    msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "first"})
    msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "second"})
    parsed = parse_dxf(_to_bytes(doc))
    names = [l.name for l in parsed.layers]
    assert names == ["second", "first"]


def test_bounds_and_center():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_line((-2, -4), (6, 10), dxfattribs={"layer": "a"})
    parsed = parse_dxf(_to_bytes(doc))
    assert parsed.bounds == [[-2.0, -4.0], [6.0, 10.0]]
    assert parsed.center == [2.0, 3.0]


def test_color_is_hex_string():
    doc = _doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 1), dxfattribs={"layer": "a", "true_color": 0xFF8800})
    parsed = parse_dxf(_to_bytes(doc))
    a = next(l for l in parsed.layers if l.name == "a")
    assert a.color.lower() == "#ff8800"


@pytest.mark.parametrize(
    "code,expected",
    [(6, "m"), (5, "cm"), (4, "mm"), (2, "ft"), (1, "in"), (0, None)],
)
def test_detects_insunits(code, expected):
    doc = _doc(units=code)
    doc.modelspace().add_line((0, 0), (1, 1), dxfattribs={"layer": "a"})
    parsed = parse_dxf(_to_bytes(doc))
    assert parsed.detected_units == expected


def test_missing_insunits_is_none():
    doc = _doc()
    doc.modelspace().add_line((0, 0), (1, 1), dxfattribs={"layer": "a"})
    lines = _to_bytes(doc).decode("utf-8").splitlines()
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "$INSUNITS" and cleaned and cleaned[-1].strip() == "9":
            cleaned.pop()
            i += 3
            continue
        cleaned.append(lines[i])
        i += 1
    data = ("\n".join(cleaned) + "\n").encode("utf-8")
    parsed = parse_dxf(data)
    assert parsed.detected_units is None


def test_empty_document_has_no_layers_and_zero_inserts():
    parsed = parse_dxf(_to_bytes(_doc()))
    assert parsed.layers == []
    assert parsed.insert_count == 0


def test_counts_inserts_when_geometry_only_in_block():
    doc = _doc()
    blk = doc.blocks.new(name="B")
    blk.add_line((0, 0), (1, 1))
    doc.modelspace().add_blockref("B", (0, 0))
    parsed = parse_dxf(_to_bytes(doc))
    assert parsed.layers == []
    assert parsed.insert_count == 1


def test_malformed_input_raises_dxf_parse_error():
    from backend.dxf_import import DxfParseError
    with pytest.raises(DxfParseError):
        parse_dxf(b"this is not a dxf file")


import numpy as np

from backend.dxf_import import bake_polylines_to_lonlat


def test_bake_polylines_identity_grid():
    mat = np.eye(4)
    grid_geom = {
        "origin": [100.0, 50.0],
        "u_vec": [1.0, 0.0],
        "v_vec": [0.0, 1.0],
        "adj_mesh": [1.0, 1.0],
    }
    out = bake_polylines_to_lonlat([[[2.0, 3.0], [4.0, 5.0]]], mat, grid_geom)
    assert out == [[[102.0, 53.0], [104.0, 55.0]]]
