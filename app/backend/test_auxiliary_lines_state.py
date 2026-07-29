from backend.state import AppState


def test_auxiliary_lines_defaults_empty():
    s = AppState()
    assert s.auxiliary_lines == []
