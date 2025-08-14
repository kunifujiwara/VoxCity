import numpy as np

from voxcity.simulator.solar import _configure_num_threads, _auto_time_batch_size


def test_configure_num_threads_smoke():
    # Should not raise and return None; respects boundaries
    _configure_num_threads(desired_threads=1, progress=False)


def test_auto_time_batch_size_bounds():
    # For small face counts, batch size should be at least 1 and not exceed steps
    batch = _auto_time_batch_size(n_faces=10, total_steps=100, user_value=None)
    assert 1 <= batch <= 100

