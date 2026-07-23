"""parallel_download must default ON (user decision, Batch 4).

Source-introspection tests are blunt but honest here: the flag's default is
what matters, and mocking the whole pipeline to observe thread usage would be
far more brittle.
"""

import inspect


def test_api_default_parallel_download_true():
    from voxcity.generator import api

    src = inspect.getsource(api.get_voxcity)
    assert 'kwargs.get("parallel_download", True)' in src


def test_pipeline_default_parallel_download_true():
    from voxcity.generator import pipeline

    src = inspect.getsource(pipeline.VoxCityPipeline.run)
    assert "getattr(cfg, 'parallel_download', True)" in src
