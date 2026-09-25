from pathlib import Path

from asap.config import get_settings


def test_tracking_is_disabled_by_default():
    """CI, the test suite and the serving path must be untouched by this
    milestone unless tracking is explicitly switched on."""
    assert get_settings().tracking.enabled is False


def test_tracking_config_is_typed():
    t = get_settings().tracking
    assert t.experiment == "asap-sentiment"
    assert t.registered_model == "asap-sentiment-clf"
    assert t.primary_metric == "test_f1"
    assert t.greater_is_better is True
    assert isinstance(t.artifact_dir, Path)


def test_experiments_dir_is_separate_from_the_served_checkpoint():
    """A variant run must never overwrite what the API serves."""
    paths = get_settings().paths
    assert paths.experiments_dir == Path("models/experiments")
    assert paths.experiments_dir != paths.cls_dir
