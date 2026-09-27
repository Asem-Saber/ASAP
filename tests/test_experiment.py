import pytest

from asap.config import get_settings
from asap.training.experiment import (
    Variant,
    load_variants,
    select_best,
    variant_out_dir,
)


def test_all_three_variants_load():
    variants = load_variants()

    assert set(variants) == {"baseline", "mlm", "arabic-base"}
    assert isinstance(variants["baseline"], Variant)


def test_variants_differ_only_by_base_model():
    """Each delta must be attributable to the encoder alone, so everything
    else has to be identical across the matrix."""
    variants = load_variants()

    assert len({v.learning_rate for v in variants.values()}) == 1
    assert len({v.epochs for v in variants.values()}) == 1
    assert len({v.base_model for v in variants.values()}) == 3


def test_unknown_key_in_the_yaml_fails_loudly(tmp_path):
    """A typo must not be silently ignored into a default."""
    path = tmp_path / "experiments.yml"
    path.write_text(
        "variants:\n  x:\n    base_model: m\n    learning_rate: 1.0e-5\n"
        "    epochs: 5\n    lr: 3.0e-5\n"
    )

    with pytest.raises(TypeError):
        load_variants(path)


def test_empty_yaml_loads_as_no_variants(tmp_path):
    path = tmp_path / "experiments.yml"
    path.write_text("")

    assert load_variants(path) == {}


def test_variant_out_dir_is_never_the_served_checkpoint():
    """models/cls is what the API serves. Three variants run in sequence would
    otherwise each overwrite the last, and the last would overwrite serving."""
    cfg = get_settings()

    for name in load_variants():
        out = variant_out_dir(name, settings=cfg)
        assert out != cfg.paths.cls_dir
        assert out.parent == cfg.paths.experiments_dir

    assert len({variant_out_dir(n) for n in load_variants()}) == 3


def test_select_best_maximises_when_greater_is_better():
    scores = {"baseline": 0.81, "mlm": 0.94, "arabic-base": 0.88}

    assert select_best(scores, greater_is_better=True) == "mlm"


def test_select_best_minimises_when_lower_is_better():
    scores = {"baseline": 0.81, "mlm": 0.94, "arabic-base": 0.88}

    assert select_best(scores, greater_is_better=False) == "baseline"


def test_select_best_rejects_an_empty_matrix():
    with pytest.raises(ValueError, match="no variant"):
        select_best({}, greater_is_better=True)


def test_experiment_module_stays_importable_without_heavy_deps():
    """This module is the CI-testable half of the runner. If it ever imports
    torch, datasets or mlflow, these tests stop running on the runner."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asap.training.experiment, sys; "
            "heavy = {'torch', 'datasets', 'mlflow', 'sklearn'} & set(sys.modules); "
            "sys.exit(f'pulled in {heavy}' if heavy else 0)",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
