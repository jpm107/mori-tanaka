import subprocess
import sys

import numpy as np

from mori_tanaka import effective_properties


def test_import_is_quiet_and_has_no_filesystem_side_effects(tmp_path):
    before = set(tmp_path.iterdir())
    completed = subprocess.run(
        [sys.executable, "-c", "import mori_tanaka"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert set(tmp_path.iterdir()) == before


def test_public_function_returns_results_without_printing(capsys):
    result = effective_properties(
        fiber_e1=3.8,
        fiber_nu12=0.4,
        matrix_e=3.8,
        matrix_nu=0.4,
        fiber_volume_fraction=0.2,
        aspect_ratio=20.0,
        orientation_a11=1.0 / 3.0,
        orientation_a22=1.0 / 3.0,
        closure="linear",
    )

    assert set(result) == {
        "E1",
        "E2",
        "E3",
        "nu12",
        "nu13",
        "nu23",
        "G12",
        "G13",
        "G23",
    }
    assert all(isinstance(value, float) for value in result.values())
    assert all(np.isfinite(value) for value in result.values())
    assert capsys.readouterr() == ("", "")
