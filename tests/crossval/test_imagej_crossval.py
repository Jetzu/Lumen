"""Every macro in tests/crossval/macros, run by ImageJ 1.54p and by Lumen, must agree within the recorded tolerance.

Known differences are recorded in ``harness.EXPECT`` (``expect="mismatch"``) and listed in docs/CROSSVALIDATION.md;
they xfail here and *fail* once they stop reproducing, so the table has to be updated when a defect is fixed.
"""
from __future__ import annotations

import pytest

from . import harness

CASES = harness.load_cases()

pytestmark = pytest.mark.crossval


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_against_imagej(case: harness.Case, crossval_env: harness.Environment, tmp_path) -> None:
    out = harness.run_case(case, crossval_env, tmp_path)
    msg = f"{case.id} [{case.title}] → {out.status}: {out.verdict}"
    if case.expect == "mismatch":
        if out.status == "mismatch":
            pytest.xfail(f"known difference (docs/CROSSVALIDATION.md): {case.note} | now: {out.verdict}")
        pytest.fail(f"{case.id} no longer mismatches ({out.status}: {out.verdict}) – promote it in tests/crossval/harness.py "
                    "EXPECT and docs/CROSSVALIDATION.md")
    assert out.status != "mismatch", msg
    if case.expect == "exact":
        assert out.status == "exact", msg


def test_every_case_has_a_macro_header() -> None:
    assert len(CASES) >= 25
    assert all(c.macro_path.is_file() for c in CASES)
    assert len({c.id for c in CASES}) == len(CASES)
