"""pytest wiring for the ImageJ cross-validation: opt-in marker, environment discovery, clean skips."""
from __future__ import annotations

import os

import pytest

from . import harness

OPT_IN_REASON = ("ImageJ cross-validation is opt-in (it starts a JVM per case): run with -m crossval, "
                 "LUMEN_CROSSVAL=1 or `pytest tests/crossval`")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "crossval: cross-validation against ImageJ 1.54p (needs java + ij.jar; opt-in, see docs/CROSSVALIDATION.md)")


def _opted_in(config: pytest.Config) -> bool:
    if os.environ.get("LUMEN_CROSSVAL", "").strip() == "1":
        return True
    expr = str(config.getoption("markexpr", "") or "")
    if "crossval" in expr and "not crossval" not in expr:
        return True
    return any("crossval" in str(a) for a in config.invocation_params.args)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _opted_in(config):
        return
    skip = pytest.mark.skip(reason=OPT_IN_REASON)
    for item in items:
        if "crossval" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def crossval_env(tmp_path_factory: pytest.TempPathFactory) -> harness.Environment:
    env = harness.Environment.discover(tmp_path_factory.mktemp("crossval"))
    if env is None:
        pytest.skip("java (JAVA_HOME / Homebrew OpenJDK 21 / PATH) or the ImageJ jar (LUMEN_IJ_JAR, default "
                    f"{harness.DEFAULT_JAR}) not found")
    return env
