"""``GET /api/luts/groups`` – the lookup tables grouped for the menu and the Display section's select.

Answers ``{"groups": [{"name": "ImageJ", "luts": ["Grays", …]}, …], "aliases": {"mpl-viridis": "…", …}}``
(:data:`lumen.core.luts.LUT_GROUPS` filtered to the tables that exist, runtime-registered tables under
"Imported", and :data:`lumen.core.luts.LUT_ALIASES` explaining the near-duplicate names).  The browser falls
back to its own copy of the table when this router is missing (older server).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...core.luts import LUT_ALIASES, lut_groups

router = APIRouter()


@router.get("/luts/groups")  # mounted under /api by create_app
def groups() -> dict[str, Any]:
    return {"groups": lut_groups(), "aliases": dict(LUT_ALIASES)}
