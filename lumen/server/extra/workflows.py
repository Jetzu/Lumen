"""Guided workflows (drop-in router).  Definitions live in ``lumen.core.workflows``.

``GET /api/workflows``                 list: id, title, description, requires, tags, step counts, ops
``GET /api/workflows/{id}``            the full definition, every step with op_name / op_kind / op_preview
``POST /api/workflows/{id}/script``    a Python script replaying the workflow (``lumen.run`` lines like the
                                       recorder).  Body: ``{"params": {step_id: {name: value}},
                                       "skip": [step_id, ...], "include_optional": true}`` – all optional.
``GET /api/workflows/{id}/script``     the same with defaults (handy for curl / docs).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...core import workflows as wfmod

router = APIRouter()


class ScriptBody(BaseModel):
    params: dict[str, dict[str, Any]] | None = None
    skip: list[str] | None = None
    include_optional: bool = True


def _get(workflow_id: str) -> dict[str, Any]:
    try:
        return wfmod.get_workflow(workflow_id)
    except KeyError:
        raise HTTPException(404, f"Unknown workflow {workflow_id!r}; one of {', '.join(w['id'] for w in wfmod.WORKFLOWS)}")


@router.get("/workflows")
def list_workflows() -> list[dict[str, Any]]:
    return wfmod.summaries()


@router.get("/workflows/{workflow_id}")
def get_workflow(workflow_id: str) -> dict[str, Any]:
    return wfmod.with_op_info(_get(workflow_id))


def _script(workflow_id: str, body: ScriptBody | None) -> dict[str, Any]:
    _get(workflow_id)
    body = body or ScriptBody()
    wf = wfmod.get_workflow(workflow_id)
    known = {s["id"] for s in wf["steps"]}
    for sid in list((body.params or {}).keys()) + list(body.skip or []):
        if sid not in known:
            raise HTTPException(400, f"Workflow {workflow_id!r} has no step {sid!r}")
    try:
        return wfmod.build_script(workflow_id, body.params, body.skip, body.include_optional)
    except ValueError as e:  # a parameter override the op's Param rejects
        raise HTTPException(400, str(e))


@router.post("/workflows/{workflow_id}/script")
def workflow_script(workflow_id: str, body: ScriptBody | None = None) -> dict[str, Any]:
    return _script(workflow_id, body)


@router.get("/workflows/{workflow_id}/script")
def workflow_script_defaults(workflow_id: str, include_optional: bool = True) -> dict[str, Any]:
    return _script(workflow_id, ScriptBody(include_optional=include_optional))
