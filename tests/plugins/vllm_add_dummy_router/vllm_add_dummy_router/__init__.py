# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dummy router plugin for testing vllm.router_plugins entry point group.

Registers two routes:
  GET  /dummy/ping  → {"pong": true}
  POST /dummy/echo  → echoes the JSON body back
"""

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/dummy", tags=["dummy-router-plugin"])


@router.get("/ping")
async def ping() -> JSONResponse:
    return JSONResponse({"pong": True})


@router.post("/echo")
async def echo(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(body)


def attach_router(app: FastAPI) -> None:
    """Entry point called by vLLM build_app() via vllm.router_plugins."""
    app.include_router(router)
