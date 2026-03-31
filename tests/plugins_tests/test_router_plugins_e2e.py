# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end tests for the vllm.router_plugins entry point group.

Requires:
  - A GPU (vllm starts a real inference server)
  - The dummy plugin installed:
      pip install -e tests/plugins/vllm_add_dummy_router/

The dummy plugin registers:
  GET  /dummy/ping  → {"pong": true}
  POST /dummy/echo  → echoes the JSON body back

Tests:
  1. Routes added by a router plugin are reachable on the live server.
  2. Routes appear in the OpenAPI schema (/openapi.json).
  3. Core vLLM routes (/health, /v1/models) are unaffected.
  4. Server starts cleanly with NO router plugins installed.
"""

import pytest
import requests

from tests.utils import RemoteOpenAIServer

# Smallest model that starts quickly; enforce-eager skips CUDA graph capture.
MODEL = "facebook/opt-125m"
BASE_ARGS = [
    "--dtype",
    "float16",
    "--max-model-len",
    "512",
    "--enforce-eager",
    "--max-num-seqs",
    "4",
]


@pytest.fixture(scope="module")
def server_with_plugin():
    """vLLM server started with the dummy_router plugin enabled."""
    with RemoteOpenAIServer(
        MODEL,
        BASE_ARGS,
        env_dict={"VLLM_PLUGINS": "dummy_router"},
    ) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_no_plugins():
    """vLLM server started with no router plugins."""
    with RemoteOpenAIServer(
        MODEL,
        BASE_ARGS,
        env_dict={"VLLM_PLUGINS": ""},
    ) as remote_server:
        yield remote_server


# ---------------------------------------------------------------------------
# Tests: server with the dummy_router plugin
# ---------------------------------------------------------------------------


def test_plugin_ping_route(server_with_plugin: RemoteOpenAIServer):
    """GET /dummy/ping returns 200 and {"pong": true}."""
    resp = requests.get(server_with_plugin.url_for("dummy", "ping"), timeout=10)
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}


def test_plugin_echo_route(server_with_plugin: RemoteOpenAIServer):
    """POST /dummy/echo returns the submitted body unchanged."""
    payload = {"hello": "world", "count": 42}
    resp = requests.post(
        server_with_plugin.url_for("dummy", "echo"),
        json=payload,
        timeout=10,
    )
    assert resp.status_code == 200
    assert resp.json() == payload


def test_plugin_routes_in_openapi_schema(server_with_plugin: RemoteOpenAIServer):
    """Plugin routes appear in /openapi.json so they are discoverable by clients."""
    resp = requests.get(server_with_plugin.url_for("openapi.json"), timeout=10)
    assert resp.status_code == 200
    paths = resp.json().get("paths", {})
    assert "/dummy/ping" in paths, f"/dummy/ping missing from schema: {list(paths)}"
    assert "/dummy/echo" in paths, f"/dummy/echo missing from schema: {list(paths)}"


def test_core_routes_unaffected_by_plugin(server_with_plugin: RemoteOpenAIServer):
    """Core vLLM routes remain reachable when a router plugin is loaded."""
    health = requests.get(server_with_plugin.url_for("health"), timeout=10)
    assert health.status_code == 200

    models = requests.get(server_with_plugin.url_for("v1", "models"), timeout=10)
    assert models.status_code == 200
    data = models.json().get("data", [])
    assert any(MODEL in m.get("id", "") for m in data), (
        f"Model {MODEL!r} not listed: {data}"
    )


def test_completions_work_alongside_plugin(server_with_plugin: RemoteOpenAIServer):
    """/v1/completions still works correctly when a router plugin is active."""
    payload = {
        "model": MODEL,
        "prompt": "Hello,",
        "max_tokens": 5,
        "temperature": 0.0,
    }
    resp = requests.post(
        server_with_plugin.url_for("v1", "completions"),
        json=payload,
        timeout=30,
    )
    assert resp.status_code == 200
    choices = resp.json().get("choices", [])
    assert len(choices) == 1
    assert choices[0].get("text")


# ---------------------------------------------------------------------------
# Tests: server with NO router plugins
# ---------------------------------------------------------------------------


def test_no_plugins_server_starts_cleanly(server_no_plugins: RemoteOpenAIServer):
    """/health returns 200 when no router plugins are installed."""
    resp = requests.get(server_no_plugins.url_for("health"), timeout=10)
    assert resp.status_code == 200


def test_no_plugins_routes_absent(server_no_plugins: RemoteOpenAIServer):
    """Plugin routes are absent when no router plugins are loaded."""
    resp = requests.get(server_no_plugins.url_for("dummy", "ping"), timeout=10)
    assert resp.status_code == 404
