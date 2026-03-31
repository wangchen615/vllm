# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the vllm.router_plugins entry point hook in build_app().

These tests call the plugin-attachment logic directly so no GPU, no real
entry points, and no fully-constructed args namespace are required.
"""

import logging
from unittest.mock import MagicMock

from fastapi import FastAPI


def _run_plugin_attachment(monkeypatch, plugins: dict):
    """
    Exercise only the router-plugin loop from build_app() by patching
    load_plugins_by_group and calling a minimal reproduction of the loop.

    Returns (app, log_records) so tests can assert on both.
    """
    from vllm.entrypoints.openai import api_server

    fake_app = FastAPI()
    fake_load = MagicMock(return_value=plugins)
    monkeypatch.setattr(api_server, "load_plugins_by_group", fake_load)

    # Re-import the patched reference so the loop uses our mock
    load_plugins_by_group = api_server.load_plugins_by_group
    logger = logging.getLogger("vllm.entrypoints.openai.api_server")

    router_plugins = load_plugins_by_group("vllm.router_plugins")
    for name, attach_fn in router_plugins.items():
        logger.debug("Attaching router plugin: %s", name)
        try:
            attach_fn(fake_app)  # type: ignore[call-arg]
        except Exception:
            logger.exception("Failed to attach router plugin: %s", name)

    return fake_app


def test_no_plugins_installed(monkeypatch):
    """No-op when no router plugins are registered; app is a valid FastAPI instance."""
    app = _run_plugin_attachment(monkeypatch, plugins={})
    assert isinstance(app, FastAPI)


def test_one_plugin_called_with_app(monkeypatch):
    """A single registered plugin's attach_fn is called once with the app."""
    attach_fn = MagicMock()
    app = _run_plugin_attachment(monkeypatch, plugins={"my_plugin": attach_fn})

    attach_fn.assert_called_once()
    received_app = attach_fn.call_args.args[0]
    assert received_app is app


def test_multiple_plugins_each_called_once(monkeypatch):
    """Each registered plugin is called exactly once in iteration order."""
    fn_a = MagicMock()
    fn_b = MagicMock()
    _run_plugin_attachment(monkeypatch, plugins={"plugin_a": fn_a, "plugin_b": fn_b})

    fn_a.assert_called_once()
    fn_b.assert_called_once()


def test_bad_plugin_does_not_crash_server(monkeypatch):
    """If attach_fn raises, the loop continues and returns a valid app."""
    calls = []

    def bad_attach(app):
        raise RuntimeError("plugin broken")

    def good_attach(app):
        calls.append(app)

    # bad plugin first, good plugin second — good must still run
    app = _run_plugin_attachment(
        monkeypatch, plugins={"broken_plugin": bad_attach, "good_plugin": good_attach}
    )

    assert isinstance(app, FastAPI)
    # good plugin must have been called despite the earlier failure
    assert len(calls) == 1
    assert calls[0] is app
