"""Offline test harness: no network, no real tokens, nothing is ever posted."""
import os
import socket
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Make sure no real credentials from a developer shell leak into tests.
for _k in list(os.environ):
    if _k.startswith(("METACULUS", "OPENROUTER", "ASKNEWS", "OPENAI", "ANTHROPIC", "PERPLEXITY", "EXA", "LANTERN")):
        os.environ.pop(_k)
os.environ["METACULUS_API_BASE_URL"] = "http://127.0.0.1:9/api"  # unroutable belt-and-braces


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def guard(*a, **k):
        raise NetworkBlocked("Network access is blocked in lanternbot tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    yield
