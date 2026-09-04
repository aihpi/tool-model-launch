"""Site branding: user-visible strings default to the upstream wording and follow
the SML_* variables when a site sets them. The module reads the environment at
import time, so the tests reload it."""

import importlib

import pytest

from swiss_ai_model_launch import site


def _reload() -> None:
    importlib.reload(site)


def test_defaults_are_upstream_wording(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SML_SITE_NAME", "SML_HEALTH_KEY_NAME", "SML_HEALTH_KEY_HELP"):
        monkeypatch.delenv(var, raising=False)
    _reload()
    assert site.APP_TITLE == "SwissAI Model Launch"
    assert site.HEALTH_KEY_PROMPT == "What is your Swiss AI Research API Key?"
    assert "serving.swissai.svc.cscs.ch" in site.HEALTH_KEY_INTRO


def test_site_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_SITE_NAME", "HPI AISC")
    monkeypatch.setenv("SML_HEALTH_KEY_NAME", "LiteLLM API key")
    monkeypatch.setenv("SML_HEALTH_KEY_HELP", "Ask the AISC team.")
    _reload()
    assert site.APP_TITLE == "HPI AISC Model Launch"
    assert site.HEALTH_KEY_PROMPT == "What is your LiteLLM API key?"
    assert site.HEALTH_KEY_INTRO.endswith("Ask the AISC team.\n")
    assert "cscs" not in site.HEALTH_KEY_INTRO
    _reload_defaults(monkeypatch)


def _reload_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SML_SITE_NAME", "SML_HEALTH_KEY_NAME", "SML_HEALTH_KEY_HELP"):
        monkeypatch.delenv(var, raising=False)
    _reload()


def test_launcher_question_resolves_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from swiss_ai_model_launch.cli.configuration.init_wizard import InitConfig

    monkeypatch.setenv("SML_LAUNCHER", "slurm")
    cfg = InitConfig()
    launcher = cfg.chain[0].head_configuration  # type: ignore[attr-defined]
    assert launcher.env_var == "SML_LAUNCHER"
    assert launcher._try_resolve_without_prompt(None) == "slurm"
