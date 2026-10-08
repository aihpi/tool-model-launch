"""Recipe conveniences applied to argv before parsing (cli/recipes.py): the site
file from SML_SITE_ARGS, --recipe NAME on SML_RECIPE_PATH, ~ in @file paths, the
`sml recipes` listing, and SML_ENVIRONMENT as the --environment default. With
none of the variables set, argv must pass through untouched."""

import argparse
from pathlib import Path

import pytest

from swiss_ai_model_launch.cli.main import _build_parser, _parse_cli
from swiss_ai_model_launch.cli.recipes import (
    RecipeError,
    expand_argv,
    format_recipe_table,
    list_recipes,
    resolve_recipe,
)


@pytest.fixture
def recipes(tmp_path: Path) -> dict[str, Path]:
    site = tmp_path / "site" / "_site.args"
    site.parent.mkdir()
    site.write_text("--no-exclusive\n--mem 48G\n--sbatch-arg=--exclude=ga03\n")
    personal = tmp_path / "personal"
    shared = tmp_path / "shared"
    personal.mkdir()
    shared.mkdir()
    (shared / "qwen.args").write_text("# Qwen from the share\n--framework vllm\n")
    (shared / "gemma.args").write_text("# Gemma\n--framework vllm\n")
    (shared / "_part.args").write_text("--disable-metrics\n")
    (personal / "qwen.args").write_text("\n# My own Qwen\n--framework sglang\n")
    return {"site": site, "personal": personal, "shared": shared}


def _env(r: dict[str, Path], **extra: str) -> dict[str, str]:
    return {"SML_SITE_ARGS": str(r["site"]), "SML_RECIPE_PATH": f"{r['personal']}:{r['shared']}", **extra}


# ── argv expansion ───────────────────────────────────────────────────────────


def test_no_variables_means_no_change() -> None:
    argv = ["advanced", "--framework", "vllm", "@x.args"]
    assert expand_argv(argv, {}) == argv


def test_other_subcommands_are_untouched(recipes: dict[str, Path]) -> None:
    for argv in (["init"], ["recipes"], ["loadtest", "--recipe", "qwen"], []):
        assert expand_argv(argv, _env(recipes)) == argv


def test_site_file_goes_first_and_recipe_resolves_personal_before_shared(recipes: dict[str, Path]) -> None:
    out = expand_argv(["advanced", "--recipe", "qwen", "--mem", "8G"], _env(recipes))
    assert out == ["advanced", f"@{recipes['site']}", f"@{recipes['personal'] / 'qwen.args'}", "--mem", "8G"]


def test_recipe_equals_form_and_explicit_suffix(recipes: dict[str, Path]) -> None:
    out = expand_argv(["advanced", "--recipe=gemma.args"], _env(recipes))
    assert out[-1] == f"@{recipes['shared'] / 'gemma.args'}"


def test_recipe_given_as_a_path(recipes: dict[str, Path]) -> None:
    path = recipes["shared"] / "gemma.args"
    assert expand_argv(["advanced", "--recipe", str(path)], {})[-1] == f"@{path}"
    with pytest.raises(RecipeError, match="not found"):
        expand_argv(["advanced", "--recipe", str(recipes["shared"] / "nope.args")], {})


def test_unknown_recipe_names_the_searched_dirs(recipes: dict[str, Path]) -> None:
    with pytest.raises(RecipeError, match="Unknown recipe 'llama'.*personal.*shared"):
        expand_argv(["advanced", "--recipe", "llama"], _env(recipes))
    with pytest.raises(RecipeError, match="needs a name"):
        expand_argv(["advanced", "--recipe"], _env(recipes))


def test_no_site_args_skips_the_site_file(recipes: dict[str, Path]) -> None:
    out = expand_argv(["advanced", "--no-site-args", "--recipe", "gemma"], _env(recipes))
    assert out == ["advanced", f"@{recipes['shared'] / 'gemma.args'}"]


def test_missing_site_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="SML_SITE_ARGS"):
        expand_argv(["advanced"], {"SML_SITE_ARGS": str(tmp_path / "missing.args")})


def test_tilde_in_at_file_is_expanded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert expand_argv(["advanced", "@~/r.args"], {}) == ["advanced", f"@{tmp_path}/r.args"]


# ── through the real parser ──────────────────────────────────────────────────


def test_flags_after_the_recipe_override_site_and_recipe(
    recipes: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in _env(recipes).items():
        monkeypatch.setenv(key, value)
    parser = _build_parser()
    args = _parse_cli(parser, ["advanced", "--environment", "e.toml", "--recipe", "gemma", "--mem", "8G"])
    assert args.framework == "vllm"
    assert args.mem == "8G"  # site file said 48G
    assert args.exclusive is False  # from the site file
    assert args.sbatch_args == ["--exclude=ga03"]


def test_recipe_inside_an_at_file_is_rejected(tmp_path: Path) -> None:
    nested = tmp_path / "nested.args"
    nested.write_text("--recipe other\n")
    with pytest.raises(SystemExit):
        _parse_cli(_build_parser(), ["advanced", "--environment", "e.toml", "--framework", "vllm", f"@{nested}"])


def test_sml_environment_makes_environment_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SML_ENVIRONMENT", raising=False)
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["advanced", "--framework", "vllm"])
    monkeypatch.setenv("SML_ENVIRONMENT", "/site/env.toml")
    args = _build_parser().parse_args(["advanced", "--framework", "vllm"])
    assert args.slurm_environment == "/site/env.toml"
    args = _build_parser().parse_args(["advanced", "--framework", "vllm", "--environment", "mine.toml"])
    assert args.slurm_environment == "mine.toml"


# ── listing ──────────────────────────────────────────────────────────────────


def test_listing_first_hit_wins_and_hides_building_blocks(recipes: dict[str, Path]) -> None:
    listed = list_recipes([recipes["personal"], recipes["shared"], recipes["site"].parent])
    assert [r.name for r in listed] == ["qwen", "gemma"]
    assert listed[0].description == "My own Qwen" and listed[0].path.parent == recipes["personal"]
    table = format_recipe_table(listed)
    assert "qwen" in table and "Gemma" in table and "--recipe <NAME>" in table


def test_listing_tolerates_missing_dirs_and_reports_empty(tmp_path: Path) -> None:
    assert list_recipes([tmp_path / "absent"]) == []
    assert "No recipes found" in format_recipe_table([])


def test_resolve_prefers_name_with_suffix(tmp_path: Path) -> None:
    (tmp_path / "x").write_text("--framework vllm\n")
    (tmp_path / "x.args").write_text("--framework sglang\n")
    assert resolve_recipe("x", [tmp_path]) == tmp_path / "x.args"


def test_recipes_subcommand_parses() -> None:
    assert isinstance(_build_parser().parse_args(["recipes"]), argparse.Namespace)
