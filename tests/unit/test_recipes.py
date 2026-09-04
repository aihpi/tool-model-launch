"""`sml advanced @file`: flags read from recipe files (shell-style lines, quoted
values, comments), and the HPI recipes rendering to the same job shape as the
example scripts."""

from pathlib import Path

import pytest

from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.framework import render_master, render_rank_scripts

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECIPES = _REPO_ROOT / "hpi" / "recipes"
_SHARED_HPI = "/sc/projects/sci-aisc/aisc-share/sml/src/hpi"


def test_args_file_accepts_shell_style_lines(tmp_path: Path) -> None:
    recipe = tmp_path / "r.args"
    recipe.write_text(
        "# a comment\n"
        "--framework vllm --mem 48G\n"
        "\n"
        '--framework-args "--model Qwen/Qwen3-0.6B --served-model-name x"  # trailing comment\n'
        "--environment ~/env.toml\n"
    )
    args = _build_parser().parse_args(["advanced", f"@{recipe}"])
    assert args.framework == "vllm" and args.mem == "48G"
    assert args.framework_args == "--model Qwen/Qwen3-0.6B --served-model-name x"
    assert args.slurm_environment == "~/env.toml"


def test_several_files_and_explicit_flags_combine(tmp_path: Path) -> None:
    site = tmp_path / "site.args"
    site.write_text("--no-exclusive\n--framework-port auto\n")
    model = tmp_path / "model.args"
    model.write_text("--framework vllm\n--environment e.toml\n")
    args = _build_parser().parse_args(["advanced", f"@{site}", f"@{model}", "--mem", "8G"])
    assert args.exclusive is False and args.framework_port == "auto" and args.mem == "8G"


def _render_recipe(name: str, tmp_path: Path) -> dict[str, str]:
    # The recipes name the shared install's paths; point them at this checkout.
    files = []
    for stem in ("_site", name):
        text = (_RECIPES / f"{stem}.args").read_text().replace(_SHARED_HPI, str(_REPO_ROOT / "hpi"))
        f = tmp_path / f"{stem}.args"
        f.write_text(text)
        files.append(f"@{f}")
    args = _build_parser().parse_args(["advanced", *files])
    launch_args = build_launch_args_from_advanced(args, username="alice", account="aisc-staff", partition="aisc-batch")
    out = {"master.sh": render_master(launch_args)}
    out.update(render_rank_scripts(launch_args))
    return out


@pytest.mark.parametrize("name", ["qwen3-0.6b", "pool"])
def test_hpi_recipes_render_the_reference_job_shape(name: str, tmp_path: Path) -> None:
    out = _render_recipe(name, tmp_path)
    master, head = out["master.sh"], out["head.sh"]
    assert "SML_CONTAINER_ARGS=(" in master and "--environment=" not in master
    assert 'sml_enroot_data="/sc/projects/sci-aisc/aisc-share/enroot-data/$USER"' in master
    assert head.index('"$WSTUNNEL_BIN" client') < head.index("$OPENTELA_BIN start")
    assert "--service.port $FRAMEWORK_PORT" in head
    for content in out.values():
        assert "capstor" not in content and "cscs" not in content
