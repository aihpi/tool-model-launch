"""Recipes: `sml advanced` flags kept in ``*.args`` files.

Three conveniences on top of argparse's ``@file`` support, all applied to argv
before parsing and all inert unless a site sets the variables:

* ``SML_SITE_ARGS``: a file of site constants prepended to every ``sml advanced``,
  so a recipe only carries what differs per model. ``--no-site-args`` skips it.
* ``--recipe NAME``: ``@<file>`` looked up by name on ``SML_RECIPE_PATH``
  (colon-separated directories, first hit wins); a value containing ``/`` is a path.
* ``@~/x.args``: ``~`` and ``$VAR`` in ``@file`` paths are expanded (the shell does
  not expand ``~`` after ``@``).

Flags later on the command line override earlier ones, so the order is: site
file, then the user's recipes and flags as written.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

RECIPE_SUFFIX = ".args"
_EXPANDED_SUBCOMMANDS = frozenset({"advanced"})


class RecipeError(ValueError):
    """A recipe or site file that cannot be found."""


@dataclass(frozen=True)
class Recipe:
    name: str
    path: Path
    description: str


def _expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def recipe_dirs(env: Mapping[str, str] | None = None) -> list[Path]:
    env = os.environ if env is None else env
    raw = env.get("SML_RECIPE_PATH", "")
    return [Path(_expand(part)) for part in raw.split(":") if part.strip()]


def resolve_recipe(name: str, dirs: list[Path]) -> Path:
    """Path of recipe ``name``: a literal path if it contains ``/``, else the first
    ``name`` or ``name.args`` on ``dirs``."""
    if "/" in name:
        path = Path(_expand(name))
        if path.is_file():
            return path
        raise RecipeError(f"Recipe file not found: {path}")
    candidates = [name] if name.endswith(RECIPE_SUFFIX) else [name + RECIPE_SUFFIX, name]
    for directory in dirs:
        for candidate in candidates:
            path = directory / candidate
            if path.is_file():
                return path
    searched = ", ".join(str(d) for d in dirs) or "(SML_RECIPE_PATH is empty)"
    raise RecipeError(f"Unknown recipe {name!r}; searched: {searched}. List them with `sml recipes`.")


def _description(path: Path) -> str:
    try:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            return ""
    except OSError:
        pass
    return ""


def list_recipes(dirs: list[Path]) -> list[Recipe]:
    """Recipes visible on ``dirs``, first hit per name wins; ``_*.args`` are
    building blocks (like the site file) and are not listed."""
    seen: dict[str, Recipe] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"*{RECIPE_SUFFIX}")):
            name = path.name.removesuffix(RECIPE_SUFFIX)
            if name.startswith("_") or name in seen:
                continue
            seen[name] = Recipe(name=name, path=path, description=_description(path))
    return list(seen.values())


def expand_argv(argv: list[str], env: Mapping[str, str] | None = None) -> list[str]:
    """Apply the site file, ``--recipe`` and ``@~`` expansion to ``argv`` (without
    the program name). Only ``sml advanced`` is affected."""
    env = os.environ if env is None else env
    if not argv or argv[0] not in _EXPANDED_SUBCOMMANDS:
        return argv
    head, rest = argv[0], argv[1:]
    out: list[str] = []
    use_site = True
    dirs: list[Path] | None = None
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg == "--no-site-args":
            use_site = False
        elif arg == "--recipe" or arg.startswith("--recipe="):
            if arg == "--recipe":
                if i + 1 >= len(rest):
                    raise RecipeError("--recipe needs a name, e.g. --recipe qwen3-0.6b (see `sml recipes`).")
                i += 1
                name = rest[i]
            else:
                name = arg.split("=", 1)[1]
            dirs = recipe_dirs(env) if dirs is None else dirs
            out.append(f"@{resolve_recipe(name, dirs)}")
        elif arg.startswith("@") and len(arg) > 1:
            out.append("@" + _expand(arg[1:]))
        else:
            out.append(arg)
        i += 1
    site = env.get("SML_SITE_ARGS", "").strip()
    if use_site and site:
        site_path = Path(_expand(site))
        if not site_path.is_file():
            raise RecipeError(f"SML_SITE_ARGS points to a missing file: {site_path}")
        out.insert(0, f"@{site_path}")
    return [head, *out]


def format_recipe_table(recipes: list[Recipe]) -> str:
    if not recipes:
        return "No recipes found. Set SML_RECIPE_PATH (the HPI env.sh does) or add *.args files to ~/.sml/recipes."
    width = max(len(r.name) for r in recipes)
    lines = [f"{'NAME'.ljust(width)}  DESCRIPTION"]
    for r in recipes:
        lines.append(f"{r.name.ljust(width)}  {r.description}")
        lines.append(f"{''.ljust(width)}  {r.path}")
    lines.append("")
    lines.append("Launch one with: sml advanced --recipe <NAME>   (extra flags after it override the recipe)")
    return "\n".join(lines)
