"""Hugging Face token for gated models, stored where the jobs can read it.

The compute nodes cannot read the login node's keyring, so the token lives in a
mode-600 file on the shared home; the rank scripts export it as ``HF_TOKEN``
(``--hf-token-file``). ``sml init`` fills the file, but only on sites that set
``SML_HF_TOKEN_FILE``; upstream init is unchanged.
"""

import os
import sys
from pathlib import Path

import questionary

PROMPT = "Hugging Face token for gated models (Enter to skip)?"


def write_token(path: Path, token: str) -> None:
    """Write ``token`` to ``path`` with mode 600, creating parent directories.
    The file is created 600 before any byte is written, never group-readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)  # an existing file keeps its old mode otherwise
        os.write(fd, (token.strip() + "\n").encode())
    finally:
        os.close(fd)


async def configure_from_env() -> None:
    """The `sml init` step: no-op unless SML_HF_TOKEN_FILE is set. The token comes
    from SML_HF_TOKEN when given (non-interactive), else from a prompt on a TTY.
    An empty answer keeps an existing file."""
    target = os.environ.get("SML_HF_TOKEN_FILE", "").strip()
    if not target:
        return
    path = Path(os.path.expanduser(target))
    token = os.environ.get("SML_HF_TOKEN", "").strip()
    if not token and sys.stdin.isatty():
        print(
            "\nGated models (Llama, Gemma, ...) need a Hugging Face read token with the model's license "
            f"accepted.\nIt is stored in {path} (mode 600) for your jobs, never in scripts or logs.\n"
        )
        token = ((await questionary.password(PROMPT).ask_async()) or "").strip()
    if token:
        write_token(path, token)
        print(f"Hugging Face token saved to {path}.")
    elif path.exists():
        print(f"Keeping the existing Hugging Face token in {path}.")
