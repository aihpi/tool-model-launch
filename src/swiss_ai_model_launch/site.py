"""User-facing site branding, overridable per deployment through SML_* variables.

The defaults are the upstream (Swiss AI / CSCS) wording. A site that fronts sml
with its own gateway sets these in its environment file (see hpi/sml.env) so
prompts and titles name the thing users actually hold, without forking the code.
"""

import os

SITE_NAME = os.environ.get("SML_SITE_NAME") or "SwissAI"
APP_TITLE = f"{SITE_NAME} Model Launch"

HEALTH_KEY_NAME = os.environ.get("SML_HEALTH_KEY_NAME") or "Swiss AI Research API Key"
HEALTH_KEY_HELP = (
    os.environ.get("SML_HEALTH_KEY_HELP")
    or "Get one at: https://serving.swissai.svc.cscs.ch  (log in -> View API Keys)"
)
HEALTH_KEY_INTRO = f"\nThe {HEALTH_KEY_NAME} is used for health checks against your served model.\n{HEALTH_KEY_HELP}\n"
HEALTH_KEY_PROMPT = f"What is your {HEALTH_KEY_NAME}?"
