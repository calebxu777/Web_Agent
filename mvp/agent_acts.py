"""
Compatibility shim for MVP act imports.

The single source of truth now lives in `src.agent_acts`.
Keep this module only so older imports do not break.
"""

from src.agent_acts import *  # noqa: F401,F403
