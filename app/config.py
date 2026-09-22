import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class AgentMode(str, Enum):
    READ_ONLY = "READ_ONLY"
    ALLOW_EDITS = "ALLOW_EDITS"
    FULL_AUTONOMOUS = "FULL_AUTONOMOUS"


OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"

PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "openrouter/free")
REVIEWER_MODEL: str = os.getenv("REVIEWER_MODEL", "openrouter/free")

PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()

AGENT_MODE: AgentMode = AgentMode(os.getenv("AGENT_MODE", "READ_ONLY"))
MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "15"))
MAX_RETRY_CYCLES: int = int(os.getenv("MAX_RETRY_CYCLES", "3"))
COMMAND_TIMEOUT: int = int(os.getenv("COMMAND_TIMEOUT", "30"))
