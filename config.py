import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data" / "generated"
MODEL_PATH = ROOT / "models" / "scorer.joblib"

# risk bands used by the triage step. tuned by eye against the synthetic data,
# not sacred - retune if you regenerate with different injection rates.
HIGH_RISK = 0.75
MED_RISK = 0.40

# hard rail: at or above this score a human MUST sign off, regardless of what
# the narrative or reviewer agent concludes. the agents can never auto-clear
# these. this is the line that keeps the thing enterprise-safe.
HUMAN_REVIEW_FLOOR = HIGH_RISK

# provider is swappable so nobody's locked to one vendor. default is claude.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
