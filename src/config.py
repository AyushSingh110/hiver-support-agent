from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_TWCS_CSV = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
RAW_SAMPLE_CSV = PROJECT_ROOT / "data" / "raw" / "sample.csv"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# e.g. "Tue Oct 31 22:10:47 +0000 2017"
TWITTER_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"

CHUNK_SIZE = 500_000
RANDOM_SEED = 42
TOP_BRAND_COUNT = 25
BRAND_TEXT_SAMPLE = 3000
MIN_FREE_DISK_GB = 1.5

# ---- Phase 6 ----

BRAND = "AmericanAir"
GOLDEN_DIR = PROJECT_ROOT / "golden"
GOLDEN_SET = GOLDEN_DIR / "golden_set_v1.csv"
GOLDEN_CONVERSATION_EXCLUSIONS = GOLDEN_DIR / "golden_exclusion_ids.txt"
GOLDEN_CUSTOMER_EXCLUSIONS = GOLDEN_DIR / "golden_customer_exclusion_ids.txt"

CORPUS_DIR = PROCESSED_DIR / "phase6"
CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache"

# Ollama runs locally; these models are already present. Never downloaded here.
OLLAMA_URL = "http://localhost:11434/api/generate"
GENERATION_MODEL = "llama3.1:8b"
# Deliberately a different family from the generator: a model grading its own
# output is a known bias.
JUDGE_MODEL = "qwen2.5:7b"
OLLAMA_TIMEOUT_SECONDS = 300

NEAR_DUPLICATE_THRESHOLD = 0.8

# Escalation thresholds. Tuned on a dev split from training data, never on golden.
ESCALATION_MIN_INTENT_CONFIDENCE = 0.50
ESCALATION_MIN_RETRIEVAL_SIMILARITY = 0.25
ESCALATION_MIN_EVIDENCE_COUNT = 2
RETRIEVAL_TOP_K = 5
