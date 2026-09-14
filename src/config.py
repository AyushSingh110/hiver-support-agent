from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_TWCS_CSV = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
RAW_SAMPLE_CSV = PROJECT_ROOT / "data" / "raw" / "sample.csv"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
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
