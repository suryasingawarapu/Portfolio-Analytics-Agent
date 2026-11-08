"""Shared configuration: loads the API key and fails fast with a clear message if it's missing."""
import logging
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

# Plain stdlib logging, as the README asks for -- lets anyone see what the agent is doing
# (which tool it picked, what SQL it generated, what failed) without needing an external
# tracing platform. An agent's failures aren't deterministic bugs, so seeing each step
# matters more here than for ordinary code -- this is the lightweight way to get that.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet the SDK's own HTTP/request-level chatter -- noise, not agent decisions.
for _noisy_logger in ("httpx2", "httpcore2", "google_genai"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    sys.exit(
        "GEMINI_API_KEY is not set.\n"
        "Create a .env file (see .env.example) or export it in your shell, then try again."
    )

# One Flash-tier model for everything: cheap and fast, plenty for SQL generation and routing.
MODEL_NAME = "gemini-3.5-flash-lite"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=genai.types.HttpOptions(api_version="v1beta"),  # required for function calling
)

# A handful of known, common failure modes translated into plain language instead of a
# raw stack trace reaching the user -- anything else falls back to a generic message.
_FRIENDLY_ERRORS = {
    401: "The Gemini API key is missing or invalid. Check your .env file.",
    429: "Gemini's rate limit was hit. Wait a moment and try again.",
}


def friendly_error(error: Exception) -> str:
    status = getattr(error, "code", None)
    return _FRIENDLY_ERRORS.get(status, f"The AI service returned an error: {error}")


def generate_content_stream(contents, config=None):
    """Streaming version of generate_content, for a typewriter-style UI effect. No retry
    here deliberately -- by the time this is called, the actual data lookup has already
    succeeded; this only re-phrases it in prose, so a transient hiccup costs a retry of the
    phrasing, not the underlying answer."""
    return client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=config)


def generate_content(contents, config=None, retries: int = 5, delay_seconds: float = 2.0):
    """client.models.generate_content, retrying on transient failures before giving up:
    a temporary 503 "model overloaded" or network hiccup (short backoff), or a 429 rate
    limit -- expected and frequent on the free tier's request-per-minute cap -- which
    waits long enough for the next quota window instead of failing the question outright.
    A non-429 client error (a malformed request) is raised immediately: retrying it would
    just fail the same way again."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
        except errors.ClientError as e:
            if e.code != 429:
                raise
            last_error, wait = e, 15.0  # free-tier quota resets per minute
        except Exception as e:  # ServerError, DNS/connection blips, timeouts, etc.
            last_error, wait = e, delay_seconds
        if attempt < retries:
            time.sleep(wait)
    raise last_error
