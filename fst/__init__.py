"""False-statement filler experiment: do models compute over appended false statements?"""

import os
from pathlib import Path

from dotenv import load_dotenv

# API keys live in the repo-root .env (gitignored); variables already set in the shell win.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# The python.org macOS build ships without CA certificates; point HTTPS clients at certifi's bundle.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass
