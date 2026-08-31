"""Load .env before test collection, so skipif conditions that check
os.environ directly (e.g. GOOGLE_API_KEY) see it."""
from dotenv import load_dotenv

load_dotenv()
