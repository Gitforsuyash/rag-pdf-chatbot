# check_env.py
from dotenv import load_dotenv
import os

load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
