
import os
from dotenv import load_dotenv

load_dotenv()

mongodb_uri = os.environ["MONGODB_URI"]
openai_api_key = os.environ["OPENAI_API_KEY"]
