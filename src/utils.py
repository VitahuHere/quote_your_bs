import os

import dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

dotenv.load_dotenv()


def get_vision_model():
    """Returns a vision model for image processing."""
    return ChatOpenAI(
        model=os.getenv("VISION_MODEL"),
        temperature=0.0,
        base_url=os.getenv("BASE_URL"),
        api_key=SecretStr(os.getenv("VISION_API_KEY", "OpenSource")),
        max_tokens=int(os.getenv("VISION_MAX_TOKENS", "4096")),
    )
