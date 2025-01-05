import os
import requests
import json
import logging
import tiktoken
from typing import Dict, List, Generator, Optional, Tuple
from time import sleep

# Constants
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 1.0
API_TIMEOUT = 30.0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="\033[91m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",
)
logger = logging.getLogger(__name__)


class Tokenizer:
    """Simple token counter using tiktoken."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.encoder = tiktoken.get_encoding(model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self.encoder.encode(text))


class DeepseekChat:
    """Deepseek Chat API client for streaming chat completions."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize the DeepseekChat client.

        Args:
            api_key: Deepseek API key
            base_url: Optional base URL for the API (defaults to official API)
        """
        self.api_key = api_key
        self.base_url = base_url or "https://api.deepseek.com/"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.tokenizer = Tokenizer(model="cl100k_base")

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Stream chat response from Deepseek LLM.

        Args:
            messages: List of message dicts [{"role": "user", "content": "your message"}]
            model: Model to use (default: deepseek-chat)
            temperature: Creativity level (0.0 to 1.0)

        Yields:
            Tuple of (response chunk, token count)

        Raises:
            Exception: If API request fails after retries
        """
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "max_tokens": 8192,
        }

        logger.info(f"Starting chat stream with {len(messages)} messages")

        for attempt in range(MAX_RETRIES):
            try:
                with requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=data,
                    stream=True,
                    timeout=API_TIMEOUT,
                ) as response:
                    if response.status_code != 200:
                        error_msg = (
                            f"API Error: {response.status_code} - {response.text}"
                        )
                        logger.error(error_msg)
                        if attempt < MAX_RETRIES - 1:
                            sleep(RETRY_DELAY)
                            continue
                        raise Exception(error_msg)

                    buffer = ""
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            buffer += chunk.decode("utf-8")
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if line.startswith("data: "):
                                    json_chunk = line[6:]
                                    if json_chunk == "[DONE]":
                                        return
                                    try:
                                        chunk_data = json.loads(json_chunk)
                                        content = chunk_data["choices"][0]["delta"].get(
                                            "content", ""
                                        )
                                        if content:
                                            yield (
                                                content,
                                                self.tokenizer.count_tokens(content),
                                            )
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSON decode error: {e}")
                                        continue
                    break  # Success - exit retry loop

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    sleep(RETRY_DELAY)
                    continue
                raise Exception(
                    f"API request failed after {MAX_RETRIES} attempts"
                ) from e


def get_api_key() -> str:
    """Get API key from environment variable."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set DEEPSEEK_API_KEY environment variable")
    return api_key