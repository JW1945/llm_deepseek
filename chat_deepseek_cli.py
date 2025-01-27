import os
import requests
import json
import shutil
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
            Response chunks as strings

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

                    for chunk in response.iter_lines():
                        if chunk:
                            decoded_chunk = chunk.decode("utf-8")
                            if decoded_chunk.startswith("data: "):
                                json_chunk = decoded_chunk[6:]
                                if json_chunk != "[DONE]":
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


def chat_interface():
    """Run the interactive chat interface."""
    api_key = get_api_key()
    chat = DeepseekChat(api_key)
    messages = []

    print("Welcome to Deepseek Chat! Type 'exit' to end the conversation.")
    print("Type 'clear' to reset the conversation history.\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                messages = []
                print("Conversation history cleared.")
                continue

            if not user_input:
                print("Please type something to continue the conversation.")
                continue

            messages.append({"role": "user", "content": user_input})

            # Count input tokens
            input_tokens = chat.tokenizer.count_tokens(json.dumps(messages))
            print(f"\033[91m[Input tokens: {input_tokens}]\033[0m\n")

            print("AI: ", end="", flush=True)
            full_response = ""
            output_tokens = 0
            try:
                for chunk, token_count in chat.chat_stream(messages):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    output_tokens += token_count
                print(f"\n\033[91m[Output tokens: {output_tokens}]\033[0m")
            except Exception as e:
                print(f"\nError: {e}")
                messages.pop()  # Remove the last message that caused the error
                continue

            # Add separator line
            terminal_width = shutil.get_terminal_size().columns
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            print(f"\n{YELLOW}{'=' * terminal_width}{RESET}")

            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            break


if __name__ == "__main__":
    chat_interface()
