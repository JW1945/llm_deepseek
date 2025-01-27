import requests
import json
import os


def process_stream_response(response):
    reasoning_content = ""
    content = ""
    last_type = None  # Track the last type of output we printed

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data: "):
                try:
                    chunk = json.loads(decoded_line[len("data: ") :])
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        if (
                            "reasoning_content" in delta
                            and delta["reasoning_content"] is not None
                        ):
                            reasoning_content += delta["reasoning_content"]
                            if last_type != "REASONING":
                                print(
                                    "\n\033[31m[REASONING] \033[0m", end="", flush=True
                                )
                                last_type = "REASONING"
                            print(delta["reasoning_content"], end="", flush=True)
                        if "content" in delta and delta["content"] is not None:
                            content += delta["content"]
                            if last_type != "CONTENT":
                                print("\n\033[33m[CONTENT] \033[0m", end="", flush=True)
                                last_type = "CONTENT"
                            print(delta["content"], end="", flush=True)
                except json.JSONDecodeError:
                    continue
    print()  # New line after stream ends
    return reasoning_content, content  # currently return object aren't used


def chat():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set the DEEPSEEK_API_KEY environment variable")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []

    print("Welcome to the Reasoning Chat! Type 'exit' to quit or press Ctrl+C to exit.")

    try:
        while True:
            try:
                user_input = input("\nYou: ")

                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break

                messages.append({"role": "user", "content": user_input})

                print("\nAssistant:")
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "deepseek-reasoner",
                        "messages": messages,
                        "stream": True,
                        "max_tokens": 8192,
                    },
                    stream=True,
                )

                reasoning_content, content = process_stream_response(response)

                # Add assistant's response to message history
                messages.append({"role": "assistant", "content": content})
            except KeyboardInterrupt:
                print("\n\nGoodbye! (Ctrl+C pressed)")
                break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Exiting...")


if __name__ == "__main__":
    chat()
