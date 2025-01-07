from flask import Flask, render_template, request, jsonify, Response
from web_deepseek import DeepseekChat, get_api_key
import json

app = Flask(__name__)
api_key = get_api_key()
chat = DeepseekChat(api_key)
messages = []


@app.route("/")
def index():
    return render_template("index.html", messages=messages)


@app.route("/chat")
def chat_endpoint():
    global messages
    user_message = request.args.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Check if the message is "clear"
    if user_message.strip().lower() == "clear":
        messages.clear()  # Clear all messages
        return jsonify({"response": "All messages have been cleared.", "tokens": 0})

    # Add user message
    messages.append({"role": "user", "content": user_message})

    def generate():
        full_response = ""
        output_tokens = 0
        try:
            for chunk, token_count in chat.chat_stream(messages):
                full_response += chunk
                output_tokens += token_count
                yield f"data: {json.dumps({'response': chunk, 'tokens': token_count})}\n\n"

            # Add final AI response to messages
            messages.append(
                {"role": "assistant", "content": full_response, "tokens": output_tokens}
            )
        except Exception as e:
            messages.pop()  # Remove the last message that caused the error
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)