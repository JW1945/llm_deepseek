# DeepSeek LLM Interface

This repository provides two different methods to interact with the DeepSeek Language Model:
1. Terminal-based interface
2. Web-based interface

## Features
- Direct terminal interaction (I'm using PowerShell)
- Web-based GUI interface

## Setup

1. Using UV
```bash
uv run app.py
# or
uv run token_deepseek.py
```

2. Set up your DeepSeek API token (`DEEPSEEK_API_KEY`) in the environment variables.

## Usage

### Method 1: Terminal-based Interface (token_deepseek.py)

Run the terminal version using:
```bash
uv run token_deepseek.py
```
This will start an interactive PowerShell session where you can directly chat with the DeepSeek model.

Features:
- Direct command-line interaction
- Simple text-based interface
- Quick responses
- Ideal for command-line users

### Method 2: Web-based Interface (web_deepseek.py) by using app.py

Launch the web interface using:
```bash
uv run app.py
```
Then open your browser and navigate to `http://localhost:5000`

Features:
- User-friendly web interface
- Chat-like experience
- More visual interaction
- Better for non-technical users
- Supports markdown formatting

## License

This project is licensed under the MIT License - see the LICENSE file for details.
