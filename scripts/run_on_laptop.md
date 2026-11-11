# Running the edited model on a laptop (CPU only, 8 GB RAM)

1. Download a llama.cpp release for your OS (Windows: the `win-cpu-x64` zip) from https://github.com/ggml-org/llama.cpp/releases
2. Download `qwen35-0.8b-ramble-edit-q8_0.gguf` (~0.8 GB) from this repo's Releases.
3. Chat in the terminal:
   llama-cli -m qwen35-0.8b-ramble-edit-q8_0.gguf -t 4 -c 8192 --jinja --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 1.5
   Or as an OpenAI-compatible server on http://127.0.0.1:8080 :
   llama-server -m qwen35-0.8b-ramble-edit-q8_0.gguf -t 4 -c 8192 --jinja
Uses about 1-1.5 GB RAM. Thinking mode is on by default.
