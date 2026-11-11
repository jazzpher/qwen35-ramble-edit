# Running the edited model on a laptop (CPU only, 8 GB RAM)

1. Download a llama.cpp release for your OS (Windows: the `win-cpu-x64` zip) from https://github.com/ggml-org/llama.cpp/releases
2. From this repo's Release, download BOTH parts into the same folder:
   - `qwen35-0.8b-ramble-edit-q8_0-00001-of-00002.gguf`
   - `qwen35-0.8b-ramble-edit-q8_0-00002-of-00002.gguf`
   llama.cpp loads the second part by itself when you point it at the first.
3. Chat in the terminal:
   llama-cli -m qwen35-0.8b-ramble-edit-q8_0-00001-of-00002.gguf -t 4 -c 8192 --jinja --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 1.5
   Or as an OpenAI-compatible server on http://127.0.0.1:8080 :
   llama-server -m qwen35-0.8b-ramble-edit-q8_0-00001-of-00002.gguf -t 4 -c 8192 --jinja
Uses about 1-1.5 GB RAM. Thinking mode is on by default.

Want one file instead? `llama-gguf-split --merge qwen35-0.8b-ramble-edit-q8_0-00001-of-00002.gguf qwen35-0.8b-ramble-edit-q8_0.gguf`
(merged sha256: b0f48208b20abe0358fb4b1bd1ff97bfe344b79c177425c4e92f7d806db0fae2)

Checked on a 2-CPU / 4 GB RAM Linux box: loads from the split parts and generates at ~37 tokens/s (1 slot, 2 threads).
