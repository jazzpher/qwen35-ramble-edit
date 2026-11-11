# qwen35-ramble-edit

Removing the "keep talking" direction from **Qwen3.5-0.8B** by editing weights directly (no training), following the idea in Astra's video (https://youtu.be/1s59NzMI5m0) and their published edit (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`).

The edit is rank-1 on 64 output columns of `mlp.down_proj` in layers 15 and 19 (131,072 weights, ~0.016% of the model):

```
u   = unit "rambling" direction in the residual stream (1024-d)
cols = the 64 columns of W_down whose output aligns most with u
W'[:, cols] = W[:, cols] - alpha * outer(u, u @ W[:, cols])
```

Astra's published edit has exactly this form with alpha = 6 (verified by rebuilding it hash-for-hash: `scripts/rebuild_astra.py`).

## How my direction was found (`scripts/find_direction.py`)

1. Generate base-model answers to 16 GSM8K **train** problems (thinking on, 2,500-token cap) -> `results/probe_base_gsm8k.jsonl`. All 16 hit the cap, so there were no cleanly-stopped answers to contrast with.
2. Instead, contrast tokens *within* those texts: "rambling" tokens = inside a repeated 8-gram, or more than 40 characters after the correct answer first appears; "productive" tokens = the rest.
3. Teacher-force the texts through the model, take the mean difference of the hidden state at layers 15 and 19 -> `u`.
4. Pick the 64 `down_proj` columns with the largest |u . W_col| and apply the update above (alpha 6, same as Astra).

Test problems (`results/gsm8k_sample.jsonl`, GSM8K test, seed 0) were never used to find the direction.

## Results

Same 12 GSM8K test problems, Qwen chat template with thinking on, temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, answer = last `\boxed{}`, **6,000-token cap**, llama.cpp Q8_0 on CPU.

| model | n | GSM8K acc | hit 6000 cap | verbatim loop rate | kept going after answer | mean tokens |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (unedited) | 12 | 25% | 75% | 42% | 50% | 5738 |
| this repo's edit (v1) | 12 | 25% | 83% | 17% | 58% | 5141 |
| Astra's published edit | 12 | 8% | 92% | 8% | 50% | 5903 |

- **hit cap**: generation stopped at 6,000 tokens.
- **verbatim loop rate**: hit the cap and >=50% of the tail is repeated 8-grams.
- **kept going**: the correct number already appeared in the reasoning, but the model didn't stop.

### Honest reading

- n = 12 on CPU, so every row is +-1 problem of noise. Treat it as a smoke test, not a benchmark.
- My edit did **not** raise accuracy (3/12 both). It cut verbatim loops (42% -> 17%) and solved two problems in ~850 tokens that base needed ~5,700 for, but one problem base solved (idx 480) now hits the cap. Most remaining caps are "Wait, let's read the sentence again" re-checking, not loops.
- Astra's edit scores lowest here, but that is mostly the cap: in their own logs its correct answers have a median of ~9,200 tokens, so a 6,000 cap cuts it off. This is not a reproduction of their 13.8% -> 67.1% claim (different cap, sampler and problem set).
- Even on a trivial question ("3 apples + 4") the edited model still second-guesses itself ("Wait, is there a trick?") for 1,500 tokens. The edit trims some loops, it does not make the model concise.
- The 2,500-cap baseline (`results/base_cap2500_*`) was 0/8, all hitting the cap.

## Run it on a laptop

See `scripts/run_on_laptop.md`. The Q8_0 GGUF (~830 MB, split in 2 parts) is attached to this repo's Release and runs on CPU with 8 GB RAM (tested on a 2-CPU / 4 GB box at ~37 tok/s).

## Reproduce

```
python scripts/eval_gsm8k.py <label> <max_tokens> <port> <n>   # against a llama-server
python scripts/analyze.py results/<label>_gsm8k.jsonl
python scripts/find_direction.py results/probe_base_gsm8k.jsonl --alpha 6 --k 64 --layers 15,19 --contrast both --out ./edited
python scripts/make_table.py
```
