| model | n | correct (any boxed) | completed correct | hit cap | verbatim loops | mean tokens | mean tokens, completed correct | completed-correct idx |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (unedited) | 12 | 3 | 3 | 9/12 | 5/12 | 5738 | 4952 | 480, 1020, 1101 |
| v1 (alpha 6, layers 15+19) | 12 | 3 | 2 | 10/12 | 2/12 | 5141 | 844 | 1020, 1101 |
| Astra's published edit | 12 | 1 | 1 | 11/12 | 1/12 | 5903 | 4838 | 1257 |
| v2 (alpha 8, layers 15+19) | 12 | 3 | 3 | 9/12 | 4/12 | 5291 | 3163 | 480, 799, 1101 |
| alpha 3 (stopped at 8/12) | 8 | 1 | 1 | 7/8 | 1/8 | 5484 | 1872 | 1101 |
| alpha 10 (stopped at 8/12) | 8 | 1 | 1 | 7/8 | 2/8 | 5836 | 4685 | 1101 |
