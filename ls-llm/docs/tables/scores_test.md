| model | arithmetic | weight edit | syntax filter | decoding | custom 200 pass@1 | custom 200 pass@10 | custom 200 syntax ok | custom 200 runs ok | MBPP pass@1 | MBPP pass@10 | MBPP syntax ok | MBPP runs ok | HumanEval pass@1 | HumanEval pass@10 | HumanEval syntax ok | HumanEval runs ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tier A | float | - | off | greedy | 4.0% | - | 93.0% | 43.0% | 0.8% | - | 80.4% | 34.2% | 0.0% | - | 51.2% | 16.5% |
| Tier A | float | - | off | 10 samples, T = 0.8 | 3.6% | 12.0% | 88.1% | 39.4% | 0.1% | 0.8% | 67.6% | 16.9% | 0.0% | 0.0% | 49.0% | 7.3% |
| Tier A | float | - | on | greedy | 4.0% | - | 95.0% | 43.5% | 0.8% | - | 87.4% | 35.6% | 0.0% | - | 55.5% | 17.1% |
| Tier A | float | - | on | 10 samples, T = 0.8 | 3.5% | 11.5% | 91.0% | 40.0% | 0.1% | 0.8% | 71.5% | 17.4% | 0.0% | 0.0% | 54.9% | 8.2% |
| Tier A | int8 | - | off | greedy | 6.5% | - | 92.0% | 49.0% | 0.4% | - | 82.2% | 36.6% | 0.0% | - | 65.2% | 23.2% |
| Tier A | int8 | - | off | 10 samples, T = 0.8 | 3.9% | 11.0% | 84.5% | 40.7% | 0.2% | 1.4% | 65.9% | 19.8% | 0.0% | 0.0% | 44.9% | 11.0% |
| Tier A | int8 | - | on | greedy | 6.5% | - | 93.5% | 49.5% | 0.4% | - | 88.8% | 37.4% | 0.0% | - | 72.6% | 25.0% |
| Tier A | int8 | - | on | 10 samples, T = 0.8 | 3.9% | 11.0% | 87.7% | 41.4% | 0.2% | 1.4% | 73.7% | 21.1% | 0.0% | 0.0% | 53.9% | 12.5% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | float | - | off | greedy | 4.5% | - | 88.5% | 38.5% | 0.4% | - | 71.2% | 36.8% | 0.0% | - | 60.4% | 28.7% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | float | - | off | 10 samples, T = 0.8 | 2.1% | 6.0% | 82.8% | 30.1% | 0.1% | 0.8% | 58.5% | 13.5% | 0.0% | 0.0% | 43.2% | 7.3% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | float | - | on | greedy | 4.5% | - | 91.5% | 39.5% | 0.4% | - | 81.8% | 39.4% | 0.0% | - | 64.0% | 29.3% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | float | - | on | 10 samples, T = 0.8 | 2.1% | 6.0% | 86.9% | 30.9% | 0.1% | 0.8% | 64.4% | 14.2% | 0.0% | 0.0% | 49.6% | 8.7% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | - | off | greedy | 5.0% | - | 86.0% | 40.0% | 0.2% | - | 72.4% | 30.8% | 0.0% | - | 56.1% | 17.7% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | - | off | 10 samples, T = 0.8 | 3.2% | 9.5% | 82.0% | 33.5% | 0.1% | 0.6% | 58.9% | 15.6% | 0.0% | 0.0% | 45.9% | 9.8% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | - | on | greedy | 5.0% | - | 90.5% | 40.0% | 0.2% | - | 80.8% | 33.0% | 0.0% | - | 59.1% | 18.9% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | - | on | 10 samples, T = 0.8 | 3.2% | 9.5% | 86.9% | 34.4% | 0.1% | 0.6% | 66.0% | 17.0% | 0.0% | 0.0% | 53.6% | 11.3% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | session s1 | on | greedy | 4.0% | - | 90.5% | 40.0% | 0.2% | - | 75.2% | 32.4% | 0.0% | - | 62.8% | 18.9% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | session s2 | on | greedy | 4.0% | - | 91.5% | 47.5% | 0.0% | - | 87.0% | 38.8% | 0.0% | - | 74.4% | 23.8% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | session s3 | on | greedy | 4.5% | - | 92.5% | 45.0% | 0.0% | - | 81.6% | 36.6% | 0.0% | - | 65.9% | 22.0% |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | session s4 | on | greedy | 3.5% | - | 84.5% | 45.5% | 0.4% | - | 86.0% | 40.4% | 0.0% | - | 82.3% | 54.3% |
| Tier B | float | - | off | greedy | 6.5% | - | 94.5% | 53.0% | 0.2% | - | 82.6% | 46.2% | 0.0% | - | 55.5% | 29.3% |
| Tier B | float | - | off | 10 samples, T = 0.8 | 5.5% | 12.5% | 90.2% | 41.7% | 0.3% | 1.6% | 70.6% | 23.2% | 0.0% | 0.0% | 53.4% | 11.7% |
| Tier B | float | - | on | greedy | 6.5% | - | 95.0% | 53.0% | 0.2% | - | 84.2% | 46.6% | 0.0% | - | 56.7% | 29.3% |
| Tier B | float | - | on | 10 samples, T = 0.8 | 5.5% | 12.5% | 92.0% | 41.9% | 0.3% | 1.6% | 74.1% | 23.6% | 0.0% | 0.0% | 59.8% | 12.4% |
| Tier B | int8 | - | off | greedy | 6.0% | - | 95.5% | 54.0% | 1.0% | - | 85.2% | 50.2% | 0.0% | - | 63.4% | 21.3% |
| Tier B | int8 | - | off | 10 samples, T = 0.8 | 5.5% | 13.0% | 89.9% | 42.2% | 0.5% | 2.0% | 75.1% | 27.9% | 0.1% | 0.6% | 57.8% | 13.7% |
| Tier B | int8 | - | on | greedy | 6.0% | - | 96.0% | 54.0% | 1.0% | - | 86.8% | 50.4% | 0.0% | - | 65.9% | 22.6% |
| Tier B | int8 | - | on | 10 samples, T = 0.8 | 5.5% | 13.0% | 92.8% | 43.0% | 0.5% | 2.0% | 78.9% | 28.9% | 0.1% | 0.6% | 64.7% | 15.4% |
| stories260K (karpathy/tinyllamas, as published) | float | - | off | greedy | 0.0% | - | 62.0% | 62.0% | 0.0% | - | 55.6% | 55.4% | 0.0% | - | 28.7% | 28.7% |
| ngram-432KB | n-gram | - | off | greedy | 0.0% | - | 23.0% | 0.0% | 0.0% | - | 32.2% | 0.0% | 0.0% | - | 62.8% | 0.0% |
| Pico-1 (as published, plain continuation) | float | - | off | greedy | 0.0% | - | 0.0% | 0.0% | 0.0% | - | 0.0% | 0.0% | 0.0% | - | 7.9% | 6.7% |
| Pico-1 (as published, plain continuation) | float | - | off | 10 samples, T = 0.8 | 0.0% | 0.0% | 0.7% | 0.5% | 0.0% | 0.0% | 0.4% | 0.4% | 0.0% | 0.0% | 0.9% | 0.4% |
| Pico-1.2 (as published, plain continuation) | float | - | off | greedy | 0.0% | - | 3.0% | 3.0% | 0.0% | - | 0.6% | 0.6% | 0.0% | - | 22.6% | 22.6% |
| Pico-1.2 (as published, plain continuation) | float | - | off | 10 samples, T = 0.8 | 0.0% | 0.0% | 5.4% | 5.2% | 0.0% | 0.0% | 6.4% | 6.4% | 0.0% | 0.0% | 13.9% | 12.7% |
| CinnabarLM-4M-Python-Base (as published, plain continuation) | float | - | off | greedy | 1.0% | - | 92.0% | 64.5% | 0.6% | - | 85.8% | 63.2% | 0.0% | - | 52.4% | 22.0% |
| Pico-1.3-Coding-Large (as published, plain continuation) | float | - | off | greedy | 8.5% | - | 84.5% | 58.0% | 1.6% | - | 66.0% | 23.0% | 0.6% | - | 57.9% | 20.7% |
| chispa-20m-python (as published, plain continuation) | float | - | off | greedy | 5.0% | - | 93.0% | 63.5% | - | - | - | - | - | - | - | - |
