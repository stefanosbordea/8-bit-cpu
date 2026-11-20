| model | arithmetic | weight edit | syntax filter | decoding | custom 200 pass@1 | custom 200 pass@10 | custom 200 syntax ok | custom 200 runs ok | MBPP pass@1 | MBPP pass@10 | MBPP syntax ok | MBPP runs ok | HumanEval pass@1 | HumanEval pass@10 | HumanEval syntax ok | HumanEval runs ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tier A | float | - | off | greedy | 2.5% | - | 92.5% | 48.0% | 0.6% | - | 83.2% | 40.3% | - | - | - | - |
| Tier A | float | - | off | 10 samples, T = 0.8 | 2.2% | 7.0% | 86.8% | 37.3% | 0.3% | 1.1% | 67.6% | 19.1% | - | - | - | - |
| Tier A | float | - | on | greedy | 2.5% | - | 94.5% | 48.5% | 0.6% | - | 87.3% | 41.4% | - | - | - | - |
| Tier A | float | - | on | 10 samples, T = 0.8 | 2.2% | 7.0% | 89.2% | 37.6% | 0.3% | 1.3% | 72.0% | 19.7% | - | - | - | - |
| Tier A | int8 | - | off | greedy | 3.0% | - | 91.0% | 48.0% | 0.9% | - | 84.7% | 37.7% | - | - | - | - |
| Tier A | int8 | - | off | 10 samples, T = 0.8 | 2.0% | 6.0% | 85.0% | 37.0% | 0.5% | 2.4% | 66.9% | 19.7% | - | - | - | - |
| Tier A | int8 | - | on | greedy | 3.0% | - | 92.0% | 48.5% | 0.9% | - | 89.2% | 38.8% | - | - | - | - |
| Tier A | int8 | - | on | 10 samples, T = 0.8 | 2.0% | 6.0% | 88.8% | 37.8% | 0.5% | 2.6% | 73.6% | 20.8% | - | - | - | - |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | float | - | off | greedy | 1.5% | - | 89.0% | 36.5% | 0.6% | - | 71.6% | 36.2% | - | - | - | - |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | float | - | off | 10 samples, T = 0.8 | 1.4% | 3.5% | 83.0% | 31.8% | 0.1% | 0.4% | 59.1% | 13.7% | - | - | - | - |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | float | - | on | greedy | 1.5% | - | 92.0% | 38.0% | 0.6% | - | 80.2% | 39.2% | - | - | - | - |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | float | - | on | 10 samples, T = 0.8 | 1.4% | 3.5% | 86.7% | 32.6% | 0.1% | 0.4% | 64.8% | 14.4% | - | - | - | - |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | - | off | greedy | 0.5% | - | 90.0% | 40.5% | 0.4% | - | 75.0% | 36.6% | - | - | - | - |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | - | off | 10 samples, T = 0.8 | 1.1% | 4.0% | 82.2% | 32.6% | 0.1% | 0.6% | 58.3% | 15.3% | - | - | - | - |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | - | on | greedy | 0.5% | - | 94.0% | 42.5% | 0.4% | - | 82.5% | 39.4% | - | - | - | - |
| Tier A, earlier model (10,000 steps; the editing sessions used it) | int8 | - | on | 10 samples, T = 0.8 | 1.1% | 4.0% | 87.7% | 33.5% | 0.1% | 0.6% | 66.2% | 16.8% | - | - | - | - |
| Tier B | float | - | off | greedy | 4.5% | - | 96.5% | 52.0% | 0.9% | - | 78.2% | 45.0% | - | - | - | - |
| Tier B | float | - | on | greedy | 4.5% | - | 97.0% | 52.0% | 0.9% | - | 79.7% | 45.7% | - | - | - | - |
