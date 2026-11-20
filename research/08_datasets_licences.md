# 08. Python training data with clear licences, and terms for AI-generated data

Checked on 2026-11-13. Every licence, size and clause below was opened at the URL given on that date.
"Measured" means: produced by `research/08_dataset_measurements.py` (standard library only, run as
`.venv/bin/python -I research/08_dataset_measurements.py`, Python 3.12, commits pinned, about 45 MB held in memory,
nothing saved, downloaded code only parsed with `ast`, never executed). "Kept" means: a top-level `def` with a
docstring and at most 40 lines that parses under Python 3.12. Quotation marks inside quotes are normalised to ASCII.

Fact-check pass, 2026-11-13 evening, by a second agent: `[VERIFIED]` marks a statement whose source was opened again
or whose number was measured again; `[UNVERIFIED: reason]` marks a statement that could not be supported; statements
without a mark were not re-checked. What was checked, and every correction, is in the "Verification log" at the end.

## 1. Answer in one paragraph

A licence-clean corpus of short documented Python functions can be built with the standard library alone and about
215 MB of downloads, but the hand-picked repositories are small, and only two bulk sources that the standard library
can read without a login are expected to reach the 20 to 80 MB target (extrapolated from one shard each, section
6.4). Nine permissive repositories (TheAlgorithms/Python, keon/algorithms, exercism/python, more-itertools, toolz,
boltons, funcy, pydash, pygorithm; all MIT or BSD-3-Clause, LICENSE files read) hold 10.1 MB of Python in 13.9 MB of
tarballs, but only 3,639 kept functions (2.30 MB); 3,255 of their top-level functions carry doctests, which is their
real value. The bulk has to come from (a) CodeSearchNet Python (457,461 documented functions; one 11.4 MB shard
gives 7,069 kept plain functions, 4.63 MB), which contains GPL, LGPL and AGPL repositories (125 of the 748
repositories in the part of its licence table that was read) and so needs the
per-repository licence table that ships with it; its shards and that table can be read without login by HTTP range
requests into a 940.9 MB zip on Hugging Face, without downloading the zip; and (b) codeparrot/codeparrot-clean (Python files with a
per-file `license` field, gzip JSON lines, no login; 59.5% of sampled files are permissive; about 235 kept functions
per compressed MB). For AI-generated training data the verdicts are: Claude outputs not allowed (Anthropic usage
policy and help-centre article), OpenAI outputs unclear, Llama 2 / Llama 3 / CodeLlama not allowed, Llama 3.1 and 3.3
allowed with a naming condition, the Apache-2.0 sizes of Qwen2.5-Coder clearly allowed, DeepSeek-Coder and StarCoder
allowed but they attach their use restrictions to our model; the three named Hugging Face synthetic datasets do not
state their generator (one of them is gated), so they fail our rule. HumanEval (MIT, 164 problems, 44,877 bytes) and
MBPP (CC-BY-4.0, 974 problems, test split = task ids 11 to 510) are single small files; our 10-gram check found at
most 5 MBPP-test solutions, 1 HumanEval docstring phrase and 1 HumanEval solution phrase overlapping in any one
scanned source: textbook algorithm code (gcd, sorting, binary search), common idioms and shared wording, and no
copied benchmark file in what was scanned (the ten repositories in full; of the two bulk sources only 1 of 16
CodeSearchNet shards and 6 MB of the 12.8 GB of codeparrot-clean). [VERIFIED] (every number in this paragraph was
measured or opened again; that the bulk sources reach the target remains an extrapolation)

## 2. Proven

### 2.1 Hand-picked permissive repositories (LICENSE file read, sizes measured)

- All nine repositories below have a LICENSE file with MIT or 3-clause BSD text. GitHub reports `NOASSERTION` for
  toolz and boltons, but their files are 3-clause BSD (three conditions, the third forbids using the names for
  endorsement): https://raw.githubusercontent.com/pytoolz/toolz/master/LICENSE.txt ,
  https://raw.githubusercontent.com/mahmoud/boltons/master/LICENSE . Per-repository rows are in table 6.1. [VERIFIED]
- Measured totals for the nine: 13.9 MB of tarballs, 2,918 `.py` files, 10.1 MB of Python, 8,036 top-level
  functions, 4,426 with a docstring, 3,255 with a doctest (`>>>`), 3,639 kept (2.30 MB). [VERIFIED] (script run
  again: identical output)
- TheAlgorithms/Python alone gives half of that: tarball about 8.46 MB (8,457,333 bytes in the earlier measurement,
  8,457,964 in the fact-check about an hour later; GitHub builds these archives on request and their bytes are not
  guaranteed to stay the same), 1,608 `.py` files (4.60 MB), 2,758
  top-level functions, 2,112 with doctests, 1,812 kept (1.26 MB); 21 files do not parse under Python 3.12.
  Licence: MIT, "Copyright (c) 2016-2022 TheAlgorithms and contributors",
  https://raw.githubusercontent.com/TheAlgorithms/Python/master/LICENSE.md [VERIFIED]
- exercism/python (MIT, "Copyright (c) 2021 Exercism", https://raw.githubusercontent.com/exercism/python/main/LICENSE )
  contains 142 practice solutions (`.meta/example.py`), 21 concept solutions (`.meta/exemplar.py`) and 163 unit-test
  files (`*_test.py`), counted with the GitHub tree API. Only 225 of its 669 top-level functions have docstrings
  (219 kept, 0.09 MB), so it is more useful as human-written tasks with tests than as bulk text. [VERIFIED]
- CPython: the LICENSE file is the "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2" (grants a "nonexclusive,
  royalty-free, world-wide license to reproduce, analyze, test, perform and/or display publicly, prepare derivative
  works, distribute, and otherwise use Python") plus the Zero-Clause BSD licence for code in the documentation:
  https://raw.githubusercontent.com/python/cpython/main/LICENSE . `Lib/` on `main` has 2,083 `.py` files
  (41.5 MB); without test directories 699 files (12.7 MB) (GitHub tree API). Source tarball
  https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tar.xz is 20,444,032 bytes. [VERIFIED] (licence text and
  tarball size; the `Lib/` file counts were not re-checked)
- LeetCode solution repositories with an MIT LICENSE file: kamyu104/LeetCode-Solutions (3,694 `.py`, about 4.3 MB),
  walkccc/LeetCode (3,169 `.py`, about 1.8 MB), neetcode-gh/leetcode (396 `.py`, about 0.25 MB), qiyuangong/leetcode
  (379 `.py`, 0.44 MB). Measured on qiyuangong/leetcode: 1 top-level function, 0 kept, 50 files do not parse under
  Python 3.12 (solutions are methods of `class Solution`). doocs/leetcode is CC-BY-SA-4.0 and 910 MB. [VERIFIED]
  (licences as reported by the GitHub API, the qiyuangong numbers and the doocs size; the `.py` counts of the other
  three repositories were not re-checked)
- Chalarangelo/30-seconds-of-python (the `30-seconds/30-seconds-of-python` name redirects to it) is CC-BY-4.0,
  archived, last push 2023-05-07, 68.5 MB: https://raw.githubusercontent.com/Chalarangelo/30-seconds-of-python/master/LICENSE
  [VERIFIED] (GitHub API)
- ActiveState/code (4,596 `.py`, about 17.4 MB, last push 2022-04-14): "By default, all newly submitted code is
  licensed under the MIT license. [...] All legacy recipes retain their original license. Please see the LICENSE
  file inside each recipe folder for details." https://github.com/ActiveState/code#what-license-is-this-code-released-under
  [VERIFIED] (quote and last push; the `.py` count was not re-checked)

### 2.2 CodeSearchNet (Python)

- The GitHub repository is archived (last push 2022-01-31); its code is MIT. Data licences: "The licenses for source
  code used as data for this project are provided with the data download for each language in `_licenses.pkl`
  files." https://github.com/github/CodeSearchNet#licenses . The table is a dictionary: "The key is the owner/name
  and the value is a tuple of (path, license content)": https://github.com/github/CodeSearchNet/blob/master/resources/README.md
  [VERIFIED] (the data differs from this description: the value is a list of such tuples, see the licence-table
  bullet below)
- Selection rule: "any projects that do not have a license or whose license does not explicitly permit the
  re-distribution of parts of the project were removed". The Hugging Face card adds: "Each example in the dataset
  has is extracted from a GitHub repository, and each repository has its own license. Example-wise license
  information is not (yet) included in this dataset: you will need to find out yourself which license the code is
  using." [sic] https://huggingface.co/datasets/code-search-net/code_search_net [VERIFIED]
- Size: 412,178 train, 23,107 validation, 22,176 test functions (card metadata; we counted 23,107 rows in the
  validation shard). Parquet files on `main`: 521,785,752 / 30,742,555 / 28,744,352 bytes, without a licence column.
  [VERIFIED]
- The download link in the GitHub README (`https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip`)
  answers 301 and the regional endpoints answer 403 (2026-11-13). It no longer works anonymously. [VERIFIED]
- The original zip is still on Hugging Face at an older revision, no login:
  `https://huggingface.co/datasets/code-search-net/code_search_net/resolve/fdc6a9e39575768c27eb8a2a5f702bf846eb4759/data/python.zip`
  (940,909,997 bytes). HTTP range requests work, so `zipfile` can list it with 2,656 bytes of traffic and read single
  members. Members: 14 train shards of 10.02 to 15.97 MB (196.45 MB together), `python_test_0.jsonl.gz` 10.72 MB,
  `python_valid_0.jsonl.gz` 11.43 MB, `python_licenses.pkl` 32.67 MB (134.17 MB unpacked),
  `python_dedupe_definitions_v2.pkl` 689.65 MB (not needed). [VERIFIED]
- Measured on the validation shard: 23,107 functions from 679 repositories, 25.9 MB of function text; median 19
  lines, 90th percentile 58 lines, 54.5% have at most 20 lines; 13,384 rows (57.9%) are methods (`func_name`
  contains a dot); of the 9,723 plain functions 130 do not parse under Python 3.12 and 7,069 are kept (4.63 MB).
  Fields per row: `repo, path, func_name, original_string, language, code, code_tokens, docstring,
  docstring_tokens, sha, url, partition`. [VERIFIED] (script run again: identical output)
- Second shard, measured in the fact-check with the same functions of the script (`python_train_0.jsonl.gz`,
  15.97 MB): 30,000 functions from only 85 repositories, 37.8 MB of function text, 13,775 methods; of the 16,225
  plain functions 22 do not parse and 11,781 are kept (8.15 MB). The 30,000 rows fit 412,178 = 13 x 30,000 + 22,178
  for the 14 train shards (only this one was counted). The kept share differs between shards (39.3% of rows here,
  30.6% in the validation shard), and this train shard draws on far fewer repositories than the validation shard
  (85 against 679). [VERIFIED]
- Licence table, read without unpickling (the opening 6,000,000 bytes of the unpacked stream, fetched with 2.1 MB of
  range requests, opcodes walked with `pickletools.genops`): the pickle is protocol 3 and maps `owner/name` to a
  list of `(path, licence text)` tuples, one per licence file found in the repository (the README shows a single
  tuple). The sample holds 748 complete repositories and 890 licence files; 696 repositories have one licence file,
  52 have two or more (pypa/pipenv has 71, from vendored packages). Licence files by keyword rules: MIT 448,
  BSD 127, Apache-2.0 120, GPL 106, LGPL 23, AGPL 11, ISC 9, MPL 7, Unlicense 1, not classified 38. Judged by the
  licence files in the repository root, 614 of the 748 repositories (82.1%) are permissive only and 125 (16.7%)
  carry a GPL, LGPL or AGPL text (examples: syncloud/lib GPL-3.0, numenta/htmresearch AGPL-3.0,
  OTTOMATIC-IO/pycine LGPL-3.0); judged by every licence file, 598 (79.9%) are permissive only. So copyleft
  repositories are in CodeSearchNet and a filter is required. [VERIFIED] (fact-check recount of a sample, see
  section 3; it replaces an earlier count of 269 entries, which a cruder parse of the same bytes had given as MIT
  141, BSD 46, Apache-2.0 35, GPL 25, LGPL 16, AGPL 3, ISC 2, MPL 1; the recount script is not in the repository,
  its method is in the verification log)
- Published cross-check: Yu et al. (ICML 2023, Table 6) looked up the repositories of CodeSearchNet with the GitHub
  API (the paper describes the whole dataset, not its Python part; the table gives a total of 114k without naming
  the unit) and report 91.88%
  permissive, 0.50% public domain, 1.26% weak copyleft and 6.37% strong copyleft.
  https://proceedings.mlr.press/v202/yu23g.html [VERIFIED]

### 2.3 Hugging Face code datasets with per-file licence metadata and no login

- codeparrot/codeparrot-clean: not gated; 54 files `file-0000000000NN.json.gz` of 142 to 248 MB (12.8 GB together);
  card: "This dataset has ~50GB of code and 5361373 files"; fields `repo_name, path, copies, size, content, license,
  hash, line_mean, line_max, alpha_frac, autogenerated`. https://huggingface.co/datasets/codeparrot/codeparrot-clean .
  Range requests work (HTTP 206). Measured on bytes 0 to 5,999,999 of file 1: 2,413 files, 24.6 MB of Python;
  licences apache-2.0 507, mit 442, bsd-3-clause 427, gpl-3.0 408, agpl-3.0 228, gpl-2.0 218, mpl-2.0 48,
  lgpl-3.0 43, bsd-2-clause 42, lgpl-2.1 22, unlicense 14, epl-1.0 8, artistic-2.0 3, cc0-1.0 2, isc 1. Permissive
  (mit, apache-2.0, bsd-2/3-clause, isc, unlicense, cc0-1.0): 1,435 files (59.5%), 14.0 MB; 188 of them do not
  parse under Python 3.12; 3,321 top-level functions, 1,684 with docstring, 1,407 kept (0.77 MB). That is about
  235 kept functions and 0.13 MB of kept text per compressed MB. [VERIFIED] (card, file list and script run again:
  identical output) Published cross-check for the whole dataset: Yu et al. (ICML 2023, Table 6, GitHub API lookups)
  report 64.98% permissive, 1.67% public domain, 3.41% weak copyleft and 29.94% strong copyleft; permissive plus
  public domain is 66.65%, about 7 points above the 59.5% of our 6 MB sample.
  https://proceedings.mlr.press/v202/yu23g.html [VERIFIED]
- Fsoft-AIC/the-vault-function: not gated; dataset licence MIT; built from The Stack; every row has "license (list):
  licenses in the repo". Python: train/small 370,657 functions (two parquet files, 278.5 and 275.7 MB), validation
  30,992 (37.6 MB), test 21,652 (30.3 MB). https://huggingface.co/datasets/Fsoft-AIC/the-vault-function .
  Parquet only; the dataset viewer API refuses it ("runs arbitrary Python code"). [VERIFIED] (card: licence, row counts,
  licence field, built from The Stack; not gated; the file sizes and the viewer refusal were not re-checked)
- codeparrot/github-code: not gated; `license` field, "There are in total 15 licenses" (GPL included); 1,126 parquet
  files of about 288 MB with 30 languages mixed. https://huggingface.co/datasets/codeparrot/github-code [VERIFIED] (the
  288 MB file size was not re-checked)
- Gated (login and click-through needed, `gated=auto` in the API): bigcode/the-stack, bigcode/the-stack-dedup,
  bigcode/the-stack-smol. bigcode/the-stack-smol-xs is not gated, but its Python file (967,268 bytes) has no licence
  field in the opening 60 KB (fields seen: `size, ext, lang, max_stars_count, content, avg_line_length,
  max_line_length, alphanum_fraction`).
  [VERIFIED] (gating flags and the 967,268 bytes; the field list was not re-checked)
- The project virtualenv has no parquet reader: `pyarrow`, `pandas`, `datasets`, `fastparquet`, `duckdb` all fail to
  import (checked 2026-11-13). Tarballs, zip, `.json.gz` and `.jsonl.gz` need only the standard library. [VERIFIED]

### 2.4 Licence label versus provenance of APPS, CodeContests, CoNaLa, MBPP, HumanEval

- APPS: label MIT (https://github.com/hendrycks/apps , https://huggingface.co/datasets/codeparrot/apps ). The card
  says: "the authors manually curated problems from open-access sites where programmers share problems with each
  other, including Codewars, AtCoder, Kattis, and Codeforces." Files: `train.jsonl` 107.1 MB, `test.jsonl`
  1,292.4 MB. The Codeforces terms say: "You are expressly and emphatically restricted from all of the following:
  selling, sublicensing and/or otherwise commercializing any Website material" https://codeforces.com/terms [VERIFIED]
- CodeContests: "The code is licensed under the Apache 2.0 License. All non-code materials provided are made
  available under the terms of the CC BY 4.0 license". Sources named: Codeforces, Description2Code ("MIT open source
  license, copyright not specified"), CodeNet ("Apache 2.0, copyright not specified"), and: "We make no
  representations here with respect to rights or abilities to use any such materials."
  https://github.com/google-deepmind/code_contests#license . 7.6 GB of parquet on Hugging Face. [VERIFIED]
- CoNaLa: label MIT on Hugging Face; "The dataset was crawled from Stack Overflow, automatically filtered, then
  curated by annotators, split into 2,379 training and 500 test examples"; the CoNaLa site counts 598,237 mined
  pairs, the Hugging Face copy of the mined set has 593,891 rows (160.3 MB).
  https://huggingface.co/datasets/neulab/conala , https://conala-corpus.github.io/ . Stack Overflow content is
  CC BY-SA (2.5, 3.0 or 4.0 by date): https://stackoverflow.com/help/licensing [VERIFIED]
- MBPP: "All datasets in this repository are released under the CC BY 4.0 International license"
  https://github.com/google-research/google-research#readme ; Hugging Face tag `cc-by-4.0`. The tasks are
  crowd-sourced; the paper notes "crowd-sourcing participants may have made use of reference materials from the
  internet" (section 4.8, https://arxiv.org/abs/2108.07732 ). [VERIFIED]
- HumanEval: MIT, "Copyright (c) OpenAI" https://github.com/openai/human-eval/blob/master/LICENSE ; "all problems
  were hand-written and not programmatically copied from existing sources" https://arxiv.org/abs/2107.03374 [VERIFIED]

### 2.5 Terms for training on AI outputs (clauses read at the source)

- Anthropic Usage Policy (effective Nov 12, 2026), under "Do Not Abuse Our Platform": "Use outputs to train an AI
  model (e.g., "model scraping" or "model distillation") without prior authorization from Anthropic".
  https://www.anthropic.com/legal/aup [VERIFIED]
- Anthropic Consumer Terms (effective Oct 8, 2025): you may not use the Services "To develop any products or
  services that compete with our Services, including to develop or train any artificial intelligence or machine
  learning algorithms or models or resell the Services." Also: "Subject to your compliance with our Terms, we assign
  to you all our right, title, and interest (if any) in Outputs." https://www.anthropic.com/legal/consumer-terms
  [VERIFIED]
- Anthropic Commercial Terms (effective Jun 17, 2025), D.4: "Customer may not and must not attempt to (a) access the
  Services to build a competing product or service, including to train competing AI models or resell the Services
  except as expressly approved by Anthropic". https://www.anthropic.com/legal/commercial-terms [VERIFIED]
- Anthropic help centre, "Can I use my Outputs to train an AI model?" (March 16, 2026): "We prohibit customers from
  using our services to train or develop AI models without our written permission." Allowed: "models that don't
  compete with Anthropic's own models", with examples "Sentiment analysis tools, Content categorization systems,
  Summarization tools, Information extraction tools, Semantic search tools, Anomaly detection tools". "Uses that are
  prohibited include: General purpose chatbots; Models designed for open-ended text generation; Using Outputs as
  training targets for models; Reverse engineering training methods".
  https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model [VERIFIED]
- OpenAI Terms of Use (effective January 1, 2026; openai.com answers 403 to scripts, read through the Wayback
  Machine snapshot of 2026-11-09): you may not "Automatically or programmatically extract data or Output" or "Use
  Output to develop models that compete with OpenAI"; "you (a) retain your ownership rights in Input and (b) own the
  Output." https://web.archive.org/web/20261109060701/https://openai.com/policies/row-terms-of-use/ [VERIFIED]
  That page is the rest-of-world version and says: "If you reside in the European Economic Area, Switzerland, or the
  UK, your use of the Services is governed by these terms", with a link to the Europe Terms of Use. For a team based
  in the EU that is the governing text (updated January 16, 2026; snapshot 2026-10-28). It forbids "Automatically or
  programmatically extracting data or Output" and "Using Output to develop models that compete with OpenAI" and has
  the same ownership sentence, so the verdict does not change.
  https://web.archive.org/web/20261028153406/https://openai.com/policies/eu-terms-of-use/ [VERIFIED]
- OpenAI Services Agreement (effective January 1, 2026; snapshot 2026-11-10): Customer will not "(e) except for a
  Permitted Exception, use Output to develop artificial intelligence models that compete with OpenAI's products and
  services". Permitted Exception: "(a) develop artificial intelligence models primarily intended to categorize,
  classify, or organize data (e.g., embeddings or classifiers), if these models are not distributed or made
  commercially available to third parties; and (b) fine tune or customize models provided as part of OpenAI's
  fine-tuning or other Services set forth on the Pricing Page." https://web.archive.org/web/20261110000152/https://openai.com/policies/services-agreement/
  [VERIFIED]
- Llama 2 (July 18, 2023) and Llama 3 (April 18, 2024): "You will not use the Llama Materials or any output or
  results of the Llama Materials to improve any other large language model (excluding Llama 2 [Meta Llama 3] or
  derivative works thereof)." CodeLlama uses the Llama 2 licence (model card tag `llama2`; its LICENSE file points
  to the Llama 2 licence). https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE ,
  https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE [VERIFIED]
- Llama 3.1 (July 23, 2024), 3.2, 3.3 and 4: that clause is gone; instead: "If you use the Llama Materials or any
  outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is
  distributed or made available, you shall also include "Llama" at the beginning of any such AI model name."
  https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE . For multimodal models of Llama 3.2
  and Llama 4 the licence rights "are not being granted to you if you are an individual domiciled in, or a company
  with a principal place of business in, the European Union."
  https://github.com/meta-llama/llama-models/blob/main/models/llama4/USE_POLICY.md [VERIFIED] (licence files of
  3.1, 3.2, 3.3 and 4 and the use policies of 3.2 and 4 were read)
- Qwen2.5-Coder: the 0.5B, 1.5B, 7B, 14B and 32B Instruct models are tagged `apache-2.0` and ship the Apache-2.0
  text ( https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE ). The 3B model is under the Qwen
  Research Licence: use "FOR NON-COMMERCIAL PURPOSES ONLY" and "If you use the Materials or any outputs or results
  therefrom to create, train, fine-tune, or improve an AI model that is distributed or made available, you shall
  prominently display "Built with Qwen" or "Improved using Qwen" in the related product documentation."
  https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/blob/main/LICENSE [VERIFIED] (the LICENSE files of the
  five Apache-2.0 sizes are byte-identical Apache-2.0 texts and contain no clause about outputs)
- DeepSeek-Coder (DeepSeek License Agreement 1.0, 23 October 2023): "Derivatives of the Model" include "any other
  model which is created or initialized by transfer of patterns of the weights, parameters, activations or output of
  the Model, to the other model, in order to cause the other model to perform similarly to the Model, including -
  but not limited to - distillation methods entailing the use of intermediate data representations or methods based
  on the generation of synthetic data by the Model for training the other model." Derivatives may be made and
  distributed royalty-free, but only with the use restrictions passed on (4.a: "Use-based restrictions as referenced
  in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement"). The
  DeepSeek-Coder-V2 repository ships the identical file.
  https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL [VERIFIED] (an earlier version of this
  bullet left out the words "in order to cause the other model to perform similarly to the Model")
- StarCoder and StarCoder2 (BigCode OpenRAIL-M v1): ""Modifications of the Model" means all changes to the Model or
  any other model which is created or initialized by transfer of patterns of the weights, parameters, activations
  or Output of the Model."; "6. The Output You Generate. Licensor claims no rights in the Output. You agree not to
  contravene any provision as stated in the License Agreement with your Use of the Output."; sharing a Modification
  requires including the Attachment A restrictions and a copy of the licence (5.1, 5.2).
  https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement [VERIFIED]
- Other open-weight code models, licence tag on the model card only (licence text not opened): IBM Granite code
  models `apache-2.0`, Devstral-Small-2505 `apache-2.0`, microsoft/phi-4 `mit`, DeepSeek-R1 `mit`, Codestral-22B
  `mnpl` (non-production), CodeGemma `gemma` (gated). [VERIFIED] (tags only)
- nampdn-ai/tiny-codes: tag `mit`, but gated ("You need to agree to share your Hugging Face information to access
  this dataset"); 981 MB of parquet; "This synthetic dataset is a collection of 1.6 millions short and clear code
  snippets"; "Please note that this dataset is not intended for code-generation purposes". The whole card can be
  read without login (`resolve/main/README.md`, 3,657 bytes, answers 200; only the `raw/main/README.md` address
  answers 401) and names no generating model; it says only "Anyone who wants to reproduce this dataset can use these
  prompts with other LLM models". https://huggingface.co/datasets/nampdn-ai/tiny-codes [VERIFIED]
- flytech/python-codes-25k: not gated; no licence in the metadata, "License MIT" in the card body; 24,813 entries;
  `python-codes-25k.jsonl` 25.4 MB; the card does not say who or what wrote the code, but it describes the `input`
  field as "Very short, introductive part of AI response or empty", so the text is at least partly model output of
  an unnamed model. https://huggingface.co/datasets/flytech/python-codes-25k [VERIFIED]
- iamtarun/python_code_instructions_18k_alpaca: not gated; no licence stated; 11.4 MB parquet; "This dataset is
  taken from sahil2801/code_instructions_120k". That upstream dataset answers HTTP 401 today, so its origin cannot
  be checked. https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca [VERIFIED]

### 2.6 Benchmarks: files, sizes, splits

- HumanEval: https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz , 44,877 bytes, 164 problems,
  fields `task_id, prompt, entry_point, canonical_solution, test`; MIT. Test-only, there is no train split. [VERIFIED]
- MBPP: https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl , 563,743 bytes,
  974 problems (ids 1 to 974), fields `text, code, task_id, test_setup_code, test_list, challenge_test_list`;
  median solution length 5 lines; CC-BY-4.0. Split from the README: "Task IDs 11-510 are used for testing. Task IDs
  1-10 were used for few-shot prompting and not for training. Task IDs 511-600 were used for validation during
  fine-tuning. Task IDs 601-974 are used for training." Counted: 10 / 500 / 90 / 374. The train split holds 66,715
  bytes of solution code. https://github.com/google-research/google-research/tree/master/mbpp [VERIFIED] (downloaded
  and counted again; every task has exactly 3 asserts)
- MBPP sanitized: https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json ,
  255,053 bytes, 427 problems (ids 2 to 809), fields `source_file, task_id, prompt, code, test_imports, test_list`.
  By the same id ranges: prompt 7, test 257, validation 43, train 120 ("the test/train/validation/prompting subsets
  were inherited from the above groupings"). [VERIFIED] (counted again; the paper speaks of 426 hand-verified
  questions, the file holds 427)
- Hugging Face mirrors (parquet): google-research-datasets/mbpp (`full` and `sanitized` configurations, 0.4 MB),
  openai/openai_humaneval (0.08 MB). EvalPlus variants with more tests: evalplus/humanevalplus (`test.jsonl`
  11.45 MB) and evalplus/mbppplus (1.13 MB parquet), both tagged `apache-2.0` (not examined further). [VERIFIED]
  (configuration names and licence tags; the file sizes were not re-checked)

### 2.7 Benchmark contamination: published

- Codex paper: tasks had to be hand-written because "there are more than ten public repositories containing
  solutions to Codeforces problems, which make up part of the recently proposed APPS dataset".
  https://arxiv.org/abs/2107.03374 [VERIFIED]
- MBPP paper, section 4.8, line overlap with the pre-training data: "Only 32 of 974 problems (3.3%) had more than
  half of their lines matched somewhere in the pre-training data and 91.5% had only one or two lines of code that
  overlapped with the training set." https://arxiv.org/abs/2108.07732 [VERIFIED]
- Riddell, Ni, Cohan (2024): "12.2% of the solutions in HumanEval have been seen by models trained on the Pile and
  18.9% have been seen by models trained on the Stack. For MBPP, 3.6% of it can be found in the Pile while as much
  as 20.8% have been seen by models trained on the Stack." StarCoderBase-15.5B scores 41.6 on MBPP, and 33.8 on what
  remains after removing the 20.8% of tasks whose gold solution has similarity score 100 with a training program. https://arxiv.org/abs/2403.04811
  [VERIFIED] (Table 1 of the paper; its MBPP set is the 500 test tasks)
- StarCoder: "The code training data was decontaminated by removing files that contained docstrings or solutions
  from HumanEval and MBPP, docstrings from APPS, questions from GSM8K, or prompts from DS1000. [...] Python is the
  language with the highest number of matches, with 558 files removed." https://arxiv.org/abs/2305.06161 [VERIFIED]
- DeepSeek-Coder (citations removed from the quote): "we filter out files containing docstrings, questions, and solutions from sources such as
  HumanEval, MBPP, GSM8K and MATH. [...] if a piece of code includes a 10-gram string identical to any in the test
  data, it is excluded from our training data. In cases where the test data comprises strings that are shorter than
  10-grams but no less than 3-grams, we use an exact match approach for filtering." https://arxiv.org/abs/2401.14196
  [VERIFIED]
- phi-1: 13-gram overlap between HumanEval docstrings and the synthetic CodeExercises set: "We found 4 humaneval
  questions with 13-gram overlap [...] all the 4 overlap cases in the 13-gram are all false positives". Stronger
  test: "even after aggressively pruning more than 40% of the CodeExercises dataset (this even prunes files that are
  only vaguely similar to HumanEval, see Appendix), the retrained phi-1 still outperforms StarCoder". The same paper on
  narrow simple data (at 1.3 B parameters): "the largest improvement in HumanEval resulted from finetuning on the
  small CodeExercises dataset (<200M tokens). CodeExercises consist exclusively of short Python tasks using only
  basic Python libraries." https://arxiv.org/abs/2306.11644 [VERIFIED]
- Matton et al. (2024) name three leak paths: "(i) direct data leakage, (ii) indirect data leakage through the use
  of synthetic data and (iii) overfitting to evaluation sets during model selection". On synthetic data: when
  fine-tuning a model with the evol-instruct set added, "HumanEval score increases by +9%, MBPP increases by +2%,
  but the LBPP score is unchanged." https://arxiv.org/abs/2407.07565 [VERIFIED]

### 2.8 Benchmark overlap of our candidate sources: measured

Method: every 10-gram of whitespace tokens from HumanEval docstrings (HE-doc), HumanEval canonical solutions
(HE-sol), MBPP test-split task texts and MBPP test-split solutions (21,693 distinct 10-grams) is looked up in every
source file. The number is how many benchmark tasks have at least one 10-gram in the source.

| Source | HE-doc | HE-sol | MBPP-test text | MBPP-test code |
|---|---|---|---|---|
| TheAlgorithms/Python (1,608 files) | 1 | 0 | 0 | 3 |
| keon/algorithms (433 files) | 0 | 0 | 0 | 2 |
| exercism/python, more-itertools, toolz, boltons, funcy, pydash | 0 | 0 | 0 | 0 |
| OmkarPathak/pygorithm (135 files) | 0 | 0 | 0 | 5 |
| qiyuangong/leetcode (379 files) | 0 | 0 | 0 | 1 |
| codeparrot-clean sample (2,413 files) | 1 | 0 | 0 | 0 |
| CodeSearchNet validation shard (23,107 functions) | 0 | 1 | 0 | 1 |

- Hits inspected (one example per source and kind; all of them for pygorithm and codeparrot-clean): the
  TheAlgorithms HE-doc hit is the Collatz wording shared by HumanEval/123 and `project_euler/problem_014/sol2.py`;
  the codeparrot-clean HE-doc hit is a list of the digits 0 to 9 in a numpy file (false positive for
  HumanEval/78); the pygorithm hits are MBPP test tasks 45, 54, 152, 316 and 428 (gcd, counting sort, merge sort,
  binary search, shell sort), where the MBPP reference solution and pygorithm share textbook code; the other code
  hits seen are common idioms (table initialisation, binary search loop, Heron's formula for HumanEval/71); the
  qiyuangong hit is MBPP task 408, which is LeetCode problem 373. [VERIFIED] (the table was reproduced by running the
  script again; the hits in the ten repositories and in the codeparrot-clean sample were listed again with the same
  index, the CodeSearchNet hits were only counted again)
- No scanned source contains an MBPP test task text. With at most 1 HumanEval docstring hit per source, nothing
  that was scanned holds a copy of the HumanEval or MBPP files. That covers the ten repositories in full, but of
  the two bulk sources only the sample: 1 of 16 CodeSearchNet shards, and 6 MB of the 12.8 GB of codeparrot-clean
  (0.05%). CodeSearchNet (2019) is older than both benchmarks (2021); codeparrot-clean was uploaded on 2021-11-02,
  after both were published, so copies in the unscanned part cannot be excluded [UNVERIFIED: only 0.05% of
  codeparrot-clean was scanned; its crawl date is not on the card]. The rank 3 filter has to run on everything that
  is downloaded, not only on the sample.
- Consequence: MBPP reference solutions for classic algorithms are textbook code that also sits in teaching
  repositories. The rank 3 filter in section 5 removes those functions from training.

## 3. Uncertain or contested

- Attribution duties of a trained model. MIT and BSD texts require the notice to stay with copies or redistributed
  source. Whether int8 weights trained on such code are a copy is a legal question we did not research (no case law
  was opened). The weights (260 to 400 KB) are 50 to 300 times smaller than a 20 to 80 MB corpus, but a model can
  still emit short memorised lines. We therefore plan an attribution file without claiming it is legally required.
- Repository licence is not file licence. CodeSearchNet, codeparrot-clean and The Vault record licences per
  repository at crawl time, not per file (CodeSearchNet: the licence files found in the repository; codeparrot-clean:
  one value; The Vault: a list). Changed licences and deleted repositories are not caught by any of them, and
  vendored files only where the vendored package kept its own licence file.
- The licence mix of CodeSearchNet Python is measured on a sample only. The recount in section 2.2 covers the
  opening 6,000,000 bytes (4.5%) of the 134.17 MB table: 748 repositories, 82.1% of them permissive only by their
  root licence files (79.9% by all their licence files), 16.7% with a GPL, LGPL or AGPL text. The classes come from
  keyword rules and were not checked by hand: 38 of the 890 files were not classified, and two Python Software
  Foundation texts vendored in pypa/pipenv were filed as GPL by the full-text fallback. Yu et al. report 91.88%
  permissive for CodeSearchNet as a whole by GitHub API lookups. Whether the start of the table is representative, and
  what share of functions (not repositories) is permissive, is open; the 60% used as a cautious floor in section 6.4
  is the codeparrot-clean sample, not a CodeSearchNet number [UNVERIFIED: the full 32.67 MB table was not read].
- `python_licenses.pkl` is a third-party pickle; `pickle.load` on it can run code. A globals-free unpickler
  (section 6.4) should be enough because the content seen is only dict, list, tuple and str (opcodes in the opening
  6,000,000 bytes: PROTO, EMPTY_DICT, MARK, BINUNICODE, EMPTY_LIST, TUPLE2, APPEND, APPENDS, BINPUT, LONG_BINPUT;
  none of them loads a global) [VERIFIED], but it has not been run on the whole file.
- Dataset labels versus upstream rights. APPS (MIT), CodeContests (CC-BY-4.0) and CoNaLa (MIT) were labelled by
  dataset authors who did not write the problems or solutions. Codeforces restricts sublicensing of its material,
  Stack Overflow content is CC BY-SA. Whether the labels hold is contested; under our rule these are "unclear".
- Creative Commons on code. CC-BY-4.0 (MBPP, 30-seconds-of-python) needs attribution, which we can give.
  CC-BY-SA-4.0 (doocs/leetcode, Stack Overflow) adds share-alike, and its effect on a trained model is unsettled.
- Anthropic: scope of the clause. The Usage Policy says any "AI model"; the help article allows non-competing models
  such as classifiers but lists "Models designed for open-ended text generation" and "Using Outputs as training
  targets for models" as prohibited. LS-LLM generates text and a synthetic corpus would be training targets, so both
  readings say no. What stays open: (a) tasks and tests written by Claude and used only for evaluation are not
  training data, but if the planned gradient-free weight editing uses pass/fail on such tasks as its signal, that is
  arguably training on outputs; (b) text produced by a template generator whose templates an AI assistant wrote.
  Conservative choice until Anthropic gives written permission: weight editing and tuning use only human-written or
  licensed tasks (MBPP train and validation, exercism tests, doctests), and template texts are written by the team.
  Point (b) has since been settled by the project rule of 2026-11-13 19:05 in `CLAUDE.md`: no text written by an AI
  assistant enters the training data, also not through a template generator.
- OpenAI: "compete with OpenAI" is not defined. A 300k-parameter model on an 8-bit CPU is no commercial competitor,
  but it is a generative code model, the consumer terms forbid programmatic extraction of Output, and the business
  exception covers only classifiers and embeddings. Not "clearly allowed".
- Llama 3.1 / 3.3 naming clause applies when the model "is distributed or made available". A course project
  published on GitHub is made available, so the model name would have to begin with "Llama". Whether a report-only
  project triggers it is unclear.
- DeepSeek-Coder and StarCoder: allowed, but our model would be a "Derivative" / "Modification" and would have to
  carry their use restrictions when shared, so the weights could not be released under plain MIT. For DeepSeek this
  is the cautious reading: its definition covers models made "to perform similarly to the Model", which a
  300k-parameter model hardly does; the BigCode definition has no such limit.
- Any synthetic data can leak benchmarks indirectly, even when the licence allows it (Matton et al.: +9% HumanEval,
  +2% MBPP, unchanged on an unseen benchmark after adding one synthetic set). The generator has seen HumanEval and
  MBPP. Generated code can also repeat licensed code from the generator's training data.
- How contaminated MBPP is depends on the corpus and the matching rule: the MBPP authors report 3.3% of problems
  with more than half of their lines matched; Riddell et al. report 20.8% of MBPP solutions seen in The Stack. Our
  10-gram check on whitespace tokens misses renamed variables; we did no similarity or semantic matching.
- Same textbook task, different text. TheAlgorithms and pygorithm implement primes, Fibonacci, sorting and similar
  tasks that HumanEval and MBPP also ask for. Published practice removes only exact docstring or solution matches
  (StarCoder, DeepSeek-Coder); phi-1 prunes by similarity. Whichever rule we use must be stated next to each score.
- Yields are from samples: the CodeSearchNet figure (7,069 kept per shard) comes from 1 of 16 shards, the
  codeparrot-clean figure (235 kept functions per compressed MB) from 6 MB of 1 of 54 files. [VERIFIED] A second
  CodeSearchNet shard measured in the fact-check (train_0: 11,781 kept of 30,000 rows, 8.15 MB) gives a higher rate:
  the yield per row differs by a factor of about 1.3 between these two shards.
- The Vault's licence column was not inspected in the data (no parquet reader here); only the card was read.
- nampdn-ai/tiny-codes: the whole card was read in the fact-check and names no generator (section 2.5). The data
  files are gated and were not opened, so a generator named only inside the data cannot be excluded.
- Not examined: copyright exceptions (EU text-and-data-mining rules, US fair use); the plan relies only on explicit
  licence grants. Also not opened: LeetCode's terms for problem statements, Rosetta Code, the data licence of
  Project CodeNet, the origin and crawl date of codeparrot-clean beyond its card [UNVERIFIED: pages not opened].
- Free disk space changed during this work: `df -h /` showed 293 MB free at the start and 3.9 GB an hour later
  (during the fact-check, 19:05 to 19:45: between 1.9 and 4.6 GB).
  The 1.0 GB floor in CLAUDE.md has to be checked before every download step.

## 4. Not tried at our scale (about 260k to 400k int8 parameters, 8-bit multiply-accumulate CPU, 450 KB)

- A model of this size trained on licence-filtered human-written Python that reads a natural-language prompt and is
  scored on HumanEval or MBPP: not found in our search (model survey in `01_tiny_code_models.md`).
- A TinyStories-style corpus for Python (very simple functions, restricted vocabulary, written by a generator whose
  licence permits training, for example an Apache-2.0 Qwen2.5-Coder) for a model below 1 M parameters: not found in
  our search. The nearest published work found is not language-model text: Naïr et al. (2024) train GPT models
  "with 1 million parameters each" on Python programs from a grammar-based generator (TinyPy Generator: assignments,
  conditionals, loops, print; character vocabulary of 41) and score code completion and output prediction, not
  HumanEval or MBPP (https://arxiv.org/abs/2407.10194 , details in `01_tiny_code_models.md`). phi-1 did the
  synthetic-exercise route at 1.3 B parameters with GPT-3.5 as the generator (6 B tokens of filtered web code plus
  1 B synthetic tokens). [VERIFIED] (both papers; a "not found" cannot itself be verified)
- Volume against narrowness. The 20 to 80 MB target is an assumption: the weights are 260 to 400 KB, so the corpus
  is 50 to 300 times larger than the model. Whether 2 MB of short tested functions trains a model of this size as
  well as 30 MB of mixed GitHub functions has not been measured in anything we found.
- Decontamination at this size. The phi-1 pruning experiment is at 1.3 B parameters. Whether removing
  benchmark-similar functions changes the scores of a 300k-parameter model is untested.
- Identifier diversity. GitHub functions use thousands of distinct identifiers and library calls; with 256 or 512
  tokens every rare name costs many tokens. Filtering or normalising identifiers for a tiny code model: not found.
- MBPP train (374 tasks, 66.7 KB of code) as the only in-format data for a model this small: not reported.
- Execution-filtered corpora (keep a function only if its doctests pass in a sandbox) and doctest-to-assert
  conversion for a model this small: not found.
- A published permissive-versus-copyleft count of CodeSearchNet Python from its own `python_licenses.pkl`, or a
  script that filters by it: not found in our search. A published count by another route exists: Yu et al. (ICML
  2023, Table 6) used GitHub API lookups for the repositories of CodeSearchNet as a whole, not only its Python part
  (91.88% permissive, 6.37% strong copyleft, section 2.2). https://proceedings.mlr.press/v202/yu23g.html [VERIFIED]

## 5. Ranked ideas for LS-LLM

None of these changes the model shape, so the effect on tokens per second and on memory is zero for every row, and
none needs a CPU change. Expected score effects are hypotheses: no published number at our scale was found in our search.

| Rank | Idea | Expected effect on score / speed / memory | Cost to try | Risk | How to measure keep-or-drop | CPU change |
|---|---|---|---|---|---|---|
| 1 | Corpus manifest with a licence gate: one JSONL row per training function (source, repository, path, SPDX licence, revision, sha256). The build stops on any licence outside MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC, PSF-2.0, 0BSD, Unlicense, CC0-1.0 (plus CC-BY-4.0 for MBPP only) and writes an attribution file. | Score: none. Makes every data claim checkable. Speed, memory: none. | 0.5 day | Repository-level licences can be wrong for single files. | A test that fails on a row without an allowed licence; rows per licence printed in the report. | No |
| 2 | Core corpus: kept functions and doctests of the nine hand-picked repositories (3,639 functions, 2.30 MB; 3,255 doctest-bearing functions). [VERIFIED] | Score (hypothesis): highest value per byte, because the functions are short, documented and carry runnable examples; expected to help syntax validity and "runs without error" most. Speed, memory: none. | 1 day, 13.9 MB download | Too small alone (overfitting). Textbook style differs from MBPP "Write a function to ..." prompts. | Same model trained with and without this set on top of the bulk data; compare practice-set syntax validity, run rate and pass rate. | No |
| 3 | Decontamination filter with published counts: drop every training function that contains a 10-gram of whitespace tokens from HumanEval prompts or solutions or from MBPP test-split texts or solutions (the DeepSeek-Coder rule). Decide this before any benchmark run and always keep it. | Score: change expected near zero (measured overlap is at most 5 tasks per scanned source). Its value is that benchmark numbers can be defended. Speed, memory: none. | 0.5 day; the index code is in `08_dataset_measurements.py` | 10-grams miss renamed copies. A stricter name-based pruning would also delete harmless textbook functions. | Report functions dropped per source. One with/without comparison on the practice sets; it is never used to choose a model on a test set. | No |
| 4 | Bulk A: CodeSearchNet Python by range requests, filtered with its licence table. Read `python_licenses.pkl` with a globals-free unpickler, classify every licence text of a repository (the value is a list of licence files; 52 of the 748 sampled repositories have more than one, so the rule for mixed repositories must be fixed, strictest: keep a repository only if all its texts are allowed), keep allowed repositories and plain functions of at most 40 lines. Five shards (valid, test, train_0 to train_2: 67.0 MB) hold about 135,000 functions (train shards hold 30,000 rows each: counted on train_0 in the fact-check [VERIFIED], train_1 and train_2 not counted); at the validation-shard rate that is about 41,000 kept functions, about 27 MB, before the licence filter (train_0 alone, measured: 11,781 kept, 8.15 MB, from only 85 repositories, so one large repository with a refused licence can remove a big part of a train shard). | Score (hypothesis): supplies the volume and variety that the core corpus lacks. Speed, memory: none. | 1 to 2 days; 99.7 MB download in members of at most 33 MB | Text matching can misfile dual-licensed repositories. Functions lean on project context. 2019-era code. | Core only against core + CodeSearchNet at equal training steps on the practice sets; share of generated samples that parse. | No |
| 5 | Bulk B: codeparrot-clean, bytes 0 to 99,999,999 of file 1, filtered by the `license` field, functions extracted with `ast`. Extrapolated from the 6 MB sample: about 23,000 kept functions, about 13 MB. | Score (hypothesis): same role as bulk A, with the licence in every row. Speed, memory: none. | 1 day; 100 MB download | Whole application files: extracted functions depend on imports and globals. 13% of permissive files (188 of 1,435) do not parse under Python 3.12 [UNVERIFIED: the cause, probably Python 2 syntax, was not measured]. Duplicates of CodeSearchNet repositories. | As for rank 4; also CodeSearchNet only against codeparrot only at equal bytes, to drop one if both are not needed. | No |
| 6 | Self-contained and simple filter, checked by execution: keep a function only if it uses no names beyond its arguments, builtins and a short standard-library list (`ast` check) and, where it has doctests, only if they pass inside `lsllm.sandbox`. Sort survivors by simplicity (lines, distinct identifiers) and train on the simplest N MB. | Score (hypothesis): larger effect than more volume, because benchmark tasks are self-contained functions and a 256 or 512 token vocabulary wastes capacity on project-specific names. Speed, memory: none. | 2 days; sandbox time for about 3,000 doctest functions | May leave only a few MB. Downloaded code must run only inside the sandbox. | Filtered against unfiltered corpus at equal steps on the practice sets; MB surviving each filter step. | No |
| 7 | MBPP train (374 tasks) mixed in or used for a final fine-tune in MBPP prompt format; MBPP validation (90 tasks) for all tuning decisions. | Score (hypothesis): teaches the MBPP prompt and assert format; the only human-written data in MBPP format with a clear licence among the sources examined here. Speed, memory: none. | 0.5 day; 0.56 MB | 66.7 KB of code: overfitting and memorised answers. Near-duplicates between MBPP train and test are possible and not measured. | MBPP validation pass rate. Never the test split. | No |
| 8 | Own procedural generator: templates produce function, docstring and asserts; every sample is run in the sandbox (see idea 6 in `01_tiny_code_models.md`). | Score (hypothesis): raises syntax validity and run rate; effect on HumanEval and MBPP unknown. Speed, memory: none. | 2 to 3 days | The model learns the templates, not Python. If an AI assistant writes the template texts, the Anthropic clause may apply to the generated corpus (section 3); the project rule of 2026-11-13 19:05 in `CLAUDE.md` therefore forbids AI-written template text in training data, so this row is open only with templates written by hand by the team. | Practice-set scores with and without the generated set; share of generated text in the final mix. | No |
| 9 | Synthetic simple-Python corpus from an Apache-2.0 open-weight model run locally (Qwen2.5-Coder 1.5B or 7B Instruct): one short function with docstring and asserts per sample; keep it only if it parses, its asserts pass in the sandbox and it survives the rank 3 filter. | Score (hypothesis): the TinyStories route; possibly the largest effect for a model this small, but no published result at this size was found in our search. Speed, memory: none. | 3 days or more; disk for weights: 4-bit GGUF files of 1.12 GB (1.5B-Instruct) or 4.68 GB (7B-Instruct), original safetensors 3.09 GB or 15.23 GB [VERIFIED]; the 7B files do not fit on the internal disk above the 1.0 GB floor (1.9 to 4.6 GB were free during the fact-check); plus an inference runtime that is not in the virtualenv (needs approval) | Indirect benchmark leakage through the generator (Matton et al.). Memorised licensed code in outputs. Weak output from a 1.5 B model. Disk space. | Practice-set scores with and without the synthetic set; benchmark-overlap counts of the synthetic set. | No |
| 10 | Extra permissive text: CPython `Lib/` without tests (699 files, 12.7 MB, PSF-2.0) and networkx (695 files, about 7.1 MB, BSD-3-Clause). | Score (hypothesis): small; long modules and classes, few short self-contained functions. Speed, memory: none. | 0.5 day; 20.4 MB plus one tarball | Low yield; style far from the benchmark tasks. | Kept functions after the rank 6 filter; drop if under 0.5 MB. | No |
| 11 | Hard exclusions: no Claude or OpenAI outputs in training text; no Llama 2, Llama 3 or CodeLlama outputs; no APPS, CodeContests, CoNaLa, LeetCode repositories or CC-BY-SA sources; no Hugging Face synthetic set without a named generator; HumanEval and the MBPP test split never enter training or tuning. | Score: none. Avoids having to retract results. | 0 | Gives up the easiest source of simple synthetic data. | Deny-list test in the manifest gate of rank 1. | No |

## 6. Tables asked for in the topic brief

### 6.1 Sources

Sizes: tarball bytes and `.py` counts measured on 2026-11-13; "kept" as defined at the top of this file.

| Source | URL | Licence (SPDX) | Verified how | Size | What it contains | Fit for tiny-model training (short clean functions?) | Verdict |
|---|---|---|---|---|---|---|---|
| TheAlgorithms/Python | https://github.com/TheAlgorithms/Python | MIT | LICENSE.md read; GitHub API [VERIFIED] | tarball 8.46 MB; 1,608 `.py`, 4.60 MB | Algorithm implementations, most with doctests | High: 2,758 top-level functions, 2,112 with doctests, 1,812 kept (1.26 MB) | Use |
| keon/algorithms | https://github.com/keon/algorithms | MIT | LICENSE read; GitHub API [VERIFIED] | tarball 0.25 MB; 433 `.py`, 0.96 MB | Algorithms and data structures | High: 596 top-level functions, 442 with doctests, 526 kept (0.34 MB) | Use |
| exercism/python | https://github.com/exercism/python | MIT | LICENSE read; GitHub API [VERIFIED] | tarball 1.50 MB; 522 `.py`, 1.18 MB | 142 + 21 reference solutions, 163 unit-test files | Medium as text (219 kept, 0.09 MB); high as human-written tasks with tests | Use (text and practice tasks) |
| more-itertools | https://github.com/more-itertools/more-itertools | MIT | LICENSE read; GitHub API [VERIFIED] | tarball 0.15 MB; 7 `.py`, 0.55 MB | Iterator recipes with doctests | High quality, generator-heavy: 138 kept (0.10 MB) | Use |
| pytoolz/toolz | https://github.com/pytoolz/toolz | BSD-3-Clause | LICENSE.txt read (API says NOASSERTION) [VERIFIED] | tarball 0.08 MB; 51 `.py`, 0.22 MB | Functional utilities | 76 kept (0.04 MB) | Use |
| mahmoud/boltons | https://github.com/mahmoud/boltons | BSD-3-Clause | LICENSE read (API says NOASSERTION) [VERIFIED] | tarball 0.28 MB; 66 `.py`, 0.89 MB | Utility functions and classes | 124 kept (0.09 MB) | Use |
| Suor/funcy | https://github.com/Suor/funcy | BSD-3-Clause | LICENSE read; GitHub API [VERIFIED] | tarball 0.13 MB; 53 `.py`, 0.26 MB | Functional helpers, no doctests | 184 kept (0.06 MB), very short functions | Use |
| dgilland/pydash | https://github.com/dgilland/pydash | MIT | LICENSE.rst read; GitHub API [VERIFIED] | tarball 0.21 MB; 43 `.py`, 0.87 MB | Utility functions with doctests | 355 kept (0.22 MB) | Use |
| OmkarPathak/pygorithm | https://github.com/OmkarPathak/pygorithm | MIT | LICENSE read; GitHub API [VERIFIED] | tarball 2.86 MB; 135 `.py`, 0.57 MB | Teaching algorithms | 205 kept (0.10 MB); 5 MBPP-test idiom hits | Use |
| CPython `Lib/` | https://github.com/python/cpython/tree/main/Lib | PSF-2.0 | LICENSE read [VERIFIED] (licence, tarball size) | 3.12.7 `.tar.xz` 20.4 MB; `Lib/` 2,083 `.py`, 41.5 MB; without tests 699 `.py`, 12.7 MB | Standard library | Low to medium: long modules and classes; yield not measured | Optional |
| networkx | https://github.com/networkx/networkx | BSD-3-Clause | LICENSE.txt read (API says NOASSERTION) [VERIFIED] (licence) | 695 `.py`, about 7.1 MB (tree API); tarball not measured | Graph algorithms with docstrings | Medium: functions need graph objects; yield not measured | Optional |
| 30-seconds-of-python | https://github.com/Chalarangelo/30-seconds-of-python | CC-BY-4.0 | LICENSE read; GitHub API [VERIFIED] (GitHub API) | repository 68.5 MB; archived | Snippets inside Markdown files (count not measured) | Low: Markdown extraction, attribution duty, frozen since 2023 | Skip |
| LeetCode repositories | https://github.com/kamyu104/LeetCode-Solutions , https://github.com/walkccc/LeetCode , https://github.com/neetcode-gh/leetcode , https://github.com/qiyuangong/leetcode | MIT (all four) | LICENSE files read [VERIFIED] (GitHub API) | about 4.3 / 1.8 / 0.25 / 0.44 MB of `.py` | LeetCode submissions | Low: methods of `class Solution`, no docstrings (0 kept in qiyuangong; the other three not scanned); MBPP task 408 is LeetCode 373 | Skip |
| doocs/leetcode | https://github.com/doocs/leetcode | CC-BY-SA-4.0 | LICENSE read [VERIFIED] (GitHub API) | 910 MB | Solutions in many languages | Low; share-alike | Skip |
| ActiveState/code | https://github.com/ActiveState/code | MIT by default; legacy recipes keep their own licence | README read [VERIFIED] | 4,596 `.py`, about 17.4 MB | Recipes from code.activestate.com | Low to medium; licence must be checked per recipe; Python 2 share not measured | Skip for now |
| CodeSearchNet Python | https://huggingface.co/datasets/code-search-net/code_search_net | Per repository, mixed (MIT, BSD, Apache-2.0, GPL, LGPL, AGPL seen); licence texts per repository in `python_licenses.pkl` | README and card read; zip listed and two members sampled by range request [VERIFIED] | zip 940.9 MB; shards 10 to 16 MB; licence table 32.67 MB | 457,461 functions with docstrings | High after filtering: 7,069 kept plain functions (4.63 MB) per 11.4 MB shard | Use with licence filter |
| codeparrot/codeparrot-clean | https://huggingface.co/datasets/codeparrot/codeparrot-clean | No dataset licence stated; per-file `license` field (15 values seen; 59.5% of files permissive) | Card read; 6 MB sampled [VERIFIED] | 54 `.json.gz` files of 142 to 248 MB | 5,361,373 Python files from GitHub | Medium: whole files, needs function extraction; 1,407 kept (0.77 MB) per 6 MB compressed | Use with licence filter |
| Fsoft-AIC/the-vault-function | https://huggingface.co/datasets/Fsoft-AIC/the-vault-function | MIT (dataset); per-row licence list taken from The Stack | Card read; data not opened [VERIFIED] (card) | Python parquet: test 30.3 MB, validation 37.6 MB, train/small 278.5 + 275.7 MB | Function and docstring pairs | High on paper; needs `pyarrow`, which is not installed | Later, only if a parquet reader is approved |
| codeparrot/github-code | https://huggingface.co/datasets/codeparrot/github-code | Per-file `license` field, 15 licences, GPL included | Card read [VERIFIED] | 1,126 parquet files of about 288 MB, 30 languages mixed | Whole files | Low: mixed languages, parquet | Skip |
| bigcode/the-stack, the-stack-dedup, the-stack-smol | https://huggingface.co/datasets/bigcode/the-stack | Per-file licences; gated | API reports `gated=auto` [VERIFIED] (gating) | 1.43 TB / 1.00 TB / 2.9 GB | Whole files | Needs login: fails the no-credentials requirement | Skip |
| bigcode/the-stack-smol-xs | https://huggingface.co/datasets/bigcode/the-stack-smol-xs | None stated; no licence field seen | API; opening 60 KB sampled [VERIFIED] (not gated, file size) | Python file 0.97 MB | Small per-language samples | Too small; no licence metadata | Skip |
| MBPP train + validation | https://github.com/google-research/google-research/tree/master/mbpp | CC-BY-4.0 | Repository README read; file counted [VERIFIED] | 563,743 bytes | 374 + 90 tasks, 3 asserts each | High (in format) but tiny: 66.7 KB of train code | Use train and validation only |
| APPS | https://huggingface.co/datasets/codeparrot/apps | MIT label; problems from Codewars, AtCoder, Kattis, Codeforces | Card read; Codeforces terms read [VERIFIED] | train 107.1 MB, test 1,292.4 MB | Contest problems, stdin/stdout programs | Low; provenance unclear | Do not train |
| CodeContests | https://github.com/google-deepmind/code_contests | CC-BY-4.0 (data), Apache-2.0 (code); third-party parts under their own terms | README read [VERIFIED] | 7.6 GB parquet | Contest problems and submissions | Low; provenance unclear | Do not train |
| CoNaLa | https://huggingface.co/datasets/neulab/conala | MIT label; content from Stack Overflow (CC BY-SA) | Card, site and Stack Overflow licensing page read [VERIFIED] | curated train 0.52 MB; mined 160.3 MB | One-line snippets with intents | Low: not functions; share-alike source | Do not train |
| HumanEval | https://github.com/openai/human-eval | MIT | LICENSE read [VERIFIED] | 44,877 bytes | 164 test problems | Test only | Never train |

### 6.2 AI-generated data: verdict per source (clauses quoted in section 2.5)

| Source of the generated text | Governing text | Decisive clause | Verdict for training LS-LLM |
|---|---|---|---|
| Claude (apps, API, Claude Code) | Anthropic Usage Policy, Consumer Terms, Commercial Terms, help-centre article | "Use outputs to train an AI model [...] without prior authorization from Anthropic"; prohibited examples include "Models designed for open-ended text generation" and "Using Outputs as training targets for models" [VERIFIED] | Not allowed without written permission from Anthropic |
| OpenAI models (ChatGPT, API) | Terms of Use (in the EU: Europe Terms of Use); Services Agreement | "Use Output to develop models that compete with OpenAI" (Europe: "Using Output ..."); no programmatic extraction of Output; exception only for classifiers and embeddings [VERIFIED] | Unclear, so not used |
| Llama 2, CodeLlama | Llama 2 Community License | no use of output "to improve any other large language model" [VERIFIED] | Not allowed |
| Llama 3 (3.0) | Meta Llama 3 Community License | same clause [VERIFIED] | Not allowed |
| Llama 3.1, 3.3 (text models) | Llama 3.1 / 3.3 Community License | training on outputs permitted; a distributed model must have "Llama" at the beginning of its name [VERIFIED] | Clearly allowed, with a naming condition we would rather avoid |
| Llama 3.2 and Llama 4 multimodal models | Llama 3.2 / Llama 4 licence and use policy | rights "are not being granted" to individuals domiciled in the European Union [VERIFIED] | Not allowed for a team based in the EU |
| Qwen2.5-Coder 0.5B, 1.5B, 7B, 14B, 32B | Apache-2.0 | the licence text has no clause about outputs [VERIFIED] | Clearly allowed |
| Qwen2.5-Coder 3B | Qwen Research Licence | "FOR NON-COMMERCIAL PURPOSES ONLY"; "Built with Qwen" notice [VERIFIED] | Allowed for non-commercial research with the notice; use an Apache-2.0 size instead |
| DeepSeek-Coder 1.3b / 6.7b, DeepSeek-Coder-V2 | DeepSeek License Agreement 1.0 | a model trained on its synthetic data "in order to cause the other model to perform similarly to the Model" is a "Derivative of the Model"; use restrictions pass on [VERIFIED] | Allowed, with pass-through conditions on our weights |
| StarCoder, StarCoder2 | BigCode OpenRAIL-M v1 | "Licensor claims no rights in the Output"; a model built from Output is a "Modification"; Attachment A passes on [VERIFIED] | Allowed, with pass-through conditions on our weights |
| nampdn-ai/tiny-codes | tag `mit`; gated; generator not named in the card, which was read in full [VERIFIED] | none available | Unclear, so not used (also needs a login) |
| flytech/python-codes-25k | "License MIT" in the card body; origin of the code not stated (one field is described as part of an "AI response") [VERIFIED] | none available | Unclear, so not used |
| iamtarun/python_code_instructions_18k_alpaca | no licence; upstream dataset answers 401 [VERIFIED] | none available | Unclear, so not used |

A licence tag chosen by a dataset uploader does not override the terms of the model that wrote the text. For every
generator, also an allowed one, the outputs can repeat licensed code and can carry benchmark tasks (section 3).

### 6.3 Benchmarks

| Benchmark | Canonical download URL | Size | Licence | Problems | Split |
|---|---|---|---|---|---|
| HumanEval | https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz | 44,877 bytes | MIT | 164 | test only [VERIFIED] |
| MBPP | https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl | 563,743 bytes | CC-BY-4.0 | 974 | by task id: prompt 1-10 (10), test 11-510 (500), validation 511-600 (90), train 601-974 (374) [VERIFIED] |
| MBPP sanitized | https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json | 255,053 bytes | CC-BY-4.0 | 427 | same id ranges: prompt 7, test 257, validation 43, train 120 [VERIFIED] |

How published work removes overlap: exact docstring or solution match (StarCoder, 558 Python files removed);
10-gram rule (DeepSeek-Coder); similarity pruning and retraining (phi-1); a fresh unseen benchmark as a control
(Matton et al.). Sources and quotes are in section 2.7; our own counts are in section 2.8. [VERIFIED]

### 6.4 Ranked download list with exact URLs

Commits and dataset revisions are pinned, so the extracted files are the same on every rebuild. The compressed
GitHub archives are not a fixed reference: the fact-check download of the same nine commits with the same
downloader, about an hour after the earlier measurement, gave sizes that differ by 26 to 756 bytes per archive
(GitHub builds these archives on request), so compare extracted files, not archive sizes or hashes. The benchmark,
Hugging Face and python.org files had unchanged sizes. Byte counts were measured on
2026-11-13. Total: about 215 MB (0.9 + 13.9 + 99.7 + 100) [VERIFIED]. Expected kept text before deduplication:
2.3 MB (core) + about 22 MB (CodeSearchNet: 27 MB times a permissive share of about 80%, the recount on the opening
4.5% of its licence table; 16 MB at a cautious 60%, 25 MB at the published 92% for the whole dataset) + about 13 MB
(codeparrot-clean) = about 37 MB (31 to 40 MB); all but the 2.3 MB are extrapolations from one shard each
[UNVERIFIED: extrapolation; the permissive share per function was not measured].

| Rank | What | Exact URL | Bytes | Notes |
|---|---|---|---|---|
| 1 | HumanEval (evaluation only) | https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz | 44,877 | needed by the decontamination filter |
| 2 | MBPP | https://raw.githubusercontent.com/google-research/google-research/f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81/mbpp/mbpp.jsonl | 563,743 | train and validation ids may be trained on; test ids never |
| 3 | MBPP sanitized | https://raw.githubusercontent.com/google-research/google-research/f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81/mbpp/sanitized-mbpp.json | 255,053 | |
| 4 | TheAlgorithms/Python | https://codeload.github.com/TheAlgorithms/Python/tar.gz/30507bbe41397065728167fb09aa820f9fce8dc7 | 8,457,964 | MIT |
| 5 | keon/algorithms | https://codeload.github.com/keon/algorithms/tar.gz/bd1d152f9ec7331c69be662831610b003b10fc7a | 248,876 | MIT |
| 6 | exercism/python | https://codeload.github.com/exercism/python/tar.gz/479f11b44b02337b0413e805b948491c2179615a | 1,498,911 | MIT |
| 7 | dgilland/pydash | https://codeload.github.com/dgilland/pydash/tar.gz/b6bd031de47134b8e3c3977fbe65bf13cc51a205 | 208,683 | MIT |
| 8 | OmkarPathak/pygorithm | https://codeload.github.com/OmkarPathak/pygorithm/tar.gz/cee38ed47e671171c3a30d12d5ec288f19c206d8 | 2,856,261 | MIT |
| 9 | Suor/funcy | https://codeload.github.com/Suor/funcy/tar.gz/a876c7e1cf11e7c746e91fce4632ebfa93865428 | 127,948 | BSD-3-Clause |
| 10 | more-itertools | https://codeload.github.com/more-itertools/more-itertools/tar.gz/ceafee8ce96f4bc1eb74c528bc247c91d648a0bb | 150,307 | MIT |
| 11 | mahmoud/boltons | https://codeload.github.com/mahmoud/boltons/tar.gz/d31d821d56c8f7756af9c30a864077f2e5855321 | 282,755 | BSD-3-Clause |
| 12 | pytoolz/toolz | https://codeload.github.com/pytoolz/toolz/tar.gz/451af60dec590a6010e2babdbf391ea8f815122f | 79,075 | BSD-3-Clause |
| 13 | CodeSearchNet Python, members read by range request (do not download the whole zip) | https://huggingface.co/datasets/code-search-net/code_search_net/resolve/fdc6a9e39575768c27eb8a2a5f702bf846eb4759/data/python.zip | 99.7 MB of 940,909,997 | members: `python_licenses.pkl` (32.67 MB), `python/final/jsonl/valid/python_valid_0.jsonl.gz` (11.43), `python/final/jsonl/test/python_test_0.jsonl.gz` (10.72), `python/final/jsonl/train/python_train_0.jsonl.gz` (15.97), `python_train_1.jsonl.gz` (15.45), `python_train_2.jsonl.gz` (13.47) |
| 14 | codeparrot-clean, file 1, request header `Range: bytes=0-99999999` | https://huggingface.co/datasets/codeparrot/codeparrot-clean/resolve/35a59fb025bc0a102f7d96eac09d145b896d487b/file-000000000001.json.gz | 100,000,000 of 246,127,841 | gzip JSON lines; the stream ends mid-record, drop the last partial line |
| 15 | CPython 3.12.7 source (optional) | https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tar.xz | 20,444,032 | PSF-2.0; use `Lib/` without test directories |
| 16 | networkx (optional) | https://codeload.github.com/networkx/networkx/tar.gz/6855cd586982cc2e4710b8eb87251ccf56674bf1 | not measured | BSD-3-Clause |

Bytes of ranks 1 to 3 and 13 to 15 were confirmed in the fact-check [VERIFIED]. Ranks 4 to 12 show the fact-check
download (13,910,780 bytes together); the earlier measurement gave 8,457,333 / 248,796 / 1,499,362 / 208,248 /
2,855,505 / 127,908 / 150,137 / 282,466 / 79,049 (13,908,804 together).

Reading notes for the implementer:

- Rank 13 needs a seekable file over HTTP range requests so that `zipfile` can open single members. The class
  `HttpRangeFile` in `research/08_dataset_measurements.py` (about 30 lines, standard library) does this and was used
  for every CodeSearchNet number in this file.
- `python_licenses.pkl` must not be opened with plain `pickle.load`. Refuse every global, so that only dict, list,
  tuple and str can be built (not yet run on the whole file). The result maps `owner/name` to a list of
  `(path, licence text)` tuples:

```python
import pickle

class NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} refused")

with open("python_licenses.pkl", "rb") as fh:
    licences = NoGlobals(fh).load()
```

- Parquet copies exist for most of these sets, but the virtualenv has no parquet reader and the rules forbid
  `pip install`; every URL above is readable with `tarfile`, `zipfile`, `gzip`, `lzma` and `json`.
- Deduplicate across sources by a hash of the whitespace-normalised function body: the same GitHub code can be in
  CodeSearchNet, codeparrot-clean and a hand-picked repository.

## 7. Sources

Papers

| Title | Authors | Year | URL |
|---|---|---|---|
| Evaluating Large Language Models Trained on Code (Codex, HumanEval) | Mark Chen, Jerry Tworek, Heewoo Jun et al. (58 authors) | 2021 | https://arxiv.org/abs/2107.03374 |
| Program Synthesis with Large Language Models (MBPP) | Jacob Austin, Augustus Odena, Maxwell Nye et al. (11 authors) | 2021 | https://arxiv.org/abs/2108.07732 |
| CodeSearchNet Challenge: Evaluating the State of Semantic Code Search | Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, Marc Brockschmidt | 2019 | https://arxiv.org/abs/1909.09436 |
| The Stack: 3 TB of permissively licensed source code | Denis Kocetkov, Raymond Li, Loubna Ben Allal et al. | 2022 | https://arxiv.org/abs/2211.15533 |
| The Vault: A Comprehensive Multilingual Dataset for Advancing Code Understanding and Generation | Dung Nguyen Manh, Nam Le Hai, Anh T. V. Dau, Anh Minh Nguyen, Khanh Nghiem, Jin Guo, Nghi D. Q. Bui | 2023 | https://arxiv.org/abs/2305.06156 |
| StarCoder: may the source be with you! | Raymond Li, Loubna Ben Allal, Yangtian Zi et al. (67 authors) | 2023 | https://arxiv.org/abs/2305.06161 |
| DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence | Daya Guo, Qihao Zhu, Dejian Yang et al. (13 authors) | 2024 | https://arxiv.org/abs/2401.14196 |
| Textbooks Are All You Need (phi-1) | Suriya Gunasekar, Yi Zhang, Jyoti Aneja et al. (19 authors) | 2023 | https://arxiv.org/abs/2306.11644 |
| Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models | Martin Riddell, Ansong Ni, Arman Cohan | 2024 | https://arxiv.org/abs/2403.04811 |
| On Leakage of Code Generation Evaluation Datasets | Alexandre Matton, Tom Sherborne, Dennis Aumiller et al. (10 authors) | 2024 | https://arxiv.org/abs/2407.07565 |
| Measuring Coding Challenge Competence With APPS | Dan Hendrycks, Steven Basart, Saurav Kadavath et al. (11 authors) | 2021 | https://arxiv.org/abs/2105.09938 |
| Competition-Level Code Generation with AlphaCode (CodeContests) | Yujia Li, David Choi, Junyoung Chung et al. (26 authors) | 2022 | https://arxiv.org/abs/2203.07814 |
| Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow (CoNaLa) | Pengcheng Yin, Bowen Deng, Edgar Chen et al. (5 authors) | 2018 | https://arxiv.org/abs/1805.08949 |
| CodeIPPrompt: Intellectual Property Infringement Assessment of Code Language Models | Zhiyuan Yu, Yuhao Wu, Ning Zhang, Chenguang Wang, Yevgeniy Vorobeychik, Chaowei Xiao | 2023 (ICML, PMLR 202) | https://proceedings.mlr.press/v202/yu23g.html |
| Curriculum Learning for Small Code Language Models | Marwa Naïr, Kamel Yamani, Lynda Said Lhadj, Riyadh Baghdadi | 2024 | https://arxiv.org/abs/2407.10194 |

Titles, author counts and years of the thirteen arXiv entries above the last two rows were compared with the arXiv
API [VERIFIED]; the last two rows were added in the fact-check.

Code, datasets and models (licence as read at the source)

| Name | Kind | URL | Licence |
|---|---|---|---|
| TheAlgorithms/Python | code | https://github.com/TheAlgorithms/Python | MIT |
| keon/algorithms | code | https://github.com/keon/algorithms | MIT |
| exercism/python | code | https://github.com/exercism/python | MIT |
| more-itertools | code | https://github.com/more-itertools/more-itertools | MIT |
| pytoolz/toolz | code | https://github.com/pytoolz/toolz | BSD-3-Clause |
| mahmoud/boltons | code | https://github.com/mahmoud/boltons | BSD-3-Clause |
| Suor/funcy | code | https://github.com/Suor/funcy | BSD-3-Clause |
| dgilland/pydash | code | https://github.com/dgilland/pydash | MIT |
| OmkarPathak/pygorithm | code | https://github.com/OmkarPathak/pygorithm | MIT |
| networkx | code | https://github.com/networkx/networkx | BSD-3-Clause |
| CPython | code | https://github.com/python/cpython | PSF-2.0 (documentation examples 0BSD) |
| Chalarangelo/30-seconds-of-python | code | https://github.com/Chalarangelo/30-seconds-of-python | CC-BY-4.0 |
| kamyu104/LeetCode-Solutions, walkccc/LeetCode, neetcode-gh/leetcode, qiyuangong/leetcode | code | https://github.com/kamyu104/LeetCode-Solutions | MIT (each) |
| doocs/leetcode | code | https://github.com/doocs/leetcode | CC-BY-SA-4.0 |
| ActiveState/code | code | https://github.com/ActiveState/code | MIT by default; legacy recipes per recipe |
| CodeSearchNet | dataset | https://github.com/github/CodeSearchNet , https://huggingface.co/datasets/code-search-net/code_search_net | tooling MIT; data per repository (table in `python_licenses.pkl`) |
| codeparrot/codeparrot-clean | dataset | https://huggingface.co/datasets/codeparrot/codeparrot-clean | none stated; per-file `license` field |
| codeparrot/github-code | dataset | https://huggingface.co/datasets/codeparrot/github-code | tag `other`; per-file `license` field |
| Fsoft-AIC/the-vault-function | dataset | https://huggingface.co/datasets/Fsoft-AIC/the-vault-function | MIT; per-row licence list |
| bigcode/the-stack (and -dedup, -smol, -smol-xs) | dataset | https://huggingface.co/datasets/bigcode/the-stack | tag `other`; gated except -smol-xs |
| HumanEval | benchmark | https://github.com/openai/human-eval | MIT |
| MBPP | benchmark | https://github.com/google-research/google-research/tree/master/mbpp | CC-BY-4.0 |
| evalplus/humanevalplus, evalplus/mbppplus | benchmark | https://huggingface.co/datasets/evalplus/humanevalplus | tag `apache-2.0` (licence text not opened) |
| APPS | dataset | https://github.com/hendrycks/apps , https://huggingface.co/datasets/codeparrot/apps | MIT label; scraped sources |
| CodeContests | dataset | https://github.com/google-deepmind/code_contests | CC-BY-4.0 data, Apache-2.0 code; third-party parts under own terms |
| CoNaLa | dataset | https://conala-corpus.github.io/ , https://huggingface.co/datasets/neulab/conala | MIT label; Stack Overflow content CC BY-SA |
| nampdn-ai/tiny-codes | dataset (synthetic) | https://huggingface.co/datasets/nampdn-ai/tiny-codes | tag `mit`; gated; generator not named |
| flytech/python-codes-25k | dataset (origin not stated) | https://huggingface.co/datasets/flytech/python-codes-25k | "MIT" in the card body |
| iamtarun/python_code_instructions_18k_alpaca | dataset (origin not checkable) | https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca | none stated |
| Qwen2.5-Coder 0.5B / 1.5B / 7B / 14B / 32B Instruct | model | https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct | Apache-2.0 |
| Qwen2.5-Coder-3B-Instruct | model | https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct | Qwen Research Licence (non-commercial) |
| Llama 2, 3, 3.1, 3.2, 3.3, 4 | model licences | https://github.com/meta-llama/llama-models/tree/main/models | Llama Community Licences (per version) |
| CodeLlama | model | https://github.com/meta-llama/codellama | Llama 2 Community License |
| DeepSeek-Coder, DeepSeek-Coder-V2 | model | https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL | DeepSeek License Agreement 1.0 |
| StarCoder, StarCoder2 | model | https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement | BigCode OpenRAIL-M v1 |

Terms and policies

| Title | Publisher | Date on the page | URL |
|---|---|---|---|
| Usage Policy | Anthropic | effective Nov 12, 2026 | https://www.anthropic.com/legal/aup |
| Consumer Terms of Service | Anthropic | effective Oct 8, 2025 | https://www.anthropic.com/legal/consumer-terms |
| Commercial Terms of Service | Anthropic | effective Jun 17, 2025 | https://www.anthropic.com/legal/commercial-terms |
| Can I use my Outputs to train an AI model? | Anthropic help centre | March 16, 2026 | https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model |
| Terms of Use (rest of world) | OpenAI | effective January 1, 2026 (Wayback snapshot 2026-11-09) | https://web.archive.org/web/20261109060701/https://openai.com/policies/row-terms-of-use/ |
| Europe Terms of Use (EEA, Switzerland, UK) | OpenAI | updated January 16, 2026 (Wayback snapshot 2026-10-28) | https://web.archive.org/web/20261028153406/https://openai.com/policies/eu-terms-of-use/ |
| Services Agreement | OpenAI | effective January 1, 2026 (Wayback snapshot 2026-11-10) | https://web.archive.org/web/20261110000152/https://openai.com/policies/services-agreement/ |
| What is the license for the content I post? | Stack Overflow | read 2026-11-13 | https://stackoverflow.com/help/licensing |
| Terms and Conditions | Codeforces | read 2026-11-13 | https://codeforces.com/terms |

## Verification log

Fact-check of 2026-11-13, 19:05 to 19:45, by a second agent. 25 claims were compared with the primary source or
measured again: 15 confirmed, 10 corrected. One explanation was deleted as unsupported (line 18). One line per claim:
claim (section), verdict, URL. Notes on method and on what was not checked follow the list.

1. Anthropic Usage Policy, effective Nov 12, 2026, forbids to "Use outputs to train an AI model ... without prior authorization from Anthropic" (2.5): confirmed. https://www.anthropic.com/legal/aup
2. Anthropic Consumer Terms (Oct 8, 2025) and Commercial Terms D.4 (Jun 17, 2025), quoted clauses and dates (2.5): confirmed. https://www.anthropic.com/legal/consumer-terms , https://www.anthropic.com/legal/commercial-terms
3. Anthropic help-centre article dated March 16, 2026: "without our written permission", allowed and prohibited lists (2.5): confirmed. https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model
4. OpenAI Terms of Use, effective January 1, 2026 (2.5, 6.2): quotes confirmed; corrected the governing text for a team in the EU: rest-of-world Terms of Use -> Europe Terms of Use (updated January 16, 2026), which hold the same two restrictions; verdict unchanged. https://web.archive.org/web/20261028153406/https://openai.com/policies/eu-terms-of-use/
5. OpenAI Services Agreement, effective January 1, 2026: restriction (e) and the "Permitted Exception" (2.5): confirmed. https://web.archive.org/web/20261110000152/https://openai.com/policies/services-agreement/
6. Llama licences (2.5, 6.2): Llama 2 and Llama 3 forbid using outputs to improve another large language model; CodeLlama points to the Llama 2 licence; Llama 3.1, 3.2, 3.3 and 4 drop that clause and require a model name beginning with "Llama"; EU exclusion for multimodal models of 3.2 and 4: confirmed. https://github.com/meta-llama/llama-models/tree/main/models
7. Qwen2.5-Coder 0.5B, 1.5B, 7B, 14B and 32B Instruct are Apache-2.0 (five byte-identical licence files without a clause about outputs); 3B is under the Qwen Research Licence (2.5, 6.2): confirmed. https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE
8. Disk needed for Qwen2.5-Coder weights (5, rank 9): corrected "about 1 to 5 GB [UNVERIFIED]" -> 4-bit GGUF 1.12 GB (1.5B) and 4.68 GB (7B); safetensors 3.09 GB and 15.23 GB. https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/tree/main
9. DeepSeek License Agreement 1.0: a model trained on the model's synthetic data is a "Derivative", use restrictions pass on, same file for DeepSeek-Coder-V2 (2.5, 3, 6.2): corrected the quote: elided "[...]" -> full wording with "in order to cause the other model to perform similarly to the Model"; verdict unchanged. https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL
10. BigCode OpenRAIL-M v1: definition of "Modifications of the Model", section 6 on Output, sections 5.1 and 5.2 (2.5): confirmed. https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement/blob/main/app.py
11. The three Hugging Face synthetic sets do not name their generator (2.5, 3, 6.2): confirmed for flytech and iamtarun; corrected for tiny-codes: "the raw README answers 401 ... not named in the text we could read" -> the whole card is readable without login and names no generator. https://huggingface.co/datasets/nampdn-ai/tiny-codes/resolve/main/README.md
12. HumanEval: MIT, 164 problems, 44,877 bytes. MBPP: CC-BY-4.0, 974 problems, 563,743 bytes, split 10 / 500 / 90 / 374 with test ids 11 to 510; sanitized 427 problems (2.6, 6.3): confirmed by downloading and counting. https://github.com/openai/human-eval , https://github.com/google-research/google-research/tree/master/mbpp
13. The nine hand-picked repositories are MIT or 3-clause BSD (2.1, 6.1): confirmed, nine LICENSE files read. https://github.com/TheAlgorithms/Python (the other eight URLs are in table 6.1)
14. Measured yield of the nine repositories: 3,639 kept functions (2.30 MB), 3,255 functions with doctests, 10.1 MB of Python in 2,918 files (2.1, 6.1): confirmed, `research/08_dataset_measurements.py` run again with identical output. https://github.com/TheAlgorithms/Python
15. Pinned GitHub tarball sizes (2.1, 6.4): corrected, for example TheAlgorithms 8,457,333 -> 8,457,964 bytes, nine archives together 13,908,804 -> 13,910,780 (still 13.9 MB); "can be rebuilt byte for byte" -> only the extracted files are fixed by the commit. https://codeload.github.com/TheAlgorithms/Python/tar.gz/30507bbe41397065728167fb09aa820f9fce8dc7
16. CodeSearchNet Python: 412,178 / 23,107 / 22,176 functions; zip of 940,909,997 bytes at the pinned revision, listable with 2,656 bytes of range requests, no login; S3 link answers 301 / 403; licence documentation quotes (2.2): confirmed. https://huggingface.co/datasets/code-search-net/code_search_net
17. Licence label versus provenance (2.4): APPS, CodeContests and Codeforces quotes and sizes confirmed; corrected CoNaLa "mined set 598,237 pairs (160.3 MB)" -> 598,237 pairs on the CoNaLa site, 593,891 rows in the 160.3 MB Hugging Face file. https://huggingface.co/datasets/neulab/conala
18. CodeSearchNet licence-table sample (2.2, 3, 6.4): corrected "269 entries, `owner/name -> (path, licence text)`, 83% permissive" -> 748 repositories with 890 licence files, `owner/name -> list of (path, licence text)`, 82.1% permissive only and 16.7% GPL, LGPL or AGPL by root licence files. Unsupported and deleted: "the pickle seems to store a repeated text once" (890 texts are stored, 690 of them distinct, and the sample has no back-reference opcode). https://huggingface.co/datasets/code-search-net/code_search_net/resolve/fdc6a9e39575768c27eb8a2a5f702bf846eb4759/data/python.zip
19. CodeSearchNet validation shard: 23,107 functions from 679 repositories, 13,384 methods, 7,069 kept plain functions (4.63 MB) (2.2): confirmed, script run again with identical output; the assumed 30,000 rows per train shard were confirmed on train_0, which gives a second yield (11,781 kept, 8.15 MB, 85 repositories). https://huggingface.co/datasets/code-search-net/code_search_net/resolve/fdc6a9e39575768c27eb8a2a5f702bf846eb4759/data/python.zip
20. codeparrot-clean: 5,361,373 files with a per-file `license` field, not gated; sample of 2,413 files 59.5% permissive, 1,407 kept, about 235 kept per compressed MB (2.3): confirmed, script run again; corrected "54 files of about 246 MB" -> 142 to 248 MB. https://huggingface.co/datasets/codeparrot/codeparrot-clean
21. Benchmark overlap table and the inspected hits, for example pygorithm with MBPP test tasks 45, 54, 152, 316, 428 and HumanEval/123 in TheAlgorithms (2.8): confirmed, table reproduced and hits listed again. https://github.com/OmkarPathak/pygorithm
22. Wording stronger than the measurement (1, 2.8, 5): corrected "no source holds a copy of the HumanEval or MBPP files" -> nothing scanned holds a copy (bulk sources: samples only); "13% of permissive files are Python 2" -> 13% do not parse, cause not measured; "only two bulk sources reach the 20 to 80 MB target" -> expected by extrapolation; "the only human-written in-format data" -> among the sources examined here; "no published number exists at our scale" -> none was found in our search. https://huggingface.co/datasets/codeparrot/codeparrot-clean
23. Riddell et al.: 12.2% / 18.9% of HumanEval and 3.6% / 20.8% of MBPP solutions seen; StarCoderBase-15.5B 41.6 -> 33.8 on MBPP after removing the seen 20.8% (2.7): confirmed (text of section 4.1 and Table 1). https://arxiv.org/abs/2403.04811
24. Published decontamination quotes (2.7, 6.3): MBPP paper section 4.8 (32 of 974 problems, 3.3%, 91.5%); Codex paper (hand-written problems, Codeforces repositories); StarCoder (558 Python files removed); DeepSeek-Coder (10-gram rule); phi-1 (13-gram check, 40% pruning, 1.3 B parameters); Matton et al. (+9% HumanEval, +2% MBPP, LBPP unchanged): confirmed. https://arxiv.org/abs/2108.07732 , https://arxiv.org/abs/2107.03374 , https://arxiv.org/abs/2305.06161 , https://arxiv.org/abs/2401.14196 , https://arxiv.org/abs/2306.11644 , https://arxiv.org/abs/2407.07565
25. "Not found in our search" (4): corrected by adding what a search does find: a published licence composition of CodeSearchNet by GitHub API lookups (Yu et al. 2023: 91.88% permissive, 6.37% strong copyleft) and grammar-generated Python for models of 1 million parameters (Naïr et al. 2024); the narrower statements (count from `python_licenses.pkl`; language-model-written corpus below 1 M parameters) stand as "not found". https://proceedings.mlr.press/v202/yu23g.html , https://arxiv.org/abs/2407.10194

Method of the licence-table recount (line 18), so that it can be repeated: `zipfile` over HTTP range requests finds
`python_licenses.pkl` in the pinned zip (deflate, 32,666,588 bytes stored, 134,172,919 unpacked); 2,097,152 stored
bytes from the start of the member are inflated with `zlib.decompressobj(-15)` and cut to 6,000,000 bytes;
`pickletools.genops` yields the opcodes, and a small stack machine that knows only PROTO, EMPTY_DICT, EMPTY_LIST,
MARK, BINUNICODE, BINPUT, LONG_BINPUT, TUPLE2, APPEND and APPENDS rebuilds keys and lists without `pickle.load`; the
cut falls inside the 749th repository, which is dropped. A licence file is classed by keywords in its opening 400
characters (affero; lesser or library; GNU general public license; Mozilla; Apache), otherwise by full-text phrases
(MIT "permission is hereby granted, free of charge", BSD "redistribution and use in source and binary forms", ISC,
Unlicense, then the GNU names). 137 of the 140 GPL, LGPL and AGPL files were found by their heading. The script ran
from a scratch directory with `.venv/bin/python -I` and is not in the repository.

Also opened while checking, not counted above (marked `[VERIFIED]` in the text): The Vault card; the github-code
card; gating flags of The Stack family; the 967,268-byte file of the-stack-smol-xs; CPython licence text and tarball
size; exercism file counts at the pinned commit; the ActiveState README; GitHub API licence values of the LeetCode
repositories, 30-seconds-of-python and doocs/leetcode; licence tags of Granite, Devstral, phi-4, DeepSeek-R1,
Codestral and CodeGemma; the absence of parquet readers in the virtualenv; titles, author counts and years of the
thirteen arXiv papers.

Not re-checked (no mark in the text): `.py` counts of CPython `Lib/`, networkx and three LeetCode repositories; file
sizes of The Vault, github-code, The Stack family and the EvalPlus sets; the field list of the-stack-smol-xs; the
dataset-viewer refusal for The Vault; the row counts of CodeSearchNet train shards 1 to 13; the disk-space history; the
permissive share of the whole CodeSearchNet licence table.

Other edits made in this pass: rank 4 names the list structure of the licence table; rank 8 points to the project
rule of 2026-11-13 19:05 in `CLAUDE.md`; the expected kept text in section 6.4 was recalculated with the recount
(about 37 MB, 31 to 40 MB, instead of 31 to 37 MB); a second CodeSearchNet shard (train_0) was measured with the
functions of the project script and added to section 2.2; Yu et al. 2023, Naïr et al. 2024 and the OpenAI Europe
Terms of Use were added to section 7.
