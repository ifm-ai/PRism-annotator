# PRism Annotator

PRism Annotator labels GitHub pull-request records stored in Parquet files. It
uses a local Hugging Face model through vLLM and writes structured annotations
for domain, change attributes, and task complexity.

The output preserves the input directory layout and all original columns. It
adds `annotation`, `annotation_parse_error`, `prompt_was_truncated`,
`prompt_tokens_original`, and `prompt_tokens_kept`. Existing output files are
skipped, so interrupted runs can be resumed.

## Setup

Use a Linux machine with a vLLM-supported GPU and a compatible CUDA driver.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Pass either one Parquet file or a directory containing Parquet files:

```bash
python build_prompt.py data/input \
  --output-dir data/annotated \
  --template-file prompt_template.txt \
  --model-name MODEL_ID_OR_PATH
```

The convenience wrapper accepts the same required values positionally:

```bash
bash build_prompt.sh data/input data/annotated MODEL_ID_OR_PATH
```

Tune the wrapper with environment variables when needed:

```bash
TP=2 MNS=128 MNBT=65536 MAX_COMPLETION_TOKENS=8192 \
  bash build_prompt.sh data/input data/annotated MODEL_ID_OR_PATH
```

`TP` is tensor parallelism, `MNS` is vLLM's maximum concurrent sequences, and
`MNBT` is its maximum batched-token budget. The defaults are `2`, `128`, and
`65536`.

## Slurm arrays

When `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT` are present, input files
are divided round-robin across array tasks. For example:

```bash
sbatch --array=0-7 --gres=gpu:2 \
  --wrap='bash build_prompt.sh data/input data/annotated MODEL_ID_OR_PATH'
```

Each task should use a shared output directory. A row with
`code.debug.git_error` is retained but left unannotated with
`annotation_parse_error="skipped_git_error"`.
