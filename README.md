# LecTrace demo

Two step-through lectures, built to show what lectrace does. [**See it here**](https://praisegee.github.io/lectrace-demo)

## Run

```sh
uv sync
uv run lectrace serve
```

Opens at http://localhost:7000 and reloads when you save.

## Files

| File                   | What it is                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `01_attention.py`      | Multi-head self-attention on six words of a clinical note, one line at a time. Ends with a convolution clip. |
| `02_write_your_own.py` | The five directives, the rendering API, and how to deploy.                                                   |
| `_helpers.py`          | Charts and tables. The underscore keeps it out of the trace.                                                 |
| `assets/`              | Figure and clip used by file 01. Regenerate with `uv run --dev python scripts/build_assets.py`.              |

Either file also runs as an ordinary script:

```sh
uv run python 01_attention.py
```

## Deploy

```sh
uv run lectrace build     # static site in _site/
uv run lectrace init      # GitHub Actions workflow, then push
```

**Enable Pages in the repo settings with source set to GitHub Actions.**
