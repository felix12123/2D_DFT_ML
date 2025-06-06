# 2D ML DFT

This repository contains a proof-of-concept implementation of bringing neural density functional theory of hard particles to higher dimensions (here 2D). It is discussed in the following preprint:

**Neural Density Functional Theory in Higher Dimensions with Convolutional Layers**
*Felix Glitsch, Jens Weimar, Martin Oettel, [Phys. Rev. E **111**, 055305](https://doi.org/10.1103/PhysRevE.111.055305), [arXiv:2502.13717 [cond-mat.stat-mech]](https://arxiv.org/abs/2502.13717)*

## Setup

This repository uses `uv` as environment manager. It can be installed with

```text
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then one can install all dependencies (including `PyTorch`) with

```text
uv sync
source .venv/bin/activate
```

and run the main file with

```text
uv run main.py
```

or execute the accompanying `jupyter` notebook.

## Tests

Some simple tests to check the basic functionality of the repository are implemented using pytest.
Run tests with

```text
pytest
```
