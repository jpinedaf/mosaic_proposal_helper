# mosaic_proposal_helper
![PyPI - License](https://img.shields.io/pypi/l/mosaic_proposal_helper?color=green)
![PyPI - Version](https://img.shields.io/pypi/v/mosaic_proposal_helper)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjpinedaf%2Fmosaic_proposal_helper%2Fmaster%2Fpyproject.toml)


Helpers to generate and visualize mosaic pointings for interferometers.

This repository now follows a src-layout package structure:

- `src/mosaic_proposal_helper/`: importable package code
- `examples/`: runnable examples
- `examples/data/`: data and outputs used by examples

## Installation

Installation using pip:
```python
pip install mosaic_proposal_helper
```

From the repository root:

```bash
python -m pip install -e .
```

## Usage

```python
from astropy import units as u
from mosaic_proposal_helper import get_offsets, compute_pointings, pb_noema

PB = pb_noema(115 * u.GHz)
offsets = get_offsets(width=2.4 * u.arcmin, height=1.2 * u.arcmin, pb=PB)
```

Run the example:

```bash
python examples/example_pointing.py
```

## Credits

Developed by Jaime E Pineda ([@jpinedaf](http://github.com/jpinedaf)).
