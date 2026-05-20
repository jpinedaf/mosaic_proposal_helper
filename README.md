# mosaic_proposal_helper

Helpers to generate and visualize mosaic pointings for interferometers.

This repository now follows a src-layout package structure:

- `src/mosaic_proposal_helper/`: importable package code
- `examples/`: runnable examples
- `examples/data/`: data and outputs used by examples

## Installation

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
