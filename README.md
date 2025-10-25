# Linear Triangular Element Assembler

Small Python library implementing simple finite-element building blocks for linear triangular elements on rectangular meshes:
- mesh generation (`src/finite_elements/triangulation.py`)
- linear triangular element data (`src/finite_elements/elements.py`)
- global FEM assembler for mass and stiffness matrices (`src/finite_elements/assembler.py`)
- unit tests in `test/test_linear_assembler.py`

## Features

- Regular and irregular rectangular mesher
- Linear triangular element support
- Assembly of mass and stiffness matrices
- Tests that verify exactness for linear functions and timing examples

## Requirements

- Python 3.8+
- pip

(Use a virtual environment on Windows: `python -m venv .venv` and `.\.venv\Scripts\activate`)

## Installation

1. Clone repository
2. Install dependencies (if any):
```bash
pip install -r requirements.txt
