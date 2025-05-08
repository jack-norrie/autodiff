# AutoDiff

An implementation of automatic differentiation (`auto`), with a neural networks sub-package (`nn`).

## Overview

This project implements an automatic differentiation engine and neural network library from scratch using only base Python. The only dependencies (matplotlib, PyQt6) are used solely for visualization in the example scripts. This was done for pedagogical reasons. The objective of this library was not to create a highly optimized and parallelized auto-differentiation library with hardware acceleration support. The main purpose was to build on my own intuitions about backpropagation through building.

## Features

- **Automatic Differentiation Engine**: Implementation of reverse-mode automatic differentiation
- **Neural Network Components**:
  - Vector and Matrix classes with autograd support
  - Linear layers with customizable activation functions
  - Sequential model for building multi-layer networks
  - Various activation functions (ReLU, Sigmoid, Tanh)
  - Weight initialization strategies (He, Xavier, LeCun)
- **Optimizers**:
  - SGD (with momentum)
  - Adam
  - RMSProp
  - AdaGrad

## Project Structure

```
autodiff/
├── src/
│   ├── auto/           # Automatic differentiation engine
│   ├── nn/             # Neural network components
│   │   ├── components/ # Layer implementations
│   │   └── ...         # Activations, optimizers, etc.
│   └── functions/      # Mathematical functions
└── main.py             # Example usage
```

## Installation

If you are using [uv](https://docs.astral.sh/uv/getting-started/installation/) then run

```
uv install
source .venv/bin/activate
```

Otherwise, setup and activate a virtual environment using Python 3.13+, and install the requirements:

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

The project includes examples demonstrating:

1. Linear regression with synthetic data
2. Non-linear function approximation using multi-layer perceptrons

To run the examples:

```bash
python main.py
```

## License

[MIT License](LICENSE)
