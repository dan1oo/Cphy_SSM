# Efficient Sequence Modelling with Structured State Spaces (S4)

## Introduction
This project implements a simplified version of the **Structured State-Space Model (S4)** introduced by Gu et al. (2022). The S4 model is designed to efficiently handle long-range dependencies in sequences by leveraging signal processing tools, state space theory, and fast convolution via the Fourier domain. Unlike RNNs and Transformers, S4 achieves linear time complexity in sequence length while still capturing complex, long-range dependencies.

S4 is part of a growing class of models that aim to overcome the limitations of attention-based architectures by revisiting classical system dynamics. It uses specially structured matrices (HiPPO operators, diagonal plus low-rank components) to maintain and evolve hidden states over time. These ideas are drawn from control theory and combine strong theoretical foundations with practical efficiency.

In our project, we study how S4 performs on synthetic tasks like:
- **Memory task:** remembering and reproducing a value from earlier in the sequence.
- **Previous-bit task:** predicting a specific bit from a prior time step.

We implemented the core model components, designed task generators, and evaluated how well S4 models long-term dependencies in toy data.

## Implementation / Algorithm

Our implementation is organized into two main components:

### 1. S4 Model core (in `SSM/`)
- `model.py`: Computes the S4 kernel, performs convolution via FFT, and applies the model to input sequences.
- `hippo.py`: Constructs the HiPPO matrix $A$ and defines its NPLR (Normal Plus Low Rank) decomposition.
- `helpers.py`: Implements the Cauchy kernel computation, FFT-based convolution, and other linear algebra tools.

We used the **DPLR (Diagonal Plus Low-Rank)** structure from the S4 paper to stabilize and accelerate kernel generation.

### 2. Synthetic Task Setup (in `Training/`)
- `generatedata.py`: Main entry point for running experiments. Takes arguments for task, sequence length, model type, etc.
- `generator_memory.py`: Defines the memory task (model must recall a token seen earlier in the input)
- `generator_prevbit.py`: Defines the previous-bit task (predict a past bit from a fixed index).
- Training notebooks (`attempt*.ipynb`) shows our earlier runs and test code.

Each task uses random binary or continuous sequences to test how well S4 captures structure and memory over long ranges.

## Package Installation and Examples
This project uses 

## Reflection and Future Work


### References