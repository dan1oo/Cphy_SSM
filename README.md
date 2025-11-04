# Cphy_SSM

### Description

This project is based on Albert Gu's paper 'Efficiently Modeling Long Sequences with Structred State Spaces': `https://arxiv.org/abs/2111.00396`. We aim to implement a single head of this model, and apply it to a simple long sequence prediction challenge and evaulate and visualize it's performance. Our goal is to understand and implement the structured S4 state space model and allow it to run on a local device following all the nuances and matrix operations outline in the paper. We want to focus on long range dependencies. We want to compute NPLR (normal + low rank parameterization) using the 'cauchy kernel'.

### Directory Structure

- `demos.ipynb` - Script for Model implementation demo on arbitrary simulation
- `.\SSM` - Containerized State Space Model (in one class ideally)
- `.\Simulations`- Simulation Scripts 
- `.\Helpers` - Helper functions for data + metric testing 
- `.\Data` - Data

### Plan of Implementation

Week 1. Implement Model V1 Basic SSM, Preprocess Data/setup rough simulation
Week 2. Training + Test Model on simulation and debug/optimize, Start visualization
Week 3. Metric analysis + presentation creation

### Contributions

- Creating project proposal: Lily, Daniel
- Model Development - Daniel
- Simulation Scripting - Lily
- Data preprocessing and testing - Nurali
- Visualization - Daniel
- Presentation - Lily

### Notes and Resources

`https://huggingface.co/blog/lbourdois/get-on-the-ssm-train`
`https://arxiv.org/abs/2111.00396`
