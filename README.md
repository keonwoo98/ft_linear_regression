# ft_linear_regression

An introduction to machine learning - Simple linear regression implementation using gradient descent.

## Overview

Predict car prices based on mileage using **linear regression** with **gradient descent** optimization.

**Linear Hypothesis:** `price = θ₀ + θ₁ × mileage`

## Quick Start

```bash
# 1. Train the model
python3 src/train.py

# 2. Make predictions
python3 src/predict.py
```

## Project Structure

```
ft_linear_regression/
├── data/
│   └── data.csv          # Training data (24 samples)
├── src/
│   ├── train.py          # Training program
│   ├── predict.py        # Prediction program
│   └── utils.py          # Helper functions
├── bonus/
│   ├── visualize.py      # Data visualization
│   └── precision.py      # Model accuracy metrics
└── models/
    └── theta.json        # Saved parameters
```

## Algorithm

### Linear Regression

Find the best-fit line through data points to make predictions.

```
estimatePrice(mileage) = θ₀ + θ₁ × mileage
```

- **θ₀ (intercept)**: Base price when mileage = 0
- **θ₁ (slope)**: Price change per km (negative = price decreases)

### Gradient Descent

Iteratively adjust θ₀ and θ₁ to minimize prediction error.

**Update Rules:**
```
tmpθ₀ = learningRate × (1/m) × Σ(estimatePrice(mileage[i]) - price[i])
tmpθ₁ = learningRate × (1/m) × Σ((estimatePrice(mileage[i]) - price[i]) × mileage[i])

θ₀ = θ₀ - tmpθ₀
θ₁ = θ₁ - tmpθ₁
```

**How it works:**
1. Start with θ₀ = 0, θ₁ = 0
2. Calculate prediction error for all data points
3. Adjust θ₀ and θ₁ in the direction that reduces error
4. Repeat until error stops decreasing

### Feature Normalization

Scale data to improve convergence:
```
normalized = (x - mean) / std
```

**Why?** Original mileage values (22,899 ~ 240,000) cause unstable learning. Normalization centers data around 0.

### Key Implementation Details

- Simultaneous parameter updates (using tmp variables)
- Z-score normalization for stable training
- No prohibited libraries (numpy.polyfit, sklearn, etc.)

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.7330 (73% variance explained) |
| Prediction Accuracy | 90.35% |
| θ₀ (intercept) | 8499.60 |
| θ₁ (slope) | -0.0214 |

## Bonus Features

```bash
# Visualization: scatter plot + regression line
python3 bonus/visualize.py

# Precision metrics: R² score + MAPE
python3 bonus/precision.py
```

## Requirements

```bash
# Optional (for visualization only)
pip install matplotlib
```

---

42 School Project - ft_linear_regression
