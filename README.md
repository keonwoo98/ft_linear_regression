# ft_linear_regression

An introduction to machine learning - Simple linear regression implementation using gradient descent.

## 📋 Project Overview

This project implements a **linear regression algorithm** from scratch to predict car prices based on mileage. The implementation uses **gradient descent** optimization without relying on high-level ML libraries.

### Key Concepts

- **Linear Hypothesis**: `price = θ₀ + θ₁ × mileage`
- **Gradient Descent**: Iterative optimization algorithm
- **Feature Normalization**: Data scaling for better convergence
- **Cost Function**: Mean Squared Error (MSE)

## 🚀 Quick Start

### Installation

```bash
# Install dependencies (optional, only needed for bonus visualization)
pip install -r requirements.txt
```

### Training the Model

```bash
python3 src/train.py
```

**Output:**
- Trains the model on [data/data.csv](data/data.csv)
- Saves parameters to `models/theta.json`
- Displays training progress and example predictions

### Making Predictions

```bash
python3 src/predict.py
```

**Interactive mode:**
- Prompts for mileage input
- Returns estimated price
- Allows multiple predictions

## 📁 Project Structure

```
ft_linear_regression/
├── data/
│   └── data.csv              # Training dataset (24 samples)
├── models/
│   └── theta.json            # Saved model parameters
├── src/
│   ├── train.py              # Training program (gradient descent)
│   ├── predict.py            # Prediction program
│   └── utils.py              # Helper functions
├── bonus/                    # Bonus features (visualization, metrics)
├── requirements.txt          # Python dependencies
└── README.md
```

## 🧮 Algorithm Details

### Gradient Descent Implementation

**Update Rules** (from project PDF):

```
tmpθ₀ = learningRate × (1/m) × Σ(estimatePrice(mileage[i]) - price[i])
tmpθ₁ = learningRate × (1/m) × Σ((estimatePrice(mileage[i]) - price[i]) × mileage[i])
```

Where:
- **m**: Number of training examples
- **learningRate**: 0.1 (tunable)
- **Σ**: Sum over all training samples

**Key Features:**
- ✅ Simultaneous parameter updates (using temporary variables)
- ✅ Feature normalization (mean/std scaling)
- ✅ Cost function monitoring (MSE)
- ✅ No prohibited libraries (numpy.polyfit, sklearn, etc.)

### Current Model Performance

**Trained Parameters:**
- **θ₀** (intercept): 8499.60
- **θ₁** (slope): -0.0214

**Interpretation:**
- Base price: ~8,500 when mileage = 0
- Price decreases by ~0.021 per km
- Negative correlation between mileage and price ✓

**Example Predictions:**

| Mileage (km) | Estimated Price |
|--------------|-----------------|
| 50,000       | 7,427           |
| 100,000      | 6,355           |
| 150,000      | 5,282           |
| 200,000      | 4,210           |

## 📊 Dataset Information

**Source:** [data/data.csv](data/data.csv)

- **Samples:** 24
- **Features:** 1 (mileage in km)
- **Target:** Price
- **Mileage range:** 22,899 - 240,000 km
- **Price range:** 3,650 - 8,290

## 🎯 Implementation Checklist

### Mandatory Part ✅

- [x] **predict.py** - Price prediction program
  - [x] Prompts for mileage input
  - [x] Uses hypothesis: `estimatePrice = θ₀ + θ₁ × mileage`
  - [x] Handles untrained model (θ₀=0, θ₁=0)
  - [x] Input validation

- [x] **train.py** - Model training program
  - [x] Reads dataset from CSV
  - [x] Implements gradient descent from scratch
  - [x] Uses specified formulas from PDF
  - [x] Simultaneous parameter updates
  - [x] Saves θ₀ and θ₁ to file

- [x] **utils.py** - Helper functions
  - [x] Data loading and normalization
  - [x] Parameter persistence (save/load)
  - [x] Cost function calculation

### Bonus Part 🎁

- [x] Data visualization (scatter plot)
- [x] Regression line plotting
- [x] Precision calculation (R², MAE, RMSE)
- [x] Cost function convergence visualization
- [x] Residual analysis

## 🛠️ Technical Details

### Why Feature Normalization?

**Problem:** Original data has large ranges
- Mileage: 22,899 - 240,000 (variance ~10¹⁰)
- Price: 3,650 - 8,290 (variance ~10⁶)

**Solution:** Normalize using z-score
```python
normalized = (x - mean) / std
```

**Benefits:**
- ✅ Faster convergence
- ✅ Better numerical stability
- ✅ Learning rate easier to tune

### Hyperparameter Tuning

**Current Settings:**
- Learning Rate: 0.1
- Iterations: 1000

**How to adjust:**
Edit in [src/train.py](src/train.py):
```python
LEARNING_RATE = 0.1  # Increase for faster learning, decrease if diverging
ITERATIONS = 1000     # Increase if not converged
```

## 📖 Usage Examples

### Basic Usage

```bash
# 1. Train model
$ python3 src/train.py

# 2. Make prediction
$ python3 src/predict.py
Enter mileage (km): 100000

============================================================
  📊 ESTIMATION RESULT
============================================================
  Mileage: 100,000 km
  Estimated Price: 6,354.70
============================================================
```

### Before Training

```bash
$ python3 src/predict.py
⚠️  Warning: Model not trained yet!
   Using default parameters (θ₀=0, θ₁=0)
   Run train.py first for accurate predictions.
```

## 🔬 Validation

**Cost Function Convergence:**
- Initial cost: 0.430367
- Final cost: 0.133513
- Converged after ~100 iterations ✓

**Sanity Checks:**
- ✅ Negative slope (higher mileage → lower price)
- ✅ Reasonable price range (4,000 - 7,500)
- ✅ No division by zero errors
- ✅ No prohibited libraries used

## 📚 Learning Resources

**Concepts Covered:**
- Supervised learning fundamentals
- Linear regression theory
- Gradient descent optimization
- Feature scaling techniques
- Model evaluation metrics

**Formula Reference:**
- Hypothesis: `h(x) = θ₀ + θ₁x`
- Cost: `J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²`
- Update: `θⱼ := θⱼ - α × ∂J(θ)/∂θⱼ`

## ⚠️ Known Limitations

- **Extrapolation Risk:** Predictions outside training range may be unreliable
- **Single Feature:** Only considers mileage (ignores year, condition, etc.)
- **Linear Assumption:** Real-world relationships may be non-linear

## 🎓 Project Requirements

**From ft_linear_regression PDF:**
- ✅ Implement linear regression with gradient descent
- ✅ No numpy.polyfit or similar cheating libraries
- ✅ Use specified hypothesis function
- ✅ Use specified training formulas
- ✅ Simultaneous parameter updates

## 📝 Author

42 School Project - ft_linear_regression

---

## 🎁 Bonus Features

### 1. Data Visualization

```bash
python3 bonus/visualize.py
```

**Features:**
- Scatter plot of training data
- Regression line overlay
- Example predictions highlighted
- Residual plot analysis

### 2. Precision Metrics

```bash
python3 bonus/precision.py
```

**Calculated Metrics:**
- **R² Score**: 0.7330 (73.30% variance explained) ✓
- **MAE**: 557.84 (average error)
- **RMSE**: 667.57 (typical error)
- **MAPE**: 9.65% (percentage error)

**Model Assessment**: Good fit ✓

### 3. Training Visualization

```bash
python3 bonus/visualize_training.py
```

**Shows:**
- Cost function convergence over iterations
- Log-scale convergence plot
- Cost reduction per iteration
- Training statistics and performance

**Results:**
- Initial Cost: 0.4304
- Final Cost: 0.1335
- Reduction: 69.0% ✓
- Converged at: ~100 iterations

---

**Experiment Ideas:**
- Try different learning rates (0.01, 0.05, 0.2)
- Adjust iteration count
- Compare normalized vs. non-normalized training
