# ft_linear_regression Evaluation Script

**Duration:** 30-40 minutes
**Format:** Follow this script step by step

---

## Part 1: Introduction (3 min)

> Hello, I'll be presenting my **ft_linear_regression** project.
>
> The goal is simple: **predict car prices based on mileage**.
>
> When buying a used car, higher mileage means lower price, right?
> We're teaching a computer to learn this pattern.

### Show the data:
```bash
cat data/data.csv | head -10
```

> As you can see:
> - 240,000km → 3,650 (high mileage, low price)
> - 48,235km → 6,900 (low mileage, high price)
>
> Our program will learn this pattern and predict prices for any mileage.

---

## Part 2: Algorithm Explanation (5 min)

### 2-1. Linear Regression

> So, how do we make predictions?
>
> The simplest way is to **draw a line** that represents the pattern.

```
Price
  ↑
  |     •
  |         •
  |   •         •
  |               •
  |     •             •
  |                       •
  |___________________________→ Mileage
```

> The points roughly go downward to the right, correct?
> Finding the **best line** that represents these points is linear regression.

> The formula is: **`price = θ₀ + θ₁ × mileage`**
>
> - **θ₀** (theta zero): The y-intercept, base price when mileage is 0
> - **θ₁** (theta one): The slope, how much price changes per km
>
> Since price decreases as mileage increases, θ₁ will be **negative**.

### 2-2. Gradient Descent

> How do we find the best θ₀ and θ₁?
>
> We use an algorithm called **gradient descent**.

> Imagine you're blindfolded on a mountain, trying to reach the lowest valley.
> You feel around, find the steepest downward direction, and take a step.
> Repeat until you reach the bottom.
>
> That's gradient descent:
> 1. Start with θ₀ = 0, θ₁ = 0
> 2. Calculate how wrong our predictions are
> 3. Adjust θ in the direction that reduces error
> 4. Repeat 1000 times

### 2-3. Normalization

> There's one problem: mileage values are huge (22,899 ~ 240,000) while prices are smaller (3,650 ~ 8,290).
>
> This causes unstable learning.
>
> **Normalization** scales all values to a similar range:
> ```
> normalized = (value - mean) / standard_deviation
> ```
>
> After normalization, all values are roughly between -3 and +3.
> This makes learning much more stable.

---

## Part 3: Mandatory Demo (15 min)

### 3-1. Prediction BEFORE Training (3 min)

> First, let's see what happens before training.

```bash
rm -f models/theta.json
python3 src/predict.py
```

**Enter:** `100000`

> The result is **0**.
>
> Why? Because θ₀ = 0 and θ₁ = 0.
> The equation is: 0 + (0 × 100000) = 0
>
> We haven't trained yet, so the model can't predict anything.

**Press Ctrl+C to exit**

---

### 3-2. Training (5 min)

> Now let's train the model.

```bash
python3 src/train.py
```

> Let me explain what's happening:
>
> **[1/5] Loading data**
> - Loaded 24 training samples from CSV
>
> **[2/5] Normalizing data**
> - Scaling mileage and price to similar ranges
>
> **[3/5] Training model**
> - This is gradient descent running
> - Watch the **Cost** value - it shows how wrong our predictions are
> - Iteration 0: Cost = 0.49 (high error)
> - Iteration 100: Cost = 0.18 (error reduced!)
> - Iteration 999: Cost = 0.13 (converged!)
>
> **[4/5] Converting to original scale**
> - θ₀ = 8499.60 (base price when mileage is 0)
> - θ₁ = -0.0214 (price decreases by ~0.02 per km)
>
> **[5/5] Saving model**
> - Parameters saved to models/theta.json

---

### 3-3. Show the Code - Key Points (4 min)

> Let me show you the important parts of the code.

#### The Hypothesis Formula
```bash
# Show src/utils.py line 166
```

```python
def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)
```

> This is exactly the formula from the PDF: **θ₀ + θ₁ × mileage**

#### Simultaneous Update (IMPORTANT!)

```bash
# Show src/train.py lines 64-68
```

```python
# Calculate both gradients first
tmp_theta0 = theta0 - (learning_rate * gradient_theta0)
tmp_theta1 = theta1 - (learning_rate * gradient_theta1)

# Then update both at the same time
theta0 = tmp_theta0
theta1 = tmp_theta1
```

> Why use tmp variables?
>
> We must update θ₀ and θ₁ **simultaneously**.
>
> If we update θ₀ first, then θ₁ calculation would use the wrong (already changed) θ₀.
>
> The tmp variables ensure both use the **original** values.

---

### 3-4. Prediction AFTER Training (3 min)

> Now let's predict again.

```bash
python3 src/predict.py
```

**Enter multiple values:**

| Mileage | Predicted Price |
|---------|----------------|
| 50,000 | ~7,427 |
| 100,000 | ~6,355 |
| 200,000 | ~4,210 |

> Now it predicts real prices!
>
> The pattern makes sense:
> - Low mileage (50k) → Higher price (~7,400)
> - High mileage (200k) → Lower price (~4,200)
>
> Let's compare with CSV data:
> - CSV: 89,000km = 5,990
> - Our prediction for 89,000km ≈ 5,590
>
> Close but not exact - this is **normal and correct**.
> If predictions were exactly the same, it would be **overfitting**.

---

## Part 4: Bonus Demo (10 min)

### 4-1. Data Visualization (3 min)

```bash
python3 bonus/visualize.py
```

> This shows two things:
> 1. **Blue dots**: The 24 training data points
> 2. **Red line**: Our regression line
>
> You can see the line passes through the middle of the dots,
> capturing the overall trend: as mileage increases, price decreases.

---

### 4-2. Precision Calculator (3 min)

```bash
python3 bonus/precision.py
```

> **R² Score = 0.7330**
>
> What does this mean?
> - R² ranges from 0 to 1
> - 0.73 means **73% of price variation** is explained by mileage
> - The remaining 27% is due to other factors (brand, year, condition, etc.)
>
> **Prediction Accuracy = 90.35%**
>
> On average, our predictions are within 10% of actual prices.
> For a simple model with just one feature, this is good!

---

### 4-3. Cost History Visualization (2 min)

```bash
python3 bonus/visualize_cost.py
```

> This shows how the **cost (error) decreases** during training.
>
> - Starts high (model knows nothing)
> - Drops quickly in first ~100 iterations
> - Then flattens (converged - no more improvement)
>
> This is proof that gradient descent is working!

---

### 4-4. Overfitting Explanation (2 min)

> **What is overfitting?**
>
> It's when the model memorizes training data instead of learning patterns.
> Like a student who memorizes answers without understanding.
>
> **Signs of overfitting:**
> - R² = 1.0 (100% - too perfect)
> - Predictions exactly match all CSV values
>
> **Our model is NOT overfitting because:**
> - R² = 0.73 (not 1.0)
> - Predictions are close but not exact
>
> For example:
> - CSV: 97,500km → 6,800
> - Our prediction: ~6,408
> - Difference exists → **NOT overfitting** ✓

---

## Part 5: Q&A Preparation

### Q: Why can't we use numpy.polyfit?

> The project's purpose is to **understand** and **implement** the algorithm ourselves.
> Using numpy.polyfit would be one line of code - too easy!
> It's like a math test where calculators aren't allowed.

### Q: How did you choose the learning rate?

> Through experimentation:
> - 0.001: Too slow (doesn't converge in 1000 iterations)
> - 0.01: Works well (converges around iteration 100)
> - 0.1: Can become unstable
> - 1.0: Diverges (gets worse instead of better)

### Q: What happens without normalization?

> Without normalization, learning becomes very unstable.
> - Mileage values (~100,000) create huge gradients
> - Price values (~6,000) create small gradients
> - The learning rate can't work for both at the same time
>
> With normalization, both are on similar scales → stable learning.

### Q: Is 24 data points enough?

> For a simple linear relationship, yes.
> Our R² of 0.73 shows the model learned the pattern.
>
> For a real-world application, we'd want:
> - More data points (100+)
> - More features (brand, year, condition)
> - More complex models

---

## Evaluation Checklist Summary

### Mandatory ✅
- [x] 2 programs: `predict.py` and `train.py`
- [x] Prediction returns 0 before training
- [x] Equation: `theta0 + (theta1 * mileage)`
- [x] Reads CSV for training
- [x] Simultaneous update with tmp variables
- [x] Saves theta0 and theta1
- [x] Prediction returns real price after training

### Bonus (5/5) ✅
1. [x] Data scatter plot (`visualize.py`)
2. [x] Regression line in same graph (`visualize.py`)
3. [x] Precision calculator (`precision.py`)
4. [x] Cost history graph (`visualize_cost.py`)
5. [x] Overfitting explanation (verbal)

---

## Quick Commands Reference

```bash
# Reset and test before training
rm -f models/theta.json && python3 src/predict.py

# Train
python3 src/train.py

# Predict after training
python3 src/predict.py

# Bonus
python3 bonus/visualize.py
python3 bonus/precision.py
python3 bonus/visualize_cost.py
```

---

**Thank you for your evaluation!**
