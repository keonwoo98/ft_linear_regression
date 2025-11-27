#!/usr/bin/env python3
"""
Bonus: Precision calculator for linear regression model
Calculates how accurate the model predictions are
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from utils import load_data, load_theta, estimate_price


def calculate_r_squared(actual, predicted):
    """
    Calculate R² (coefficient of determination)
    R² = 1 - (SS_res / SS_tot)

    Range: 0 to 1 (1 = perfect prediction)
    """
    n = len(actual)
    mean_actual = sum(actual) / n

    ss_tot = sum((y - mean_actual) ** 2 for y in actual)
    ss_res = sum((y - pred) ** 2 for y, pred in zip(actual, predicted))

    if ss_tot == 0:
        return 0

    return 1 - (ss_res / ss_tot)


def calculate_mape(actual, predicted):
    """
    Calculate MAPE (Mean Absolute Percentage Error)
    MAPE = (1/n) × Σ|actual - predicted| / actual × 100

    Returns: error rate in percentage
    """
    n = len(actual)
    total_error = sum(abs(a - p) / a for a, p in zip(actual, predicted))
    return (total_error / n) * 100


def main():
    # Load data
    mileages, prices = load_data('data/data.csv')
    if mileages is None:
        print("Error: Cannot load data")
        return

    # Load model
    theta0, theta1, _ = load_theta('models/theta.json')
    if theta0 == 0 and theta1 == 0:
        print("Error: Model not trained. Run train.py first.")
        return

    # Calculate predictions
    predictions = [estimate_price(km, theta0, theta1) for km in mileages]

    # Calculate metrics
    r2 = calculate_r_squared(prices, predictions)
    mape = calculate_mape(prices, predictions)
    accuracy = 100 - mape

    # Display result
    print(f"\n{'='*50}")
    print(f"  MODEL PRECISION")
    print(f"{'='*50}")

    # R² Score (variance explanation)
    print(f"\n  [1] R² Score (Variance Explained)")
    print(f"  ─────────────────────────────────")
    print(f"  R² = {r2:.4f}")
    print(f"  → {r2*100:.2f}% of price variance explained by mileage")
    if r2 >= 0.7:
        print(f"  → Good fit ✓")
    elif r2 >= 0.5:
        print(f"  → Moderate fit")
    else:
        print(f"  → Poor fit")

    # MAPE (prediction accuracy)
    print(f"\n  [2] Prediction Accuracy (MAPE)")
    print(f"  ─────────────────────────────────")
    print(f"  Average Error Rate = {mape:.2f}%")
    print(f"  Average Accuracy   = {accuracy:.2f}%")
    if accuracy >= 90:
        print(f"  → Excellent ✓")
    elif accuracy >= 80:
        print(f"  → Good ✓")
    else:
        print(f"  → Needs improvement")

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()
