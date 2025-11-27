#!/usr/bin/env python3
"""
Precision calculator for linear regression model
Calculates R², MAE, RMSE, and other metrics
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from utils import load_data, load_theta, estimate_price


def calculate_r_squared(actual, predicted):
    """
    Calculate R² (coefficient of determination)

    R² = 1 - (SS_res / SS_tot)
    where:
        SS_res = Σ(actual - predicted)²  (residual sum of squares)
        SS_tot = Σ(actual - mean)²       (total sum of squares)

    R² ranges from 0 to 1:
        1.0 = perfect prediction
        0.0 = no better than predicting the mean
        <0  = worse than mean

    Args:
        actual: List of actual values
        predicted: List of predicted values

    Returns:
        float: R² score
    """
    n = len(actual)
    if n == 0:
        return 0

    # Calculate mean of actual values
    mean_actual = sum(actual) / n

    # SS_tot: Total sum of squares
    ss_tot = sum((y - mean_actual) ** 2 for y in actual)

    # SS_res: Residual sum of squares
    ss_res = sum((y_actual - y_pred) ** 2
                 for y_actual, y_pred in zip(actual, predicted))

    # Avoid division by zero
    if ss_tot == 0:
        return 0

    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error (MAE)

    MAE = (1/n) × Σ|actual - predicted|

    Args:
        actual: List of actual values
        predicted: List of predicted values

    Returns:
        float: MAE value
    """
    n = len(actual)
    if n == 0:
        return 0

    mae = sum(abs(y_actual - y_pred)
              for y_actual, y_pred in zip(actual, predicted)) / n

    return mae


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error (RMSE)

    RMSE = √[(1/n) × Σ(actual - predicted)²]

    Args:
        actual: List of actual values
        predicted: List of predicted values

    Returns:
        float: RMSE value
    """
    n = len(actual)
    if n == 0:
        return 0

    mse = sum((y_actual - y_pred) ** 2
              for y_actual, y_pred in zip(actual, predicted)) / n

    rmse = mse ** 0.5

    return rmse


def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    MAPE = (100/n) × Σ|actual - predicted| / |actual|

    Args:
        actual: List of actual values
        predicted: List of predicted values

    Returns:
        float: MAPE value (percentage)
    """
    n = len(actual)
    if n == 0:
        return 0

    # Skip zero values to avoid division by zero
    errors = []
    for y_actual, y_pred in zip(actual, predicted):
        if y_actual != 0:
            errors.append(abs((y_actual - y_pred) / y_actual))

    if not errors:
        return 0

    mape = (sum(errors) / len(errors)) * 100

    return mape


def interpret_r_squared(r2):
    """
    Provide interpretation of R² score

    Args:
        r2: R² score

    Returns:
        str: Interpretation message
    """
    if r2 >= 0.9:
        return "Excellent fit 🌟"
    elif r2 >= 0.8:
        return "Very good fit ✓"
    elif r2 >= 0.7:
        return "Good fit ✓"
    elif r2 >= 0.5:
        return "Moderate fit ~"
    elif r2 >= 0.3:
        return "Weak fit ⚠"
    else:
        return "Poor fit ❌"


def analyze_residuals(actual, predicted):
    """
    Analyze residuals for patterns

    Args:
        actual: List of actual values
        predicted: List of predicted values

    Returns:
        dict: Residual statistics
    """
    residuals = [y_actual - y_pred for y_actual, y_pred in zip(actual, predicted)]
    n = len(residuals)

    if n == 0:
        return {}

    mean_residual = sum(residuals) / n
    std_residual = (sum((r - mean_residual) ** 2 for r in residuals) / n) ** 0.5
    min_residual = min(residuals)
    max_residual = max(residuals)

    return {
        'mean': mean_residual,
        'std': std_residual,
        'min': min_residual,
        'max': max_residual,
        'range': max_residual - min_residual
    }


def main():
    """Main precision calculation function"""

    print("\n" + "="*70)
    print("  MODEL PRECISION ANALYSIS")
    print("="*70 + "\n")

    # Load data
    print("[1/3] Loading data...")
    mileages, prices = load_data('data/data.csv')

    if mileages is None or prices is None:
        print("❌ Failed to load data")
        return

    print(f"✓ Loaded {len(mileages)} data points")

    # Load model
    print("\n[2/3] Loading model...")
    theta0, theta1, norm_params = load_theta('models/theta.json')

    if theta0 == 0 and theta1 == 0:
        print("❌ Error: Model not trained!")
        print("   Run train.py first")
        return

    print(f"✓ Model loaded")
    print(f"  θ₀ = {theta0:.4f}")
    print(f"  θ₁ = {theta1:.8f}")

    # Calculate predictions
    print("\n[3/3] Calculating metrics...")
    predictions = [estimate_price(km, theta0, theta1) for km in mileages]

    # Calculate metrics
    r2 = calculate_r_squared(prices, predictions)
    mae = calculate_mae(prices, predictions)
    rmse = calculate_rmse(prices, predictions)
    mape = calculate_mape(prices, predictions)

    # Analyze residuals
    residual_stats = analyze_residuals(prices, predictions)

    # Display results
    print("\n" + "="*70)
    print("  📊 PERFORMANCE METRICS")
    print("="*70)

    print("\n1. Coefficient of Determination (R²)")
    print("-" * 70)
    print(f"   R² Score: {r2:.6f}")
    print(f"   Interpretation: {interpret_r_squared(r2)}")
    print(f"   Explanation: {r2*100:.2f}% of variance is explained by the model")

    print("\n2. Mean Absolute Error (MAE)")
    print("-" * 70)
    print(f"   MAE: {mae:.2f}")
    print(f"   Interpretation: Average error is ±{mae:.2f} units")

    print("\n3. Root Mean Squared Error (RMSE)")
    print("-" * 70)
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Interpretation: Typical error is around {rmse:.2f} units")
    print(f"   Note: RMSE penalizes larger errors more than MAE")

    print("\n4. Mean Absolute Percentage Error (MAPE)")
    print("-" * 70)
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Interpretation: Average error is {mape:.2f}% of actual value")

    print("\n5. Residual Analysis")
    print("-" * 70)
    print(f"   Mean Residual: {residual_stats['mean']:.2f}")
    print(f"   Std Deviation: {residual_stats['std']:.2f}")
    print(f"   Min Residual: {residual_stats['min']:.2f}")
    print(f"   Max Residual: {residual_stats['max']:.2f}")
    print(f"   Range: {residual_stats['range']:.2f}")

    if abs(residual_stats['mean']) < 0.1:
        print("   ✓ Residuals are well-centered around zero")
    else:
        print("   ⚠ Residuals show bias (mean not close to zero)")

    # Overall assessment
    print("\n" + "="*70)
    print("  🎯 OVERALL ASSESSMENT")
    print("="*70)

    if r2 >= 0.8 and mape <= 15:
        print("\n   ✓ Model performance is EXCELLENT")
        print("   The model provides reliable predictions")
    elif r2 >= 0.6 and mape <= 25:
        print("\n   ✓ Model performance is GOOD")
        print("   The model is suitable for general predictions")
    elif r2 >= 0.4:
        print("\n   ~ Model performance is MODERATE")
        print("   Predictions may be unreliable in some cases")
    else:
        print("\n   ❌ Model performance is POOR")
        print("   Consider using more features or a different model")

    # Detailed predictions table
    print("\n" + "="*70)
    print("  📋 DETAILED PREDICTIONS (First 10 samples)")
    print("="*70)
    print(f"\n{'Mileage':>10} | {'Actual':>8} | {'Predicted':>10} | {'Error':>8} | {'% Error':>8}")
    print("-" * 70)

    for i in range(min(10, len(mileages))):
        km = mileages[i]
        actual = prices[i]
        pred = predictions[i]
        error = actual - pred
        pct_error = (error / actual * 100) if actual != 0 else 0

        print(f"{km:10.0f} | {actual:8.2f} | {pred:10.2f} | "
              f"{error:8.2f} | {pct_error:7.2f}%")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
