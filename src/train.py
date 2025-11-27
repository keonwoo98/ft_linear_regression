#!/usr/bin/env python3
"""
Training program for linear regression model
Implements gradient descent algorithm from scratch
"""
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_data,
    normalize_data,
    denormalize_theta,
    save_theta,
    estimate_price,
    calculate_cost
)
import json


def gradient_descent(mileages, prices, learning_rate=0.01, iterations=1000,
                     verbose=True):
    """
    Perform gradient descent to learn theta0 and theta1

    Args:
        mileages: List of mileage values (normalized)
        prices: List of price values (normalized)
        learning_rate: Learning rate (alpha)
        iterations: Maximum number of iterations
        verbose: Print progress information

    Returns:
        tuple: (theta0, theta1, cost_history)
    """
    m = len(prices)
    theta0 = 0.0
    theta1 = 0.0
    cost_history = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 Starting Gradient Descent")
        print(f"{'='*60}")
        print(f"Training samples: {m}")
        print(f"Learning rate: {learning_rate}")
        print(f"Max iterations: {iterations}")
        print(f"{'='*60}\n")

    for i in range(iterations):
        # Calculate predictions and errors for all samples
        predictions = [estimate_price(km, theta0, theta1) for km in mileages]
        errors = [pred - actual for pred, actual in zip(predictions, prices)]

        # Calculate gradients (from PDF formulas)
        # tmpθ₀ = learningRate × (1/m) × Σ(errors)
        # tmpθ₁ = learningRate × (1/m) × Σ(errors × mileage)

        gradient_theta0 = sum(errors) / m
        gradient_theta1 = sum(err * km for err, km in zip(errors, mileages)) / m

        # Update parameters SIMULTANEOUSLY
        tmp_theta0 = theta0 - (learning_rate * gradient_theta0)
        tmp_theta1 = theta1 - (learning_rate * gradient_theta1)

        theta0 = tmp_theta0
        theta1 = tmp_theta1

        # Calculate and store cost
        cost = calculate_cost(mileages, prices, theta0, theta1)
        cost_history.append(cost)

        # Print progress
        if verbose and (i % 100 == 0 or i == iterations - 1):
            print(f"Iteration {i:4d} | Cost: {cost:.6f} | "
                  f"θ₀: {theta0:8.4f} | θ₁: {theta1:8.4f}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Training completed!")
        print(f"{'='*60}\n")

    return theta0, theta1, cost_history


def main():
    """Main training function"""

    print("\n" + "="*60)
    print("  LINEAR REGRESSION TRAINING")
    print("="*60)

    # Step 1: Load data
    print("\n[1/5] Loading data...")
    mileages, prices = load_data('data/data.csv')

    if mileages is None or prices is None:
        print("❌ Failed to load data")
        return

    print(f"✓ Loaded {len(mileages)} training examples")
    print(f"  Mileage range: {min(mileages):.0f} - {max(mileages):.0f} km")
    print(f"  Price range: {min(prices):.0f} - {max(prices):.0f}")

    # Step 2: Normalize data
    print("\n[2/5] Normalizing data...")
    norm_mileages, km_mean, km_std = normalize_data(mileages)
    norm_prices, price_mean, price_std = normalize_data(prices)

    print(f"✓ Data normalized")
    print(f"  Mileage: μ={km_mean:.2f}, σ={km_std:.2f}")
    print(f"  Price: μ={price_mean:.2f}, σ={price_std:.2f}")

    # Step 3: Train model
    print("\n[3/5] Training model...")

    # You can adjust these hyperparameters
    LEARNING_RATE = 0.1
    ITERATIONS = 1000

    theta0_norm, theta1_norm, cost_history = gradient_descent(
        norm_mileages,
        norm_prices,
        learning_rate=LEARNING_RATE,
        iterations=ITERATIONS,
        verbose=True
    )

    # Step 4: Denormalize parameters
    print("\n[4/5] Converting to original scale...")
    theta0, theta1 = denormalize_theta(
        theta0_norm, theta1_norm,
        km_mean, km_std,
        price_mean, price_std
    )

    print(f"✓ Final parameters:")
    print(f"  θ₀ (intercept): {theta0:.4f}")
    print(f"  θ₁ (slope): {theta1:.8f}")

    # Step 5: Save parameters
    print("\n[5/5] Saving model...")
    normalization_params = {
        'km_mean': km_mean,
        'km_std': km_std,
        'price_mean': price_mean,
        'price_std': price_std
    }

    save_theta(theta0, theta1, normalization_params=normalization_params)

    # Save cost history for visualization
    os.makedirs('models', exist_ok=True)
    with open('models/cost_history.json', 'w') as f:
        json.dump({'cost_history': cost_history}, f, indent=2)
    print("✓ Cost history saved for visualization")

    # Show some example predictions
    print("\n" + "="*60)
    print("  EXAMPLE PREDICTIONS")
    print("="*60)

    test_mileages = [50000, 100000, 150000, 200000]
    for km in test_mileages:
        predicted_price = estimate_price(km, theta0, theta1)
        print(f"  {km:6d} km → {predicted_price:7.2f} €")

    print("\n" + "="*60)
    print("✓ Training complete! Use predict.py to make predictions.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
