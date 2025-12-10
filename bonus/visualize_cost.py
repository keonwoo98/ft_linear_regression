#!/usr/bin/env python3
"""
Bonus: Training process visualization
Shows how cost decreases during gradient descent
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

from utils import load_data, normalize_data, estimate_price, calculate_cost


def gradient_descent_with_history(mileages, prices, learning_rate=0.01, iterations=1000):
    """Run gradient descent and return cost history"""
    m = len(prices)
    theta0 = 0.0
    theta1 = 0.0
    cost_history = []

    for _ in range(iterations):
        predictions = [estimate_price(km, theta0, theta1) for km in mileages]
        errors = [pred - actual for pred, actual in zip(predictions, prices)]

        gradient_theta0 = sum(errors) / m
        gradient_theta1 = sum(err * km for err, km in zip(errors, mileages)) / m

        tmp_theta0 = theta0 - (learning_rate * gradient_theta0)
        tmp_theta1 = theta1 - (learning_rate * gradient_theta1)

        theta0 = tmp_theta0
        theta1 = tmp_theta1

        cost = calculate_cost(mileages, prices, theta0, theta1)
        cost_history.append(cost)

    return cost_history


def main():
    # Load and normalize data
    mileages, prices = load_data('data/data.csv')
    if mileages is None:
        print("Error: Cannot load data")
        return

    norm_mileages, _, _ = normalize_data(mileages)
    norm_prices, _, _ = normalize_data(prices)

    # Train and get cost history
    print("Training model...")
    cost_history = gradient_descent_with_history(norm_mileages, norm_prices)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='blue', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Gradient Descent: Cost vs Iterations')
    plt.grid(True, alpha=0.3)

    # Mark convergence point
    for i in range(1, len(cost_history)):
        if abs(cost_history[i] - cost_history[i-1]) < 0.0001:
            plt.axvline(x=i, color='red', linestyle='--', label=f'Converged at iteration {i}')
            break

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
