#!/usr/bin/env python3
"""
Bonus: Data visualization with regression line
- Plotting the data into a graph to see their repartition
- Plotting the regression line into the same graph
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

from utils import load_data, load_theta, estimate_price


def main():
    # Load data
    mileages, prices = load_data('data/data.csv')
    if mileages is None:
        print("Error: Cannot load data")
        return

    # Load model
    theta0, theta1, _ = load_theta('models/theta.json')

    # Create plot
    plt.figure(figsize=(10, 6))

    # 1. Scatter plot (data repartition)
    plt.scatter(mileages, prices, color='blue', label='Training Data')

    # 2. Regression line (if trained)
    if theta0 != 0 or theta1 != 0:
        x_line = [min(mileages), max(mileages)]
        y_line = [estimate_price(x, theta0, theta1) for x in x_line]
        plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')

    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.title('Linear Regression: Car Price vs Mileage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
