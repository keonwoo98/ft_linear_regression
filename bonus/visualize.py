#!/usr/bin/env python3
"""
Visualization tool for linear regression
Plots training data and regression line
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required for visualization")
    print("Install it with: pip install matplotlib")
    sys.exit(1)

from utils import load_data, load_theta, estimate_price


def plot_data_and_regression():
    """
    Plot training data as scatter plot and regression line
    """
    print("\n" + "="*60)
    print("  LINEAR REGRESSION VISUALIZATION")
    print("="*60 + "\n")

    # Load training data
    print("[1/3] Loading data...")
    mileages, prices = load_data('data/data.csv')

    if mileages is None or prices is None:
        print("❌ Failed to load data")
        return

    print(f"✓ Loaded {len(mileages)} data points")

    # Load trained parameters
    print("\n[2/3] Loading model parameters...")
    theta0, theta1, norm_params = load_theta('models/theta.json')

    if theta0 == 0 and theta1 == 0:
        print("⚠️  Warning: Model not trained!")
        print("   Showing data only, no regression line")
        show_regression = False
    else:
        print(f"✓ Model loaded")
        print(f"  θ₀ = {theta0:.4f}")
        print(f"  θ₁ = {theta1:.8f}")
        show_regression = True

    # Create figure with subplots
    print("\n[3/3] Generating visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Linear Regression: Car Price Prediction', fontsize=16, fontweight='bold')

    # Plot 1: Scatter plot with regression line
    ax1.scatter(mileages, prices, color='blue', alpha=0.6, s=80,
                edgecolors='darkblue', linewidth=1.5, label='Training Data')
    ax1.set_xlabel('Mileage (km)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title('Data Distribution & Regression Line', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')

    if show_regression:
        # Generate points for regression line
        min_km = min(mileages)
        max_km = max(mileages)
        margin = (max_km - min_km) * 0.1

        x_line = [min_km - margin, max_km + margin]
        y_line = [estimate_price(x, theta0, theta1) for x in x_line]

        ax1.plot(x_line, y_line, color='red', linewidth=2.5,
                label=f'Regression Line\ny = {theta0:.2f} + {theta1:.6f}x',
                linestyle='-', alpha=0.8)

        # Add prediction examples
        example_km = [50000, 100000, 150000]
        example_prices = [estimate_price(km, theta0, theta1) for km in example_km]
        ax1.scatter(example_km, example_prices, color='green', s=120,
                   marker='*', edgecolors='darkgreen', linewidth=1.5,
                   label='Example Predictions', zorder=5)

    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # Plot 2: Residual plot (if model is trained)
    if show_regression:
        predictions = [estimate_price(km, theta0, theta1) for km in mileages]
        residuals = [actual - pred for actual, pred in zip(prices, predictions)]

        ax2.scatter(mileages, residuals, color='purple', alpha=0.6, s=80,
                   edgecolors='darkviolet', linewidth=1.5)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Fit')
        ax2.set_xlabel('Mileage (km)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Plot', fontsize=13)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=10)

        # Add statistics
        mean_residual = sum(residuals) / len(residuals)
        residual_std = (sum((r - mean_residual)**2 for r in residuals) / len(residuals)) ** 0.5

        textstr = f'Mean Residual: {mean_residual:.2f}\nStd Dev: {residual_std:.2f}'
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Show data distribution histogram
        ax2.hist(prices, bins=15, color='skyblue', edgecolor='darkblue',
                alpha=0.7, linewidth=1.5)
        ax2.set_xlabel('Price', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Price Distribution', fontsize=13)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    print("✓ Visualization generated")
    print("\n" + "="*60)
    print("  Close the plot window to exit")
    print("="*60 + "\n")

    plt.show()


def plot_training_progress():
    """
    Plot cost function convergence during training
    Note: This requires saving cost history during training
    """
    print("\n📊 Training progress visualization")
    print("   (Requires cost history - feature not yet implemented)")
    print("   Tip: Modify train.py to save cost_history to file\n")


if __name__ == "__main__":
    plot_data_and_regression()
