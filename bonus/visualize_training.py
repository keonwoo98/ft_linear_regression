#!/usr/bin/env python3
"""
Training visualization tool
Shows cost function convergence during gradient descent
"""
import sys
import os
import json

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required for visualization")
    print("Install it with: pip install matplotlib")
    sys.exit(1)


def load_cost_history(filepath='models/cost_history.json'):
    """
    Load cost history from training

    Args:
        filepath: Path to cost history file

    Returns:
        list: Cost values for each iteration
    """
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get('cost_history', [])
    except Exception as e:
        print(f"Error loading cost history: {e}")
        return None


def plot_cost_convergence():
    """
    Plot cost function convergence over iterations
    """
    print("\n" + "="*60)
    print("  TRAINING CONVERGENCE VISUALIZATION")
    print("="*60 + "\n")

    # Load cost history
    print("[1/2] Loading training history...")
    cost_history = load_cost_history()

    if cost_history is None:
        print("❌ No training history found")
        print("   Run train.py to generate training data")
        return

    if len(cost_history) == 0:
        print("❌ Cost history is empty")
        return

    print(f"✓ Loaded {len(cost_history)} iterations")

    # Create visualization
    print("\n[2/2] Generating plots...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Descent Training Analysis', fontsize=16, fontweight='bold')

    iterations = list(range(len(cost_history)))

    # Plot 1: Full cost history
    ax1.plot(iterations, cost_history, color='blue', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cost (MSE)', fontsize=11, fontweight='bold')
    ax1.set_title('Cost Function Convergence', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add initial and final cost annotations
    initial_cost = cost_history[0]
    final_cost = cost_history[-1]
    ax1.annotate(f'Initial: {initial_cost:.6f}',
                xy=(0, initial_cost), xytext=(len(iterations)*0.3, initial_cost),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax1.annotate(f'Final: {final_cost:.6f}',
                xy=(len(iterations)-1, final_cost),
                xytext=(len(iterations)*0.5, final_cost + (initial_cost-final_cost)*0.2),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Plot 2: Log scale cost (better for seeing convergence)
    ax2.semilogy(iterations, cost_history, color='red', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cost (MSE) - Log Scale', fontsize=11, fontweight='bold')
    ax2.set_title('Cost Convergence (Log Scale)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Plot 3: Cost improvement per iteration
    if len(cost_history) > 1:
        cost_diff = [cost_history[i] - cost_history[i+1]
                     for i in range(len(cost_history)-1)]
        ax3.plot(range(len(cost_diff)), cost_diff, color='green',
                linewidth=1.5, alpha=0.7)
        ax3.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Cost Improvement', fontsize=11, fontweight='bold')
        ax3.set_title('Cost Reduction per Iteration', fontsize=12)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 4: Training statistics
    ax4.axis('off')

    # Calculate statistics
    total_reduction = initial_cost - final_cost
    pct_reduction = (total_reduction / initial_cost * 100) if initial_cost != 0 else 0

    # Find convergence point (when improvement drops below threshold)
    convergence_iter = len(cost_history)
    threshold = 0.0001
    for i in range(1, len(cost_history)):
        if abs(cost_history[i] - cost_history[i-1]) < threshold:
            convergence_iter = i
            break

    stats_text = f"""
    TRAINING STATISTICS
    {'='*40}

    Initial Cost:        {initial_cost:.6f}
    Final Cost:          {final_cost:.6f}
    Total Reduction:     {total_reduction:.6f}
    Percentage Drop:     {pct_reduction:.2f}%

    Total Iterations:    {len(cost_history)}
    Convergence at:      ~{convergence_iter} iterations
    Convergence Threshold: {threshold}

    Model Status:        {'✓ Converged' if convergence_iter < len(cost_history) else '~ Converging'}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Add performance indicator
    if pct_reduction > 50:
        performance = "Excellent ✓"
        color = 'green'
    elif pct_reduction > 30:
        performance = "Good ✓"
        color = 'blue'
    elif pct_reduction > 10:
        performance = "Moderate ~"
        color = 'orange'
    else:
        performance = "Poor ❌"
        color = 'red'

    ax4.text(0.5, 0.15, f"Performance: {performance}",
            transform=ax4.transAxes, fontsize=13, fontweight='bold',
            horizontalalignment='center', color=color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))

    plt.tight_layout()

    print("✓ Visualization generated")
    print("\n" + "="*60)
    print("  Close the plot window to exit")
    print("="*60 + "\n")

    plt.show()


if __name__ == "__main__":
    plot_cost_convergence()
