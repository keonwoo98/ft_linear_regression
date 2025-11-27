#!/usr/bin/env python3
"""
Prediction program for linear regression model
Estimates car price based on mileage
"""
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_theta, estimate_price


def get_mileage_input():
    """
    Get mileage input from user with validation

    Returns:
        float: Valid mileage value or None if invalid
    """
    try:
        mileage_str = input("Enter mileage (km): ").strip()

        # Handle empty input
        if not mileage_str:
            print("❌ Error: Please enter a value")
            return None

        mileage = float(mileage_str)

        # Validate non-negative
        if mileage < 0:
            print("❌ Error: Mileage cannot be negative")
            return None

        return mileage

    except ValueError:
        print("❌ Error: Please enter a valid number")
        return None
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


def main():
    """Main prediction function"""

    print("\n" + "="*60)
    print("  CAR PRICE ESTIMATOR")
    print("="*60 + "\n")

    # Load trained parameters
    theta0, theta1, norm_params = load_theta('models/theta.json')

    # Check if model has been trained
    if theta0 == 0 and theta1 == 0:
        print("⚠️  Warning: Model not trained yet!")
        print("   Using default parameters (θ₀=0, θ₁=0)")
        print("   Run train.py first for accurate predictions.\n")
    else:
        print(f"✓ Model loaded successfully")
        print(f"  θ₀ = {theta0:.4f}")
        print(f"  θ₁ = {theta1:.8f}\n")

    print("-" * 60)

    # Interactive prediction loop
    while True:
        mileage = get_mileage_input()

        if mileage is None:
            continue

        # Calculate estimated price
        estimated_price = estimate_price(mileage, theta0, theta1)

        # Display result
        print("\n" + "="*60)
        print(f"  📊 ESTIMATION RESULT")
        print("="*60)
        print(f"  Mileage: {mileage:,.0f} km")
        print(f"  Estimated Price: {estimated_price:,.2f}")

        # Warning for unrealistic predictions
        if estimated_price < 0:
            print("\n  ⚠️  Warning: Negative price predicted!")
            print("     The model may not be reliable for this mileage range.")
        elif estimated_price > 20000:
            print("\n  ⚠️  Warning: Very high price predicted!")
            print("     This mileage may be outside the training data range.")

        print("="*60 + "\n")
        print("-" * 60)

        # Ask if user wants to continue
        try:
            continue_input = input("\nMake another prediction? (y/n): ").strip().lower()
            if continue_input not in ['y', 'yes', '']:
                print("\nGoodbye! 👋\n")
                break
            print("\n" + "-" * 60)
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break


if __name__ == "__main__":
    main()
