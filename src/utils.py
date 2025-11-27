"""
Utility functions for linear regression implementation
"""
import json
import csv
import os


def load_data(filepath='data/data.csv'):
    """
    Load training data from CSV file

    Args:
        filepath: Path to CSV file with 'km' and 'price' columns

    Returns:
        tuple: (mileages, prices) as lists of floats
    """
    mileages = []
    prices = []

    try:
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['km'] and row['price']:  # Skip empty rows
                    mileages.append(float(row['km']))
                    prices.append(float(row['price']))
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    return mileages, prices


def normalize_data(data):
    """
    Normalize data using mean and standard deviation

    Args:
        data: List of numerical values

    Returns:
        tuple: (normalized_data, mean, std)
    """
    if not data:
        return [], 0, 1

    mean = sum(data) / len(data)

    # Calculate standard deviation manually
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5

    # Avoid division by zero
    if std == 0:
        std = 1

    normalized = [(x - mean) / std for x in data]

    return normalized, mean, std


def denormalize_theta(theta0_norm, theta1_norm, km_mean, km_std, price_mean, price_std):
    """
    Convert normalized theta values back to original scale

    Formula:
        price = price_mean + price_std * (theta0_norm + theta1_norm * (km - km_mean) / km_std)
        price = (price_mean + price_std * (theta0_norm - theta1_norm * km_mean / km_std))
                + (price_std * theta1_norm / km_std) * km

    Therefore:
        theta0 = price_mean + price_std * (theta0_norm - theta1_norm * km_mean / km_std)
        theta1 = price_std * theta1_norm / km_std

    Args:
        theta0_norm: Normalized intercept
        theta1_norm: Normalized slope
        km_mean: Mean of mileage data
        km_std: Standard deviation of mileage data
        price_mean: Mean of price data
        price_std: Standard deviation of price data

    Returns:
        tuple: (theta0, theta1) in original scale
    """
    theta1 = (price_std * theta1_norm) / km_std
    theta0 = price_mean + price_std * (theta0_norm - (theta1_norm * km_mean) / km_std)

    return theta0, theta1


def save_theta(theta0, theta1, filepath='models/theta.json',
               normalization_params=None):
    """
    Save trained parameters to JSON file

    Args:
        theta0: Intercept parameter
        theta1: Slope parameter
        filepath: Path to save file
        normalization_params: Optional dict with normalization parameters
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = {
        'theta0': theta0,
        'theta1': theta1
    }

    if normalization_params:
        data['normalization'] = normalization_params

    try:
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"✓ Parameters saved to {filepath}")
    except Exception as e:
        print(f"Error saving parameters: {e}")


def load_theta(filepath='models/theta.json'):
    """
    Load trained parameters from JSON file

    Args:
        filepath: Path to saved parameters file

    Returns:
        tuple: (theta0, theta1, normalization_params)
               Returns (0, 0, None) if file doesn't exist
    """
    if not os.path.exists(filepath):
        return 0, 0, None

    try:
        with open(filepath, 'r') as file:
            data = json.load(file)

        theta0 = data.get('theta0', 0)
        theta1 = data.get('theta1', 0)
        norm_params = data.get('normalization', None)

        return theta0, theta1, norm_params
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return 0, 0, None


def estimate_price(mileage, theta0, theta1):
    """
    Estimate price using linear hypothesis

    Args:
        mileage: Mileage value (km)
        theta0: Intercept parameter
        theta1: Slope parameter

    Returns:
        float: Estimated price
    """
    return theta0 + (theta1 * mileage)


def calculate_cost(mileages, prices, theta0, theta1):
    """
    Calculate Mean Squared Error cost function

    Cost = (1/2m) * Σ(estimate_price(mileage[i]) - price[i])^2

    Args:
        mileages: List of mileage values
        prices: List of actual prices
        theta0: Current intercept parameter
        theta1: Current slope parameter

    Returns:
        float: Cost value
    """
    m = len(prices)
    if m == 0:
        return 0

    total_error = 0
    for i in range(m):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = prediction - prices[i]
        total_error += error ** 2

    return total_error / (2 * m)
