import numpy as np
import json
import argparse
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

def load_data_from_csv(filename):
    """Load and preprocess data from a CSV file"""
    try:
        # Load the data - skip the header row
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        
        # Print data info
        print(f"Loaded {len(data)} records from {filename}")
        print(f"Data shape: {data.shape}")
        
        return data
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        sys.exit(1)

class PolynomialPredictor:
    """Class for making predictions using polynomial fits of any degree"""
    
    def __init__(self, window_size=10, degree=2):
        self.window_size = window_size
        self.degree = degree  # Polynomial degree (2=quadratic, 3=cubic, etc.)
        self.model_params = None
    
    def fit(self, X, y):
        """Fit polynomial models to each feature"""
        n_features = y.shape[1] if len(y.shape) > 1 else 1
        self.model_params = []
        
        # For each feature, fit a separate polynomial model
        for i in range(n_features):
            y_feature = y[:, i] if n_features > 1 else y
            # X values are just indices for the time steps
            x_indices = np.arange(len(y_feature))
            # Fit polynomial: y = a*x^n + ... + b*x + c
            coeffs = np.polyfit(x_indices, y_feature, self.degree)
            self.model_params.append(coeffs)
        
        return self
    
    def predict(self, X=None):
        """Make predictions using the fitted polynomial models"""
        if self.model_params is None:
            raise ValueError("Model not fitted yet")
        
        # X is ignored since we're using indices as X values
        n_features = len(self.model_params)
        # Predict one step beyond the last point
        next_index = len(X) if X is not None else 0
        
        predictions = np.zeros(n_features)
        for i in range(n_features):
            # Generate prediction from the polynomial equation
            coeffs = self.model_params[i]
            predictions[i] = np.polyval(coeffs, next_index)
        
        return predictions
    
    def to_json(self):
        """Convert model parameters to a JSON-serializable dictionary"""
        if self.model_params is None:
            raise ValueError("Model not fitted yet")
        
        model_dict = {
            "algorithm": f"polynomial_fit_degree_{self.degree}",
            "window_size": self.window_size,
            "degree": self.degree,
            "coefficients": [coeff.tolist() for coeff in self.model_params],
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "created_by": os.environ.get("USER", "unknown")
            }
        }
        return model_dict

def generate_polynomial_predictions(data, window_size=10, degree=2):
    """Generate predictions using sliding window polynomial fit"""
    
    # Extract timestamps and features
    timestamps = data[:, 0]
    features = data[:, 1:]  # All columns except timestamp
    
    # We need at least window_size + 1 data points (to have something to predict)
    if len(data) <= window_size:
        raise ValueError(f"Not enough data points. Need more than {window_size} records.")
    
    # Prepare arrays for storing predictions and actuals
    predictions = []
    actuals = []
    prediction_timestamps = []
    all_models = []
    
    # Use a sliding window approach
    for i in range(window_size, len(data)):
        # Use the last window_size points to fit the model
        window_data = features[i-window_size:i]
        
        # Fit polynomial model
        model = PolynomialPredictor(window_size, degree)
        X = np.arange(window_size).reshape(-1, 1)  # Use indices as X
        model.fit(X, window_data)
        
        # Store the model
        all_models.append(model.to_json())
        
        # Predict the next point
        next_point_pred = model.predict(window_data)
        
        # Store prediction, actual, and timestamp
        predictions.append(next_point_pred)
        actuals.append(features[i])
        prediction_timestamps.append(timestamps[i])
    
    return np.array(predictions), np.array(actuals), np.array(prediction_timestamps), all_models

def plot_predictions_vs_actual(timestamps, predictions, actuals, degree, output_prefix='prediction'):
    """Plot predictions versus actual values for each feature"""
    n_features = predictions.shape[1]
    feature_names = ['Mid Price', 'Bid Price', 'Ask Price']
    
    # Create a figure with subplots for each feature
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
    fig.suptitle(f'Degree {degree} Polynomial Fit Predictions vs Actual Values', fontsize=16)
    
    if n_features == 1:
        axes = [axes]  # Make sure axes is always a list
    
    for i in range(n_features):
        ax = axes[i]
        
        # Plot actual values
        ax.plot(timestamps, actuals[:, i], 'b-', label='Actual', alpha=0.7)
        
        # Plot predicted values
        ax.plot(timestamps, predictions[:, i], 'r--', label='Predicted', alpha=0.7)
        
        # Add labels and legend
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i+1}'
        ax.set_title(f'{feature_name} - Prediction vs Actual')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    filename = f'{output_prefix}_vs_actual.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved prediction vs actual plot to {filename}")
    
    return fig

def plot_sample_polynomial_fits(data, window_size, degree, output_name='polynomial_fits.png'):
    """Plot some sample polynomial fits to visualize the approach"""
    # Extract features
    features = data[:, 1:]  # All columns except timestamp
    feature_names = ['Mid Price', 'Bid Price', 'Ask Price']
    
    # Take 3 random positions to show fits (near beginning, middle, and end)
    positions = [
        window_size + 5,  # Near beginning
        len(data) // 2,   # Middle
        len(data) - 5     # Near end
    ]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(positions), len(feature_names), 
                            figsize=(15, 4 * len(positions)))
    fig.suptitle(f'Sample Degree {degree} Polynomial Fits ({window_size}-point windows)', fontsize=16)
    
    for row, pos in enumerate(positions):
        for col, feature_idx in enumerate(range(features.shape[1])):
            ax = axes[row, col]
            
            # Get window data
            window_data = features[pos-window_size:pos, feature_idx]
            x_window = np.arange(window_size)
            
            # Fit polynomial
            coeffs = np.polyfit(x_window, window_data, degree)
            
            # Generate smooth curve from polynomial
            x_smooth = np.linspace(0, window_size, 100)
            y_smooth = np.polyval(coeffs, x_smooth)
            
            # Plot window points
            ax.scatter(x_window, window_data, color='blue', label='Data points')
            
            # Plot the polynomial fit
            ax.plot(x_smooth, y_smooth, 'r-', label=f'Degree {degree} fit')
            
            # Plot the prediction point
            prediction = np.polyval(coeffs, window_size)
            ax.scatter([window_size], [prediction], color='green', s=100, 
                      marker='*', label='Prediction')
            
            # Add actual next point if available
            if pos < len(features):
                actual = features[pos, feature_idx]
                ax.scatter([window_size], [actual], color='black', s=100,
                          marker='x', label='Actual')
            
            # Add labels
            feature_name = feature_names[col] if col < len(feature_names) else f'Feature {col+1}'
            position_label = ["Beginning", "Middle", "End"][row]
            ax.set_title(f'{feature_name} ({position_label})')
            ax.set_xlabel('Window Position')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            
            if row == 0 and col == 0:
                ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_name, dpi=300)
    print(f"Saved sample polynomial fits to {output_name}")
    
    return fig

def main():
    # Current date and time info
    current_time = "2025-04-08 07:40:44"  # From user
    current_user = "Lachy-Dauth"          # From user
    print(f"Script executed by {current_user} at {current_time} UTC")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate predictions using polynomial fits')
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('--output', default='polynomial_model.json', help='Path for the output model file')
    parser.add_argument('--window-size', type=int, default=10, help='Size of the sliding window for polynomial fits')
    parser.add_argument('--degree', type=int, default=2, help='Degree of the polynomial fit (1=linear, 2=quadratic, 3=cubic, ...)')
    
    args = parser.parse_args()
    
    # Load data from CSV
    print(f"Loading data from {args.input_csv}")
    data = load_data_from_csv(args.input_csv)
    
    # Generate predictions using polynomial fits
    print(f"Generating predictions using degree {args.degree} polynomial fits with window size {args.window_size}")
    predictions, actuals, timestamps, models = generate_polynomial_predictions(
        data, window_size=args.window_size, degree=args.degree
    )
    
    # Save the last model to a JSON file
    output_filename = f"poly_degree{args.degree}_model.json"
    if args.output != 'polynomial_model.json':  # If user provided custom name
        output_filename = args.output
        
    with open(output_filename, 'w') as f:
        json.dump(models[-1], f, indent=2)
    
    print(f"Final polynomial model saved to {output_filename}")
    
    # Plot and save sample polynomial fits
    fits_filename = f"poly_degree{args.degree}_fits.png"
    plot_sample_polynomial_fits(data, args.window_size, args.degree, fits_filename)
    
    # Plot and save the predictions vs actual values
    output_prefix = os.path.splitext(output_filename)[0]
    plot_predictions_vs_actual(timestamps, predictions, actuals, args.degree, output_prefix)
    
    # Calculate and print metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    print(f"\nPrediction Performance:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    
    # Feature-specific metrics
    feature_names = ['Mid Price', 'Bid Price', 'Ask Price']
    for i in range(predictions.shape[1]):
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i+1}'
        feature_mse = np.mean((predictions[:, i] - actuals[:, i]) ** 2)
        feature_mae = np.mean(np.abs(predictions[:, i] - actuals[:, i]))
        print(f"{feature_name} - MSE: {feature_mse:.6f}, MAE: {feature_mae:.6f}")

    # Calculate and print percentage errors for each feature
    print("\nPercentage Errors:")
    for i in range(predictions.shape[1]):
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i+1}'
        percentage_errors = 100 * np.abs(predictions[:, i] - actuals[:, i]) / actuals[:, i]
        mean_percentage_error = np.mean(percentage_errors)
        print(f"{feature_name} - Mean Percentage Error: {mean_percentage_error:.2f}%")
    
    print(f"\nPolynomial degree {args.degree} prediction analysis complete!")

if __name__ == "__main__":
    main()