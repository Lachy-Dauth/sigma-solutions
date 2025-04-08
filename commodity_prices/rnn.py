import numpy as np
import json
import argparse
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        # Initialize weights
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # Initialize weights with small random values
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((output_size, 1))  # output bias
        
    def forward(self, inputs):
        h_prev = np.zeros((self.Wh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = {0: h_prev}

        # Perform forward pass
        for i, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            h_prev = np.tanh(np.dot(self.Wx, x) + np.dot(self.Wh, h_prev) + self.bh)
            self.last_hs[i + 1] = h_prev

        # Compute the output
        self.last_o = np.dot(self.Wy, h_prev) + self.by
        return self.last_o

    def backward(self, d_y, learning_rate=2e-2):
        n = len(self.last_inputs)
        d_Wx, d_Wh, d_Wy = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.Wy)
        d_bh, d_by = np.zeros_like(self.bh), np.zeros_like(self.by)
        d_h = np.dot(self.Wy.T, d_y)

        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            d_bh += temp
            d_Wx += np.dot(temp, self.last_inputs[t].reshape(1, -1))
            d_Wh += np.dot(temp, self.last_hs[t].reshape(1, -1))
            d_h = np.dot(self.Wh.T, temp)

        d_Wy = np.dot(d_y, self.last_hs[n].T)
        d_by = d_y

        # Clip gradients to prevent exploding gradients
        for d in [d_Wx, d_Wh, d_Wy, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Update weights and biases using gradient descent
        self.Wx -= learning_rate * d_Wx
        self.Wh -= learning_rate * d_Wh
        self.Wy -= learning_rate * d_Wy
        self.bh -= learning_rate * d_bh
        self.by -= learning_rate * d_by

    def predict(self, inputs):
        output = self.forward(inputs)
        return output
    
    def to_json(self):
        """Convert the RNN model parameters to a JSON-serializable dictionary"""
        model_dict = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "weights": {
                "Wx": self.Wx.tolist(),
                "Wh": self.Wh.tolist(),
                "Wy": self.Wy.tolist(),
                "bh": self.bh.tolist(),
                "by": self.by.tolist()
            },
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "created_by": os.environ.get("USER", "unknown")
            }
        }
        return model_dict
    
    @classmethod
    def from_json(cls, model_dict):
        """Create a new RNN instance from a JSON dictionary"""
        rnn = cls(
            input_size=model_dict["input_size"],
            output_size=model_dict["output_size"],
            hidden_size=model_dict["hidden_size"]
        )
        
        # Load the weights
        rnn.Wx = np.array(model_dict["weights"]["Wx"])
        rnn.Wh = np.array(model_dict["weights"]["Wh"])
        rnn.Wy = np.array(model_dict["weights"]["Wy"])
        rnn.bh = np.array(model_dict["weights"]["bh"])
        rnn.by = np.array(model_dict["weights"]["by"])
        
        return rnn

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

def normalize_data(data):
    """Normalize the data to have zero mean and unit variance"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    return (data - mean) / std, mean, std

def create_sequences(data, sequence_length):
    """Create input/output sequences from the time series data"""
    inputs = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        inputs.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    
    return np.array(inputs), np.array(targets)

def train_rnn(data, validation_split=0.2, epochs=100, hidden_size=64, sequence_length=5):
    """Train an RNN on the provided data with validation"""
    # Use all columns except the first one (timestamp) as features
    features = data[:, 1:]
    timestamps = data[:, 0]
    
    # Normalize the features
    normalized_features, mean, std = normalize_data(features)
    
    input_size = normalized_features.shape[1]
    output_size = normalized_features.shape[1]
    
    # Create the RNN model
    rnn = RNN(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    
    # Create sequences
    print(f"Creating sequences with length {sequence_length}")
    inputs, targets = create_sequences(normalized_features, sequence_length)
    
    # Split into training and validation sets
    split_idx = int(len(inputs) * (1 - validation_split))
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    print(f"Training with {len(train_inputs)} sequences, validating with {len(val_inputs)} sequences")
    
    # Train the RNN
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        train_epoch_losses = []
        for i in range(len(train_inputs)):
            input_seq = train_inputs[i]
            target_seq = train_targets[i]
            
            # Forward pass
            output = rnn.forward(input_seq)
            
            # Compute loss (mean squared error)
            loss = np.mean((output - target_seq.reshape(-1, 1)) ** 2)
            train_epoch_losses.append(loss)
            
            # Backward pass
            rnn.backward(output - target_seq.reshape(-1, 1))
        
        avg_train_loss = np.mean(train_epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_epoch_losses = []
        for i in range(len(val_inputs)):
            input_seq = val_inputs[i]
            target_seq = val_targets[i]
            
            # Forward pass (no backward pass for validation)
            output = rnn.forward(input_seq)
            
            # Compute validation loss
            loss = np.mean((output - target_seq.reshape(-1, 1)) ** 2)
            val_epoch_losses.append(loss)
        
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Generate predictions for plotting
    all_predictions = []
    for i in range(len(inputs)):
        input_seq = inputs[i]
        prediction = rnn.predict(input_seq)
        all_predictions.append(prediction.flatten())
    
    all_predictions = np.array(all_predictions)
    
    # Denormalize predictions and targets
    denorm_predictions = (all_predictions * std) + mean
    denorm_targets = (targets * std) + mean
    
    # Add sequence_length to get the correct timestamps
    prediction_timestamps = timestamps[sequence_length:]
    
    # Add training statistics to the model
    rnn_dict = rnn.to_json()
    rnn_dict["training"] = {
        "epochs": epochs,
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "sequence_length": sequence_length,
        "normalization_params": {
            "mean": mean.tolist(),
            "std": std.tolist()
        }
    }
    
    return rnn_dict, denorm_predictions, denorm_targets, prediction_timestamps, train_losses, val_losses

def plot_predictions_vs_actual(timestamps, predictions, actuals, output_prefix='prediction'):
    """Plot predictions versus actual values for each feature"""
    n_features = predictions.shape[1]
    feature_names = ['Mid Price', 'Bid Price', 'Ask Price']
    
    # Create a figure with subplots for each feature
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
    fig.suptitle('RNN Predictions vs Actual Values', fontsize=16)
    
    for i in range(n_features):
        ax = axes[i] if n_features > 1 else axes
        
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

def plot_training_loss(train_losses, val_losses=None, output_name='training_loss.png'):
    """Plot the training and validation loss over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    
    if val_losses is not None:
        plt.plot(val_losses, 'r-', label='Validation Loss')
    
    plt.title('RNN Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"Saved training loss plot to {output_name}")

def main():
    # Current date and time info
    current_time = "2025-04-08 07:10:32" # Hardcoded as provided
    current_user = "Lachy-Dauth" # Hardcoded as provided
    print(f"Script executed by {current_user} at {current_time}")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train an RNN on time series data from a CSV file')
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('--output', default='rnn_model.json', help='Path for the output JSON model file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden-size', type=int, default=64, help='Size of the hidden layer')
    parser.add_argument('--sequence-length', type=int, default=5, help='Sequence length for training')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Fraction of data to use for validation')
    
    args = parser.parse_args()
    
    # Load and preprocess the data
    print(f"Loading data from {args.input_csv}")
    data = load_data_from_csv(args.input_csv)
    
    # Train the RNN model and get predictions
    print(f"Training RNN with {args.epochs} epochs and hidden size {args.hidden_size}")
    rnn_dict, predictions, actuals, timestamps, train_losses, val_losses = train_rnn(
        data, 
        validation_split=args.validation_split,
        epochs=args.epochs, 
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length
    )
    
    # Save the model to a JSON file
    with open(args.output, 'w') as f:
        json.dump(rnn_dict, f, indent=2)
    
    print(f"RNN model saved to {args.output}")
    
    # Plot and save the training/validation loss curves
    plot_training_loss(train_losses, val_losses)
    
    # Plot and save the predictions vs actual values
    output_prefix = os.path.splitext(args.output)[0]
    plot_predictions_vs_actual(timestamps, predictions, actuals, output_prefix)
    
    # Print some metrics
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

if __name__ == "__main__":
    main()