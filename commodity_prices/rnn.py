import numpy as np
import json
import argparse
import os
import sys
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

def train_rnn(data, epochs=100, hidden_size=64, sequence_length=5):
    """Train an RNN on the provided data"""
    # Use all columns except the first one (timestamp) as features
    features = data[:, 1:]
    
    # Normalize the features
    normalized_features, mean, std = normalize_data(features)
    
    input_size = normalized_features.shape[1]
    output_size = normalized_features.shape[1]
    
    # Create the RNN model
    rnn = RNN(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    
    # Create training sequences
    print(f"Creating training sequences with length {sequence_length}")
    inputs = []
    targets = []
    
    for i in range(len(normalized_features) - sequence_length):
        inputs.append(normalized_features[i:i+sequence_length])
        targets.append(normalized_features[i+sequence_length])
    
    inputs = np.array(inputs)
    targets = np.array(targets)
    
    print(f"Training with {len(inputs)} sequences")
    
    # Train the RNN
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        
        for i in range(len(inputs)):
            input_seq = inputs[i]
            target_seq = targets[i]
            
            # Forward pass
            output = rnn.forward(input_seq)
            
            # Compute loss (mean squared error)
            loss = np.mean((output - target_seq.reshape(-1, 1)) ** 2)
            epoch_losses.append(loss)
            
            # Backward pass
            rnn.backward(output - target_seq.reshape(-1, 1))
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Add training statistics to the model
    rnn_dict = rnn.to_json()
    rnn_dict["training"] = {
        "epochs": epochs,
        "final_loss": float(losses[-1]),
        "sequence_length": sequence_length,
        "normalization_params": {
            "mean": mean.tolist(),
            "std": std.tolist()
        }
    }
    
    return rnn_dict

def main():
    # Print execution information
    print(f"Script executed by Lachy-Dauth at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train an RNN on time series data from a CSV file')
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('--output', default='rnn_model.json', help='Path for the output JSON model file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden-size', type=int, default=64, help='Size of the hidden layer')
    parser.add_argument('--sequence-length', type=int, default=5, help='Sequence length for training')
    
    args = parser.parse_args()
    
    # Load and preprocess the data
    print(f"Loading data from {args.input_csv}")
    data = load_data_from_csv(args.input_csv)
    
    # Train the RNN model
    print(f"Training RNN with {args.epochs} epochs and hidden size {args.hidden_size}")
    rnn_dict = train_rnn(
        data, 
        epochs=args.epochs, 
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length
    )
    
    # Save the model to a JSON file
    with open(args.output, 'w') as f:
        json.dump(rnn_dict, f, indent=2)
    
    print(f"RNN model saved to {args.output}")
    
    # Generate a sample prediction
    try:
        # Recreate RNN from the saved dictionary
        rnn = RNN.from_json(rnn_dict)
        
        # Get the last sequence from the data
        features = data[:, 1:]
        mean = np.array(rnn_dict["training"]["normalization_params"]["mean"])
        std = np.array(rnn_dict["training"]["normalization_params"]["std"])
        
        # Normalize the last sequence
        normalized_features = (features - mean) / std
        last_sequence = normalized_features[-args.sequence_length:]
        
        # Make a prediction
        prediction = rnn.predict(last_sequence)
        
        # Denormalize the prediction
        denormalized_prediction = (prediction.flatten() * std) + mean
        
        print("\nSample prediction for the next timestep:")
        print(f"Mid price: {denormalized_prediction[0]:.2f}")
        print(f"Bid price: {denormalized_prediction[1]:.2f}")
        print(f"Ask price: {denormalized_prediction[2]:.2f}")
        
    except Exception as e:
        print(f"Error generating sample prediction: {e}")

if __name__ == "__main__":
    main()