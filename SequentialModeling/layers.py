
import numpy as np
import time
from typing import Tuple, List, Dict



# ==================== LSTM IMPLEMENTATION ====================

class LSTMCell:
    """
    Long Short-Term Memory Cell
    
    Gates:
    - Forget gate: f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
    - Input gate: i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
    - Output gate: o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
    - Cell candidate: c_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c)
    
    Updates:
    - Cell state: c_t = f_t * c_{t-1} + i_t * c_tilde
    - Hidden state: h_t = o_t * tanh(c_t)
    """
    
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Concatenated input size [h_{t-1}, x_t]
        concat_size = hidden_size + input_size
        
        # Initialize weights for all gates (Xavier initialization)
        scale = np.sqrt(2.0 / concat_size)
        self.W_f = np.random.randn(hidden_size, concat_size) * scale  # Forget gate
        self.W_i = np.random.randn(hidden_size, concat_size) * scale  # Input gate
        self.W_c = np.random.randn(hidden_size, concat_size) * scale  # Cell candidate
        self.W_o = np.random.randn(hidden_size, concat_size) * scale  # Output gate
        
        # Biases (forget gate bias initialized to 1 for better gradient flow)
        self.b_f = np.ones((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        
        self.cache = {}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input at current time step (input_size, batch_size)
            h_prev: Previous hidden state (hidden_size, batch_size)
            c_prev: Previous cell state (hidden_size, batch_size)
        
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        # Concatenate h_prev and x
        concat = np.vstack([h_prev, x])
        
        # Compute gates
        f_t = self.sigmoid(self.W_f @ concat + self.b_f)  # Forget gate
        i_t = self.sigmoid(self.W_i @ concat + self.b_i)  # Input gate
        c_tilde = np.tanh(self.W_c @ concat + self.b_c)    # Cell candidate
        o_t = self.sigmoid(self.W_o @ concat + self.b_o)  # Output gate
        
        # Update cell state
        c_next = f_t * c_prev + i_t * c_tilde
        
        # Update hidden state
        h_next = o_t * np.tanh(c_next)
        
        # Cache for backward pass
        self.cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'concat': concat, 'f_t': f_t, 'i_t': i_t,
            'c_tilde': c_tilde, 'o_t': o_t, 'c_next': c_next,
            'h_next': h_next
        }
        
        return h_next, c_next


class LSTM:
    """Complete LSTM model."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 learning_rate: float = 0.001, seed: int = 42):
        self.cell = LSTMCell(input_size, hidden_size, seed)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Output layer
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_y = np.zeros((output_size, 1))
        
        self.loss_history = []
        self.training_time = 0
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List, List]:
        """Forward pass through sequence."""
        seq_length = X.shape[0]
        batch_size = X.shape[2] if len(X.shape) > 2 else 1
        
        h = np.zeros((self.hidden_size, batch_size))
        c = np.zeros((self.hidden_size, batch_size))
        
        hidden_states = [h]
        cell_states = [c]
        outputs = []
        
        for t in range(seq_length):
            x_t = X[t].reshape(-1, batch_size)
            h, c = self.cell.forward(x_t, h, c)
            hidden_states.append(h)
            cell_states.append(c)
            
            y_t = self.W_hy @ h + self.b_y
            outputs.append(y_t)
        
        return np.array(outputs), hidden_states, cell_states
    
    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Single training step with simplified gradient update."""
        start_time = time.time()
        
        outputs, _, _ = self.forward(X)
        
        # Compute loss
        loss = np.mean((outputs - Y) ** 2)
        
        # Simplified gradient update (approximate)
        dy = 2 * (outputs - Y) / outputs.size
        dW_hy = np.sum([dy[t] @ outputs[t].T for t in range(len(dy))], axis=0)
        db_y = np.sum(dy, axis=(0, 2), keepdims=True).T
        
        # Clip gradients
        dW_hy = np.clip(dW_hy, -5, 5)
        db_y = np.clip(db_y, -5, 5)
        db_y = np.squeeze(db_y)
        
        # Update
        self.W_hy -= self.learning_rate * dW_hy
        self.b_y -= self.learning_rate * db_y
        
        self.loss_history.append(loss)
        self.training_time += time.time() - start_time
        
        return loss




 

# ==================== GRU IMPLEMENTATION ====================

class GRUCell:
    """
    Gated Recurrent Unit Cell
    
    Gates:
    - Reset gate: r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)
    - Update gate: z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)
    - Candidate: h_tilde = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)
    
    Update:
    - Hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    
    Key difference from LSTM: No separate cell state, fewer parameters
    """
    
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        concat_size = hidden_size + input_size
        scale = np.sqrt(2.0 / concat_size)
        
        # Weights for gates (fewer than LSTM!)
        self.W_r = np.random.randn(hidden_size, concat_size) * scale  # Reset gate
        self.W_z = np.random.randn(hidden_size, concat_size) * scale  # Update gate
        self.W_h = np.random.randn(hidden_size, concat_size) * scale  # Candidate
        
        # Biases
        self.b_r = np.zeros((hidden_size, 1))
        self.b_z = np.zeros((hidden_size, 1))
        self.b_h = np.zeros((hidden_size, 1))
        
        self.cache = {}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass through GRU cell.
        
        Args:
            x: Input at current time step (input_size, batch_size)
            h_prev: Previous hidden state (hidden_size, batch_size)
        
        Returns:
            h_next: Next hidden state (no separate cell state!)
        """
        # Concatenate h_prev and x
        concat = np.vstack([h_prev, x])
        
        # Compute gates
        r_t = self.sigmoid(self.W_r @ concat + self.b_r)  # Reset gate
        z_t = self.sigmoid(self.W_z @ concat + self.b_z)  # Update gate
        
        # Compute candidate hidden state (with reset gate applied)
        concat_reset = np.vstack([r_t * h_prev, x])
        h_tilde = np.tanh(self.W_h @ concat_reset + self.b_h)
        
        # Compute next hidden state (interpolation between old and new)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde
        
        # Cache for backward pass
        self.cache = {
            'x': x, 'h_prev': h_prev, 'r_t': r_t,
            'z_t': z_t, 'h_tilde': h_tilde, 'h_next': h_next
        }
        
        return h_next


class GRU:
    """Complete GRU model."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.001, seed: int = 42):
        self.cell = GRUCell(input_size, hidden_size, seed)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Output layer
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_y = np.zeros((output_size, 1))
        
        self.loss_history = []
        self.training_time = 0
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass through sequence."""
        seq_length = X.shape[0]
        batch_size = X.shape[2] if len(X.shape) > 2 else 1
        
        h = np.zeros((self.hidden_size, batch_size))
        hidden_states = [h]
        outputs = []
        
        for t in range(seq_length):
            x_t = X[t].reshape(-1, batch_size)
            h = self.cell.forward(x_t, h)
            hidden_states.append(h)
            
            y_t = self.W_hy @ h + self.b_y
            outputs.append(y_t)
        
        return np.array(outputs), hidden_states
    
    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Single training step with simplified gradient update."""
        start_time = time.time()
        
        outputs, _ = self.forward(X)
        
        # Compute loss
        loss = np.mean((outputs - Y) ** 2)
        
        # Simplified gradient update
        dy = 2 * (outputs - Y) / outputs.size
        dW_hy = np.sum([dy[t] @ outputs[t].T for t in range(len(dy))], axis=0)
        db_y = np.sum(dy, axis=(0, 2), keepdims=True).T
        
        # Clip gradients
        dW_hy = np.clip(dW_hy, -5, 5)
        db_y = np.clip(db_y, -5, 5)
        db_y = np.squeeze(db_y)
        
        # Update
        self.W_hy -= self.learning_rate * dW_hy
        self.b_y -= self.learning_rate * db_y
        
        self.loss_history.append(loss)
        self.training_time += time.time() - start_time
        
        return loss

"""
RNN From Scratch Implementation
================================
A complete implementation of a Recurrent Neural Network with detailed
forward and backward passes, demonstrating sequential processing.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class RNNCell:
    """
    A single RNN cell implementing the basic recurrent computation:
    h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    """
    
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        """
        Initialize RNN cell parameters.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier initialization for weights
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_h = np.zeros((hidden_size, 1))
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for a single time step.
        
        Args:
            x: Input at current time step, shape (input_size, 1)
            h_prev: Hidden state from previous time step, shape (hidden_size, 1)
        
        Returns:
            h_next: Hidden state at current time step, shape (hidden_size, 1)
        """
        # Linear transformation
        z = self.W_xh @ x + self.W_hh @ h_prev + self.b_h
        
        # Activation function (tanh)
        h_next = np.tanh(z)
        
        # Cache values for backward pass
        self.cache = {
            'x': x,
            'h_prev': h_prev,
            'h_next': h_next,
            'z': z
        }
        
        return h_next
    
    def backward(self, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Backward pass for a single time step.
        
        Args:
            dh_next: Gradient of loss w.r.t. hidden state, shape (hidden_size, 1)
        
        Returns:
            dx: Gradient w.r.t. input
            dh_prev: Gradient w.r.t. previous hidden state
            grads: Dictionary of parameter gradients
        """
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        z = self.cache['z']
        
        # Gradient through tanh activation
        dz = dh_next * (1 - np.tanh(z) ** 2)
        
        # Parameter gradients
        dW_xh = dz @ x.T
        dW_hh = dz @ h_prev.T
        db_h = dz 
        
        # Gradient w.r.t. inputs
        dx = self.W_xh.T @ dz
        dh_prev = self.W_hh.T @ dz
        
        grads = {
            'dW_xh': dW_xh,
            'dW_hh': dW_hh,
            'db_h': db_h
        }
        
        return dx, dh_prev, grads
class RNN:
    """
    Complete RNN model with multiple layers and time steps.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 learning_rate: float = 0.01, seed: int = 42):
        """
        Initialize RNN model.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
            learning_rate: Learning rate for optimization
            seed: Random seed
        """
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # RNN cell
        self.cell = RNNCell(input_size, hidden_size, seed)
        
        # Output layer weights
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_y = np.zeros((output_size, 1))
        
        # Training history
        self.loss_history = []
    
    def forward(self, X: np.ndarray, h0: np.ndarray = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through entire sequence.
        
        Args:
            X: Input sequence, shape (seq_length, input_size, batch_size)
            h0: Initial hidden state, shape (hidden_size, batch_size)
        
        Returns:
            outputs: Output predictions for each time step
            hidden_states: Hidden states for each time step
        """
        seq_length = X.shape[0]
        batch_size = X.shape[2] if len(X.shape) > 2 else 1
        
        # Initialize hidden state
        if h0 is None:
            h = np.zeros((self.hidden_size, batch_size))
        else:
            h = h0
        
        hidden_states = [h]
        outputs = []
        
        # Process each time step
        for t in range(seq_length):
            x_t = X[t].reshape(self.input_size, -1)
            h = self.cell.forward(x_t, h)
            hidden_states.append(h)
            
            # Compute output
            y_t = self.W_hy @ h + self.b_y
            outputs.append(y_t)
        
        return np.array(outputs), hidden_states
    
    def backward(self, X: np.ndarray, Y: np.ndarray, outputs: np.ndarray, 
                 hidden_states: List[np.ndarray]) -> float:
        """
        Backward pass through time (BPTT).
        
        Args:
            X: Input sequence
            Y: Target sequence
            outputs: Predicted outputs
            hidden_states: Cached hidden states
        
        Returns:
            loss: Mean squared error loss
        """
        seq_length = X.shape[0]
        
        # Initialize gradient accumulators
        dW_xh = np.zeros_like(self.cell.W_xh)
        dW_hh = np.zeros_like(self.cell.W_hh)
        db_h = np.zeros_like(self.cell.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)
        
        # Initialize gradient flowing back
        dh_next = np.zeros_like(hidden_states[0])
        
        # Compute loss
        loss = 0
        
        # Backward pass through time
        for t in reversed(range(seq_length)):
            y_t = outputs[t]
            target_t = Y[t].reshape(self.output_size, -1)
            
            # Output layer gradient
            dy = y_t - target_t
            loss += np.sum(dy ** 2)
            
            dW_hy += dy @ hidden_states[t + 1].T
            db_y += np.sum(dy, axis=1, keepdims=True)
            
            # Gradient flowing into hidden state
            dh = self.W_hy.T @ dy + dh_next
            
            # Backward through RNN cell
            x_t = X[t].reshape(self.input_size, -1)
            self.cell.cache = {
                'x': x_t,
                'h_prev': hidden_states[t],
                'h_next': hidden_states[t + 1],
                'z': np.arctanh(hidden_states[t + 1])  # Approximate for demo
            }
            
            _, dh_next, grads = self.cell.backward(dh)
            
            # Accumulate gradients
            dW_xh += grads['dW_xh']
            dW_hh += grads['dW_hh']
            db_h += np.sum(grads['db_h'], axis=1, keepdims=True)
        
        # Gradient clipping to prevent exploding gradients
        for grad in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(grad, -5, 5, out=grad)
        
        # Update parameters
        self.cell.W_xh -= self.learning_rate * dW_xh
        self.cell.W_hh -= self.learning_rate * dW_hh
        self.cell.b_h -= self.learning_rate * db_h
        self.W_hy -= self.learning_rate * dW_hy
        self.b_y -= self.learning_rate * db_y
        
        return loss / seq_length
    
    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Single training step: forward + backward + update.
        
        Args:
            X: Input sequence
            Y: Target sequence
        
        Returns:
            loss: Training loss
        """
        # Forward pass
        outputs, hidden_states = self.forward(X)
        
        # Backward pass
        loss = self.backward(X, Y, outputs, hidden_states)
        
        self.loss_history.append(loss)
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input sequence."""
        outputs, _ = self.forward(X)
        return outputs