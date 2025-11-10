"""
Encoder-Decoder Architecture for Neural Machine Translation
============================================================
Implements sequence-to-sequence model for translation tasks using LSTM-based
encoder-decoder architecture with attention mechanism.

This will be split into 3 notebooks:
1. encoder_decoder.ipynb - Basic Seq2Seq (THIS FILE)
2. attention_mechanism.ipynb - Adding attention
3. translation_demo.ipynb - Full translation pipeline
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import re
from collections import Counter

# ==============================================================
# Ensure the SequentialModeling directory is discoverable
# ==============================================================

# Get current working directory
current_dir = os.getcwd()

# Check if SequentialModeling folder exists in current directory
if not os.path.exists(os.path.join(current_dir, "SequentialModeling")):
    # If not, assume notebook is in a subfolder (like notebooks/)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
else:
    project_root = current_dir

# Add project root to system path (for imports)
if project_root not in sys.path:
    sys.path.append(project_root)

# ==============================================================
# Import LSTM implementations from SequentialModeling/layers.py
# ==============================================================

try:
    from SequentialModeling.layers import LSTMCell, LSTM
    print("✓ Successfully imported LSTMCell and LSTM from SequentialModeling/layers.py")
except ImportError as e:
    print("⚠ ImportError: Could not import LSTMCell or LSTM.")
    print("  Make sure SequentialModeling/layers.py exists and contains both classes.")
    print("  Full error:", e)
    raise



class Vocabulary:
    """Manages vocabulary for source and target languages."""
    
    def __init__(self, language: str):
        self.language = language
        self.token2idx = {}
        self.idx2token = {}
        self.token_counts = Counter()
        
        # Special tokens
        self.PAD_token = 0
        self.SOS_token = 1  # Start of sequence
        self.EOS_token = 2  # End of sequence
        self.UNK_token = 3  # Unknown token
        
        self.token2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2token = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.n_tokens = 4
    
    def add_sentence(self, sentence: str):
        """Add all tokens from sentence to vocabulary."""
        tokens = self.tokenize(sentence)
        for token in tokens:
            self.add_token(token)
    
    def add_token(self, token: str):
        """Add a single token to vocabulary."""
        self.token_counts[token] += 1
        if token not in self.token2idx:
            self.token2idx[token] = self.n_tokens
            self.idx2token[self.n_tokens] = token
            self.n_tokens += 1
    
    def tokenize(self, sentence: str) -> List[str]:
        """Simple whitespace tokenization with punctuation handling."""
        # Lowercase and basic tokenization
        sentence = sentence.lower()
        # Add space around punctuation
        sentence = re.sub(r'([.!?])', r' \1', sentence)
        tokens = sentence.split()
        return tokens
    
    def encode(self, sentence: str) -> List[int]:
        """Convert sentence to list of token indices."""
        tokens = self.tokenize(sentence)
        indices = [self.token2idx.get(token, self.UNK_token) for token in tokens]
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Convert indices back to sentence."""
        tokens = []
        for idx in indices:
            if skip_special and idx in [self.PAD_token, self.SOS_token, self.EOS_token]:
                continue
            tokens.append(self.idx2token.get(idx, '<UNK>'))
        return ' '.join(tokens)


# ==================== ENCODER ====================

class Encoder:
    """
    LSTM-based encoder that processes source sequence.
    
    Input: Source sentence
    Output: Final hidden state (context vector) + all hidden states
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, seed: int = 42):
        """
        Initialize encoder.
        
        Args:
            vocab_size: Size of source vocabulary
            embedding_dim: Dimension of embeddings
            hidden_size: Size of LSTM hidden state
        """
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # LSTM cell
        self.lstm_cell = LSTMCell(embedding_dim, hidden_size, seed)
        
    def forward(self, input_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Forward pass through encoder.
        
        Args:
            input_indices: List of token indices from source sentence
            
        Returns:
            final_hidden: Final hidden state (context vector)
            final_cell: Final cell state
            all_hidden: All hidden states (for attention later)
        """
        seq_length = len(input_indices)
        
        # Initialize states
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        all_hidden = []
        
        # Process each token
        for t in range(seq_length):
            # Get embedding
            token_idx = input_indices[t]
            x = self.embeddings[token_idx].reshape(-1, 1)
            
            # LSTM step
            h, c = self.lstm_cell.forward(x, h, c)
            all_hidden.append(h.copy())
        
        return h, c, all_hidden

# ==================== DECODER ====================

class Decoder:
    """
    LSTM-based decoder that generates target sequence.
    
    Input: Context vector from encoder + previous token
    Output: Next token prediction
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, seed: int = 42):
        """
        Initialize decoder.
        
        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of embeddings
            hidden_size: Size of LSTM hidden state (should match encoder)
        """
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # LSTM cell
        self.lstm_cell = LSTMCell(embedding_dim, hidden_size, seed)
        
        # Output layer (hidden state -> vocabulary probabilities)
        self.W_out = np.random.randn(vocab_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros((vocab_size, 1))
        
    def forward_step(self, token_idx: int, h: np.ndarray, c: np.ndarray) -> Tuple:
        """
        Single decoder step.
        
        Args:
            token_idx: Previous token index
            h: Previous hidden state
            c: Previous cell state
            
        Returns:
            logits: Output logits over vocabulary
            h_next: Next hidden state
            c_next: Next cell state
        """
        # Get embedding
        x = self.embeddings[token_idx].reshape(-1, 1)
        
        # LSTM step
        h_next, c_next = self.lstm_cell.forward(x, h, c)
        
        # Output projection
        logits = self.W_out @ h_next + self.b_out
        
        return logits, h_next, c_next
    
    def forward(self, target_indices: List[int], encoder_hidden: np.ndarray,
                encoder_cell: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through entire target sequence (training mode).
        
        Args:
            target_indices: Target sequence indices (with <SOS> prepended)
            encoder_hidden: Initial hidden state from encoder
            encoder_cell: Initial cell state from encoder
            
        Returns:
            all_logits: Logits for each time step
            all_hidden: Hidden states for each time step
        """
        h = encoder_hidden
        c = encoder_cell
        
        all_logits = []
        all_hidden = []
        
        # Teacher forcing: use ground truth tokens as input
        for t in range(len(target_indices) - 1):
            token_idx = target_indices[t]
            logits, h, c = self.forward_step(token_idx, h, c)
            
            all_logits.append(logits)
            all_hidden.append(h)
        
        return all_logits, all_hidden
# ==================== SEQ2SEQ MODEL ====================

class Seq2Seq:
    """Complete sequence-to-sequence model."""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 embedding_dim: int, hidden_size: int, learning_rate: float = 0.01):
        """
        Initialize Seq2Seq model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            embedding_dim: Embedding dimension
            hidden_size: Hidden state size
            learning_rate: Learning rate for training
        """
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_size)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_size)
        
        self.learning_rate = learning_rate
        self.loss_history = []
        
    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def compute_loss(self, logits_list: List[np.ndarray], target_indices: List[int]) -> float:
        """Compute cross-entropy loss."""
        loss = 0
        for logits, target_idx in zip(logits_list, target_indices[1:]):  # Skip <SOS>
            probs = self.softmax(logits)
            loss += -np.log(probs[target_idx, 0] + 1e-10)
        return loss / len(logits_list)
    
    def train_step(self, src_indices: List[int], tgt_indices: List[int]) -> float:
        """
        Single training step.
        
        Args:
            src_indices: Source sequence indices
            tgt_indices: Target sequence indices (should include <SOS> and <EOS>)
        """
        # Encode source
        enc_hidden, enc_cell, _ = self.encoder.forward(src_indices)
        
        # Decode target
        logits_list, _ = self.decoder.forward(tgt_indices, enc_hidden, enc_cell)
        
        # Compute loss
        loss = self.compute_loss(logits_list, tgt_indices)
        
        # Simplified gradient update (approximate)
        for i, (logits, target_idx) in enumerate(zip(logits_list, tgt_indices[1:])):
            probs = self.softmax(logits)
            dlogits = probs.copy()
            dlogits[target_idx, 0] -= 1
            
            # Clip gradients
            dlogits = np.clip(dlogits, -5, 5)
            
            # Update decoder output layer
            self.decoder.W_out -= self.learning_rate * dlogits * 0.001
            self.decoder.b_out -= self.learning_rate * dlogits * 0.001
        
        self.loss_history.append(loss)
        return loss
    
    def translate(self, src_indices: List[int], tgt_vocab: Vocabulary,
                  max_length: int = 20, method: str = 'greedy') -> List[int]:
        """
        Translate source sequence to target sequence.
        
        Args:
            src_indices: Source sequence indices
            tgt_vocab: Target vocabulary
            max_length: Maximum length of translation
            method: 'greedy' or 'sample'
            
        Returns:
            predicted_indices: Predicted target sequence
        """
        # Encode source
        enc_hidden, enc_cell, _ = self.encoder.forward(src_indices)
        
        # Initialize decoder with <SOS>
        h = enc_hidden
        c = enc_cell
        predicted_indices = [tgt_vocab.SOS_token]
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Decoder step
            logits, h, c = self.decoder.forward_step(predicted_indices[-1], h, c)
            
            # Get next token
            if method == 'greedy':
                next_token = np.argmax(logits)
            else:  # sample
                probs = self.softmax(logits)
                next_token = np.random.choice(len(probs), p=probs[:, 0])
            
            predicted_indices.append(next_token)
            
            # Stop if <EOS> generated
            if next_token == tgt_vocab.EOS_token:
                break
        
        return predicted_indices

# ==================== DATA PREPARATION ====================

def load_translation_pairs() -> List[Tuple[str, str]]:
    """
    Load sample translation pairs (English -> French).
    In practice, you'd load from a real dataset.
    """
    pairs = [
        ("hello", "bonjour"),
        ("goodbye", "au revoir"),
        ("thank you", "merci"),
        ("how are you", "comment allez vous"),
        ("good morning", "bonjour"),
        ("good night", "bonne nuit"),
        ("i love you", "je t aime"),
        ("what is your name", "comment vous appelez vous"),
        ("my name is john", "je m appelle john"),
        ("where is the station", "ou est la gare"),
        ("i am hungry", "j ai faim"),
        ("i am tired", "je suis fatigue"),
        ("yes", "oui"),
        ("no", "non"),
        ("please", "s il vous plait"),
        ("excuse me", "excusez moi"),
        ("i do not understand", "je ne comprends pas"),
        ("how much", "combien"),
        ("where", "ou"),
        ("when", "quand"),
        ("what", "quoi"),
        ("who", "qui"),
        ("why", "pourquoi"),
        ("how", "comment"),
    ]
    return pairs

