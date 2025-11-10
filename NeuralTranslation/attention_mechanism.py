"""
Attention Mechanism for Neural Machine Translation
==================================================
Implements Bahdanau (additive) attention mechanism that solves the bottleneck
problem in encoder-decoder models. Visualizes attention weights to show which
source words the model focuses on when generating each target word.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

# ==============================================================
# Ensure the SequentialModeling directory and models.py are discoverable
# ==============================================================

# Get current working directory
current_dir = os.getcwd()

# Check if SequentialModeling folder exists in current directory
if os.path.exists(os.path.join(current_dir, "SequentialModeling")):
    project_root = current_dir
else:
    # If not, assume notebook is inside a subfolder (like notebooks/)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Add project root to Python path if not already present
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"✓ Project root detected: {project_root}")

# ==============================================================
# Import modules from existing implementations
# ==============================================================

try:
    from SequentialModeling.layers import LSTMCell, LSTM
    print("✓ Successfully imported LSTMCell and LSTM from SequentialModeling/layers.py")
except ImportError as e:
    print("⚠ ImportError: Could not import LSTMCell or LSTM.")
    print("  Make sure SequentialModeling/layers.py exists and defines both classes.")
    print("  Full error:", e)
    raise

try:
    from models import Encoder, Vocabulary, load_translation_pairs
    print("✓ Successfully imported Encoder, Vocabulary, and load_translation_pairs from models.py")
except ImportError as e:
    print("⚠ ImportError: Could not import Encoder, Vocabulary, or load_translation_pairs.")
    print("  Make sure models.py exists and defines them correctly.")
    print("  Full error:", e)
    raise
# ==================== ATTENTION LAYER (Bahdanau / Additive) ====================

class BahdanauAttention:
    """
    Bahdanau (Additive) Attention.

    score(h_dec, h_enc_j) = v^T * tanh(W_decoder * h_dec + W_encoder * h_enc_j)

    Returns:
        context_vector: (hidden_size, 1)
        attention_weights: (src_length,)  -- numpy 1-D array
    """
    def __init__(self, hidden_size: int, seed: int = 42):
        np.random.seed(seed)
        self.hidden_size = hidden_size

        # Projection matrices
        self.W_decoder = np.random.randn(hidden_size, hidden_size) * 0.01  # W1
        self.W_encoder = np.random.randn(hidden_size, hidden_size) * 0.01  # W2

        # v vector (project to scalar)
        self.v = np.random.randn(1, hidden_size) * 0.01

    def compute_scores(self, decoder_hidden: np.ndarray,
                       encoder_hiddens: List[np.ndarray]) -> np.ndarray:
        """
        Compute raw scores for each encoder hidden state.

        Args:
            decoder_hidden: (hidden_size, 1)
            encoder_hiddens: list of (hidden_size, 1), length = src_len

        Returns:
            scores: (src_len,) 1-D numpy array of floats
        """
        decoder_proj = self.W_decoder @ decoder_hidden  # (hidden_size, 1)
        scores = np.zeros(len(encoder_hiddens), dtype=float)

        for i, h_enc in enumerate(encoder_hiddens):
            enc_proj = self.W_encoder @ h_enc  # (hidden_size, 1)
            combined = np.tanh(decoder_proj + enc_proj)  # (hidden_size, 1)
            score = (self.v @ combined)[0, 0]  # scalar
            scores[i] = score
        return scores

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax for 1-D array -> returns 1-D array."""
        x = np.array(x, dtype=float)
        if x.size == 0:
            return x
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def forward(self, decoder_hidden: np.ndarray,
                encoder_hiddens: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute context vector and normalized attention weights.

        Returns:
            context_vector: (hidden_size, 1)
            attention_weights: (src_len,) 1-D numpy array
        """
        if len(encoder_hiddens) == 0:
            # empty encoder case (defensive)
            return np.zeros((self.hidden_size, 1)), np.array([])

        scores = self.compute_scores(decoder_hidden, encoder_hiddens)  # (src_len,)
        attn_weights = self.softmax(scores)  # (src_len,)

        # Context: weighted sum of encoder hidden states
        context = np.zeros((self.hidden_size, 1))
        for w, h_enc in zip(attn_weights, encoder_hiddens):
            context += w * h_enc  # (hidden_size, 1)

        return context, attn_weights
# ==================== ATTENTION LAYER (Bahdanau / Additive) ====================

class BahdanauAttention:
    """
    Bahdanau (Additive) Attention.

    score(h_dec, h_enc_j) = v^T * tanh(W_decoder * h_dec + W_encoder * h_enc_j)

    Returns:
        context_vector: (hidden_size, 1)
        attention_weights: (src_length,)  -- numpy 1-D array
    """
    def __init__(self, hidden_size: int, seed: int = 42):
        np.random.seed(seed)
        self.hidden_size = hidden_size

        # Projection matrices
        self.W_decoder = np.random.randn(hidden_size, hidden_size) * 0.01  # W1
        self.W_encoder = np.random.randn(hidden_size, hidden_size) * 0.01  # W2

        # v vector (project to scalar)
        self.v = np.random.randn(1, hidden_size) * 0.01

    def compute_scores(self, decoder_hidden: np.ndarray,
                       encoder_hiddens: List[np.ndarray]) -> np.ndarray:
        """
        Compute raw scores for each encoder hidden state.

        Args:
            decoder_hidden: (hidden_size, 1)
            encoder_hiddens: list of (hidden_size, 1), length = src_len

        Returns:
            scores: (src_len,) 1-D numpy array of floats
        """
        decoder_proj = self.W_decoder @ decoder_hidden  # (hidden_size, 1)
        scores = np.zeros(len(encoder_hiddens), dtype=float)

        for i, h_enc in enumerate(encoder_hiddens):
            enc_proj = self.W_encoder @ h_enc  # (hidden_size, 1)
            combined = np.tanh(decoder_proj + enc_proj)  # (hidden_size, 1)
            score = (self.v @ combined)[0, 0]  # scalar
            scores[i] = score
        return scores

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax for 1-D array -> returns 1-D array."""
        x = np.array(x, dtype=float)
        if x.size == 0:
            return x
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def forward(self, decoder_hidden: np.ndarray,
                encoder_hiddens: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute context vector and normalized attention weights.

        Returns:
            context_vector: (hidden_size, 1)
            attention_weights: (src_len,) 1-D numpy array
        """
        if len(encoder_hiddens) == 0:
            # empty encoder case (defensive)
            return np.zeros((self.hidden_size, 1)), np.array([])

        scores = self.compute_scores(decoder_hidden, encoder_hiddens)  # (src_len,)
        attn_weights = self.softmax(scores)  # (src_len,)

        # Context: weighted sum of encoder hidden states
        context = np.zeros((self.hidden_size, 1))
        for w, h_enc in zip(attn_weights, encoder_hiddens):
            context += w * h_enc  # (hidden_size, 1)

        return context, attn_weights


# ==================== ATTENTION DECODER ====================

class AttentionDecoder:
    """
    LSTM-based decoder with Bahdanau attention.

    Important returns:
      - forward_step(...) -> logits, h_next, c_next, attention_weights (1-D np.array)
      - forward(...) -> all_logits (list), all_attention (list of 1-D np.arrays)
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, seed: int = 42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Attention
        self.attention = BahdanauAttention(hidden_size, seed)

        # LSTM cell input is [embedding; context] => embedding_dim + hidden_size
        self.lstm_cell = LSTMCell(embedding_dim + hidden_size, hidden_size, seed)

        # Output layer: combine decoder hidden and context (hidden_size * 2) -> vocab
        self.W_out = np.random.randn(vocab_size, hidden_size * 2) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros((vocab_size, 1))

    def forward_step(self, token_idx: int, h: np.ndarray, c: np.ndarray,
                     encoder_hiddens: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Single decoding step with attention.

        Args:
            token_idx: integer index of previous token
            h: (hidden_size, 1) decoder hidden
            c: (hidden_size, 1) decoder cell
            encoder_hiddens: list of (hidden_size, 1)

        Returns:
            logits: (vocab_size, 1)
            h_next: (hidden_size, 1)
            c_next: (hidden_size, 1)
            attention_weights: (src_len,) 1-D numpy array
        """
        # Embedding
        x = self.embeddings[token_idx].reshape(-1, 1)  # (embedding_dim, 1)

        # Attention
        context_vector, attention_weights = self.attention.forward(h, encoder_hiddens)  # context (hidden_size,1), attn (src_len,)

        # LSTM input: concat embedding and context
        lstm_input = np.vstack([x, context_vector])  # (embedding_dim + hidden_size, 1)
        h_next, c_next = self.lstm_cell.forward(lstm_input, h, c)

        # Output projection: combine h_next and context
        combined = np.vstack([h_next, context_vector])  # (hidden_size*2, 1)
        logits = self.W_out @ combined + self.b_out  # (vocab_size, 1)

        # Ensure attention_weights is a 1-D numpy array
        attention_weights = np.array(attention_weights, dtype=float).flatten()

        return logits, h_next, c_next, attention_weights

    def forward(self, target_indices: List[int], encoder_hidden: np.ndarray,
                encoder_cell: np.ndarray, encoder_hiddens: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Full forward (training) pass with teacher forcing.

        Args:
            target_indices: list with <SOS> at index 0
            encoder_hidden: (hidden_size, 1) initial decoder hidden (usually encoder final)
            encoder_cell: (hidden_size, 1)
            encoder_hiddens: list of encoder hidden states (each (hidden_size,1))

        Returns:
            all_logits: list of logits per step (each (vocab_size,1))
            all_attention: list of attention weight arrays (1-D)
        """
        h = encoder_hidden
        c = encoder_cell
        all_logits = []
        all_attention = []

        # teacher forcing: iterate over input tokens except final token
        for t in range(len(target_indices) - 1):
            token_idx = target_indices[t]
            logits, h, c, attn = self.forward_step(token_idx, h, c, encoder_hiddens)
            all_logits.append(logits)
            all_attention.append(attn)  # 1-D numpy array

        return all_logits, all_attention
# ==================== SEQ2SEQ WITH ATTENTION ====================

class Seq2SeqWithAttention:
    """Complete sequence-to-sequence model with attention."""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 embedding_dim: int, hidden_size: int, learning_rate: float = 0.01):
        """Initialize Seq2Seq model with attention."""
        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_size)
        self.decoder = AttentionDecoder(tgt_vocab_size, embedding_dim, hidden_size)
        
        self.learning_rate = learning_rate
        self.loss_history = []
        
    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def compute_loss(self, logits_list: List[np.ndarray], target_indices: List[int]) -> float:
        """Compute cross-entropy loss."""
        loss = 0
        for logits, target_idx in zip(logits_list, target_indices[1:]):
            probs = self.softmax(logits)
            loss += -np.log(probs[target_idx, 0] + 1e-10)
        return loss / len(logits_list)
    
    def train_step(self, src_indices: List[int], tgt_indices: List[int]) -> float:
        """Single training step."""
        # Encode source
        enc_hidden, enc_cell, enc_hiddens = self.encoder.forward(src_indices)
        
        # Decode target with attention
        logits_list, _ = self.decoder.forward(tgt_indices, enc_hidden, enc_cell, enc_hiddens)
        
        # Compute loss
        loss = self.compute_loss(logits_list, tgt_indices)
        
        # Simplified gradient update
        for logits, target_idx in zip(logits_list, tgt_indices[1:]):
            probs = self.softmax(logits)
            dlogits = probs.copy()
            dlogits[target_idx, 0] -= 1
            dlogits = np.clip(dlogits, -5, 5)
            
            # Update decoder output layer
            self.decoder.W_out -= self.learning_rate * dlogits * 0.001
            self.decoder.b_out -= self.learning_rate * dlogits * 0.001
        
        self.loss_history.append(loss)
        return loss
    
    def translate(self, src_indices: List[int], tgt_vocab: Vocabulary,
                  max_length: int = 20) -> Tuple[List[int], List[np.ndarray]]:
        """
        Translate source sequence with attention tracking.
        
        Returns:
            predicted_indices: Predicted target sequence
            attention_weights_list: Attention weights for each step
        """
        # Encode source
        enc_hidden, enc_cell, enc_hiddens = self.encoder.forward(src_indices)
        
        # Initialize decoder
        h = enc_hidden
        c = enc_cell
        predicted_indices = [tgt_vocab.SOS_token]
        attention_weights_list = []
        
        # Generate tokens
        for _ in range(max_length):
            logits, h, c, attn_weights = self.decoder.forward_step(
                predicted_indices[-1], h, c, enc_hiddens
            )
            
            attention_weights_list.append(attn_weights)
            
            # Get next token (greedy)
            next_token = np.argmax(logits)
            predicted_indices.append(next_token)
            
            if next_token == tgt_vocab.EOS_token:
                break
        
        return predicted_indices, attention_weights_list
#==================== VISUALIZATION ====================

def visualize_attention(src_sentence: str, tgt_sentence: str, pred_sentence: str,
                       attention_weights: List[np.ndarray], 
                       src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                       save_name: str = "attention_weights.png"):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        src_sentence: Source sentence
        tgt_sentence: Target sentence
        pred_sentence: Predicted sentence
        attention_weights: List of attention weight arrays
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
    """
    # Tokenize
    src_tokens = src_vocab.tokenize(src_sentence)
    tgt_tokens = tgt_vocab.tokenize(pred_sentence)
    
    # --- Handle attention weights safely ---
    if not attention_weights or len(attention_weights) == 0:
        print("⚠ No attention weights available to visualize.")
        return
    
    # Handle variable-length or ragged attention arrays
    max_src_len = len(src_tokens)
    max_tgt_len = len(tgt_tokens)
    matrix = np.zeros((max_tgt_len, max_src_len))
    
    for i, row in enumerate(attention_weights[:max_tgt_len]):
        row = np.array(row).flatten()
        matrix[i, :len(row)] = row[:max_src_len]

    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, 
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                linewidths=0.5,
                linecolor='gray')
    
    plt.xlabel('Source Words (English)', fontsize=12, fontweight='bold')
    plt.ylabel('Target Words (French)', fontsize=12, fontweight='bold')
    plt.title(f'Attention Weights Visualization\n\nSource: "{src_sentence}"\nTarget: "{tgt_sentence}"\nPredicted: "{pred_sentence}"',
              fontsize=13, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✓ Attention visualization saved as '{save_name}'")
    plt.show()


def visualize_training_comparison(model_with_attn: Seq2SeqWithAttention,
                                  src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                                  test_pairs: List[Tuple[str, str]]):
    """Comprehensive visualization of model with attention."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Plot 1: Loss curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(model_with_attn.loss_history, linewidth=2, color='#10b981', alpha=0.8, label='With Attention')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss (Seq2Seq WITH Attention)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Translation examples
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    
    example_text = "📝 Translation Examples\n" + "="*55 + "\n\n"
    for i, (src, tgt) in enumerate(test_pairs[:6], 1):
        src_indices = src_vocab.encode(src)
        predicted_indices, _ = model_with_attn.translate(src_indices, tgt_vocab)
        predicted = tgt_vocab.decode(predicted_indices)
        
        example_text += f"{i}. EN: {src}\n"
        example_text += f"   FR (true): {tgt}\n"
        example_text += f"   FR (pred): {predicted}\n\n"
    
    ax2.text(0.05, 0.95, example_text, fontsize=8, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Plot 3: Attention mechanism explanation
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    mech_text = """
    🔍 How Attention Works
    
    STEP 1: Compute Scores
    For each encoder hidden state h_enc[i]:
      score[i] = v^T * tanh(W1*h_dec + W2*h_enc[i])
    
    STEP 2: Normalize with Softmax
      attention_weights = softmax(scores)
    
    STEP 3: Compute Context Vector
      context = Σ attention_weights[i] * h_enc[i]
    
    STEP 4: Use Context in Decoder
      decoder_input = [embedding; context]
    
    ✨ KEY BENEFIT:
    Decoder can focus on RELEVANT parts
    of source, not just final encoder state!
    
    High attention weight = model is 
    "looking at" that source word
    """
    
    ax3.text(0.5, 0.5, mech_text, fontsize=9, family='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4))
    
    # Plot 4: Sample attention heatmap
    ax4 = fig.add_subplot(gs[2, :])
    src, tgt = test_pairs[0]
    src_indices = src_vocab.encode(src)
    predicted_indices, attention_weights = model_with_attn.translate(src_indices, tgt_vocab)
    
    src_tokens = src_vocab.tokenize(src)
    pred_tokens = [tgt_vocab.idx2token[idx] for idx in predicted_indices[1:-1]]  # Skip SOS/EOS
    
    if attention_weights:
        # --- Safe shape handling ---
        max_src_len = len(src_tokens)
        max_tgt_len = len(pred_tokens)
        matrix = np.zeros((max_tgt_len, max_src_len))
        for i, row in enumerate(attention_weights[:max_tgt_len]):
            row = np.array(row).flatten()
            matrix[i, :len(row)] = row[:max_src_len]

        sns.heatmap(matrix,
                    xticklabels=src_tokens,
                    yticklabels=pred_tokens,
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Attention Weight'},
                    ax=ax4,
                    linewidths=0.5,
                    linecolor='gray')
        
        ax4.set_xlabel('Source Words', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Generated Words', fontsize=11, fontweight='bold')
        ax4.set_title(f'Sample Attention Weights: \"{src}\" → \"{tgt}\"', 
                     fontsize=12, fontweight='bold')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.savefig('attention_mechanism_results.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive visualization saved as 'attention_mechanism_results.png'")
    plt.show()
# ==================== MAIN DEMONSTRATION ====================

def main():
    """Main demonstration of attention mechanism."""
    
    print("=" * 70)
    print("ATTENTION MECHANISM FOR NEURAL MACHINE TRANSLATION")
    print("=" * 70)
    
    # Configuration
    embedding_dim = 32
    hidden_size = 64
    learning_rate = 0.01
    epochs = 600
    
    print("\n📋 Configuration:")
    print(f"   Task: English → French (with ATTENTION)")
    print(f"   Embedding Dimension: {embedding_dim}")
    print(f"   Hidden Size: {hidden_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    
    # Load data
    print("\n📚 Loading translation pairs...")
    pairs = load_translation_pairs()
    print(f"   Total pairs: {len(pairs)}")
    
    # Build vocabularies
    print("\n🔤 Building vocabularies...")
    src_vocab = Vocabulary('english')
    tgt_vocab = Vocabulary('french')
    
    for src, tgt in pairs:
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)
    
    print(f"   Source vocab size: {src_vocab.n_tokens}")
    print(f"   Target vocab size: {tgt_vocab.n_tokens}")
    
    # Initialize model with attention
    print("\n🧠 Initializing Seq2Seq model WITH ATTENTION...")
    model = Seq2SeqWithAttention(
        src_vocab_size=src_vocab.n_tokens,
        tgt_vocab_size=tgt_vocab.n_tokens,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        learning_rate=learning_rate
    )
    
    # Training
    print("\n🚀 Training model with attention...")
    print("-" * 70)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for src, tgt in pairs:
            src_indices = src_vocab.encode(src)
            tgt_indices = [tgt_vocab.SOS_token] + tgt_vocab.encode(tgt) + [tgt_vocab.EOS_token]
            
            loss = model.train_step(src_indices, tgt_indices)
            total_loss += loss
        
        avg_loss = total_loss / len(pairs)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | Avg Loss: {avg_loss:.4f}")
    
    print("-" * 70)
    print("✓ Training complete!")
    
    # Evaluation with attention visualization
    print("\n📝 Translation Examples with Attention Weights:")
    print("=" * 70)
    
    test_pairs = pairs[:5]
    for i, (src, tgt) in enumerate(test_pairs):
        src_indices = src_vocab.encode(src)
        predicted_indices, attention_weights = model.translate(src_indices, tgt_vocab)
        predicted = tgt_vocab.decode(predicted_indices)
        
        print(f"\n{i+1}. Source (EN):    {src}")
        print(f"   Target (FR):    {tgt}")
        print(f"   Predicted (FR): {predicted}")
        print(f"   Attention shape: {len(attention_weights)} steps × {len(attention_weights[0])} source tokens")
        print("-" * 70)
        
        # Visualize attention for first example
        if i == 0:
            print("\n📊 Generating detailed attention visualization for first example...")
            visualize_attention(src, tgt, predicted, attention_weights, 
                              src_vocab, tgt_vocab, "detailed_attention.png")
    
    # Comprehensive visualization
    print("\n📊 Generating comprehensive visualization...")
    visualize_training_comparison(model, src_vocab, tgt_vocab, test_pairs)
    
    print("\n" + "=" * 70)
    print("KEY CONCEPTS:")
    print("-" * 70)
    print("1. Attention: Decoder focuses on relevant source words at each step")
    print("2. Context Vector: Weighted sum of encoder hiddens (not just last)")
    print("3. Alignment: Attention weights show word-to-word correspondences")
    print("4. Bottleneck Solved: All source info accessible, not compressed")
    print("5. Interpretability: Can visualize what model pays attention to!")
    print("=" * 70)
    print("\nDEMONSTRATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()