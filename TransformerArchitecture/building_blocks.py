"""
Transformer Building Blocks
Complete implementation of core transformer components for educational purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention is All You Need"
    
    Args:
        d_model: Dimension of the model (embedding size)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention
        
        Args:
            Q: Query tensor (batch, num_heads, seq_len, d_k)
            K: Key tensor (batch, num_heads, seq_len, d_k)
            V: Value tensor (batch, num_heads, seq_len, d_k)
            mask: Optional mask tensor
            
        Returns:
            attention_output: Attention-weighted values
            attention_weights: Attention distribution
        """
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass through multi-head attention
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: Multi-head attention output (batch, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections and split into multiple heads
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attention_output)
        
        return output


class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network
    Applies two linear transformations with a ReLU activation in between
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically 4 * d_model)
        dropout: Dropout probability
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass: FFN(x) = max(0, xW1 + b1)W2 + b2
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            output: Transformed tensor (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Complete Transformer Encoder Block
    Combines multi-head attention, feed-forward network, layer normalization, and residual connections
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass through transformer block
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Transformed tensor (batch, seq_len, d_model)
        """
        # Multi-head attention with residual connection and layer norm
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attention_output))
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions
    Adds position information to token embeddings
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            
        Returns:
            output: Embeddings with positional information
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("=" * 60)
    print("Testing Transformer Building Blocks")
    print("=" * 60)
    
    # Test Multi-Head Attention
    print("\n1. Multi-Head Attention")
    mha = MultiHeadAttention(d_model, num_heads)
    attention_output = mha(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {attention_output.shape}")
    
    # Test Position-Wise FFN
    print("\n2. Position-Wise Feed-Forward Network")
    ffn = PositionWiseFFN(d_model, d_ff)
    ffn_output = ffn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {ffn_output.shape}")
    
    # Test Transformer Block
    print("\n3. Complete Transformer Block")
    transformer_block = TransformerBlock(d_model, num_heads, d_ff)
    block_output = transformer_block(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {block_output.shape}")
    
    # Test Positional Encoding
    print("\n4. Positional Encoding")
    pos_encoding = PositionalEncoding(d_model)
    encoded_output = pos_encoding(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {encoded_output.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)