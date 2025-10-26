import numpy as np
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class SkipGram:
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.025):
        """
        Skip-gram model for word2vec
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            learning_rate: Learning rate for training
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        
        # Initialize weight matrices with Xavier initialization
        limit = np.sqrt(6.0 / (vocab_size + embedding_dim))
        self.W1 = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-limit, limit, (embedding_dim, vocab_size))
        
    def softmax(self, x):
        """Compute softmax with numerical stability"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def forward(self, center_word_idx):
        """
        Forward pass
        
        Args:
            center_word_idx: Index of center word
            
        Returns:
            h: Hidden layer (word embedding)
            u: Output layer scores
            y_pred: Predicted probabilities
        """
        # Hidden layer: get embedding for center word
        h = self.W1[center_word_idx]
        
        # Output layer
        u = np.dot(h, self.W2)
        y_pred = self.softmax(u)
        
        return h, u, y_pred
    
    def backward(self, center_word_idx, context_word_idx, h, y_pred):
        """
        Backward pass and weight update
        
        Args:
            center_word_idx: Index of center word
            context_word_idx: Index of context word
            h: Hidden layer activation
            y_pred: Predicted probabilities
        """
        # Create one-hot encoded target
        y_true = np.zeros(self.vocab_size)
        y_true[context_word_idx] = 1
        
        # Compute error (gradient of cross-entropy loss)
        e = y_pred - y_true
        
        # Update W2 (hidden to output)
        dW2 = np.outer(h, e)
        self.W2 -= self.lr * dW2
        
        # Update W1 (input to hidden) - only for center word
        dh = np.dot(self.W2, e)
        self.W1[center_word_idx] -= self.lr * dh
    
    def train_step(self, center_word_idx, context_word_idx):
        """Single training step"""
        h, u, y_pred = self.forward(center_word_idx)
        self.backward(center_word_idx, context_word_idx, h, y_pred)
        
        # Calculate loss (cross-entropy)
        loss = -np.log(y_pred[context_word_idx] + 1e-10)
        return loss
    
    def get_embedding(self, word_idx):
        """Get embedding vector for a word"""
        return self.W1[word_idx]

class Word2VecModel:
    def __init__(self, sentences, window_size=2, embedding_dim=50, 
                 learning_rate=0.025, epochs=200, min_count=1, 
                 lr_decay=0.98, early_stopping_patience=20):
        """
        Word2Vec trainer using Skip-gram with enhancements
        
        Args:
            sentences: List of sentences (each sentence is a list of words)
            window_size: Context window size
            embedding_dim: Dimension of embeddings
            learning_rate: Initial learning rate
            epochs: Maximum number of training epochs
            min_count: Minimum word frequency to include in vocabulary
            lr_decay: Learning rate decay factor per epoch (0.98 = 2% decay)
            early_stopping_patience: Stop if no improvement for this many epochs
        """
        self.sentences = sentences
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.initial_lr = learning_rate
        self.epochs = epochs
        self.min_count = min_count
        self.lr_decay = lr_decay
        self.patience = early_stopping_patience
        
        # Build vocabulary
        self.build_vocab()
        
        # Initialize model
        self.model = SkipGram(len(self.word2idx), embedding_dim, learning_rate)
        
    def build_vocab(self):
        """Build vocabulary from sentences"""
        word_counts = Counter()
        for sentence in self.sentences:
            word_counts.update(sentence)
        
        # Filter by min_count
        vocab = [word for word, count in word_counts.items() if count >= self.min_count]
        
        # Create mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def generate_training_data(self):
        """Generate (center_word, context_word) pairs"""
        training_data = []
        
        for sentence in self.sentences:
            # Convert words to indices
            indices = [self.word2idx[word] for word in sentence if word in self.word2idx]
            
            # Generate pairs
            for i, center_idx in enumerate(indices):
                # Get context words within window
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_idx = indices[j]
                        training_data.append((center_idx, context_idx))
        
        return training_data
    
    def train(self, verbose=True):
        """Train the model with learning rate decay and early stopping"""
        training_data = self.generate_training_data()
        if verbose:
            print(f"Training pairs: {len(training_data)}")
            print(f"Starting training...\n")
        
        best_loss = float('inf')
        no_improve_count = 0
        loss_history = []
        
        for epoch in range(self.epochs):
            # Shuffle training data each epoch
            np.random.shuffle(training_data)
            
            total_loss = 0
            for center_idx, context_idx in training_data:
                loss = self.model.train_step(center_idx, context_idx)
                total_loss += loss
            
            avg_loss = total_loss / len(training_data)
            loss_history.append(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Learning rate decay
            self.model.lr *= self.lr_decay
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} | Loss: {avg_loss:.4f} | LR: {self.model.lr:.6f}")
            
            # Early stopping
            if no_improve_count >= self.patience:
                if verbose:
                    print(f"\n⚠ Early stopping at epoch {epoch + 1}")
                    print(f"  No improvement for {self.patience} epochs")
                    print(f"  Best loss: {best_loss:.4f}")
                break
        
        # Final summary
        if verbose:
            if no_improve_count < self.patience:
                print(f"\n✓ Training completed all {self.epochs} epochs")
            print(f"  Final loss: {avg_loss:.4f}")
            print(f"  Total improvement: {loss_history[0] - loss_history[-1]:.4f}")
        
        return loss_history
    
    def get_vector(self, word):
        """Get word vector"""
        if word in self.word2idx:
            return self.model.get_embedding(self.word2idx[word])
        return None
    
    def most_similar(self, word, top_n=5):
        """Find most similar words using cosine similarity"""
        if word not in self.word2idx:
            return []
        
        word_vec = self.get_vector(word)
        
        similarities = []
        for other_word in self.word2idx:
            if other_word != word:
                other_vec = self.get_vector(other_word)
                # Cosine similarity
                cos_sim = np.dot(word_vec, other_vec) / (
                    np.linalg.norm(word_vec) * np.linalg.norm(other_vec) + 1e-10
                )
                similarities.append((other_word, cos_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def analogy(self, word_a, word_b, word_c, top_n=1):
        """
        Solve word analogies: word_a is to word_b as word_c is to ?
        Example: king - man + woman = queen
        """
        if not all(w in self.word2idx for w in [word_a, word_b, word_c]):
            return []
        
        vec_a = self.get_vector(word_a)
        vec_b = self.get_vector(word_b)
        vec_c = self.get_vector(word_c)
        
        # Compute: vec_b - vec_a + vec_c
        target_vec = vec_b - vec_a + vec_c
        
        similarities = []
        for word in self.word2idx:
            if word not in [word_a, word_b, word_c]:
                vec = self.get_vector(word)
                cos_sim = np.dot(target_vec, vec) / (
                    np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10
                )
                similarities.append((word, cos_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

class GloVeModel:
    def __init__(self, cooccurrence_matrix, vocab, word2idx, 
                 embedding_dim=100, x_max=100, alpha=0.75):
        """
        GloVe: Global Vectors for Word Representation
        
        Args:
            cooccurrence_matrix: numpy array of co-occurrence counts [vocab_size, vocab_size]
            vocab: list of vocabulary words
            word2idx: dictionary mapping words to indices
            embedding_dim: dimension of word embeddings
            x_max: cutoff for weighting function
            alpha: exponent for weighting function
        """
        self.matrix = cooccurrence_matrix
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha
        
        # Initialize embeddings with small random values
        scale = 0.5
        self.W = np.random.uniform(-scale, scale, (self.vocab_size, embedding_dim))
        self.W_context = np.random.uniform(-scale, scale, (self.vocab_size, embedding_dim))
        
        # Initialize biases
        self.b = np.random.uniform(-scale, scale, self.vocab_size)
        self.b_context = np.random.uniform(-scale, scale, self.vocab_size)
        
        print(f"✓ Initialized GloVe model:")
        print(f"  - Vocabulary size: {self.vocab_size}")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - x_max: {x_max}, alpha: {alpha}")
    
    def weighting_function(self, X_ij):
        """Weighting function to prevent very frequent co-occurrences from dominating"""
        if X_ij < self.x_max:
            return (X_ij / self.x_max) ** self.alpha
        else:
            return 1.0
    
    def compute_cost(self, i, j, X_ij):
        """Compute weighted squared error for single co-occurrence"""
        prediction = np.dot(self.W[i], self.W_context[j]) + self.b[i] + self.b_context[j]
        target = np.log(X_ij + 1e-10)
        weight = self.weighting_function(X_ij)
        cost = weight * (prediction - target) ** 2
        return cost
    
    def update_parameters(self, i, j, X_ij, learning_rate):
        """Compute gradients and update parameters using gradient descent"""
        prediction = np.dot(self.W[i], self.W_context[j]) + self.b[i] + self.b_context[j]
        target = np.log(X_ij + 1e-10)
        weight = self.weighting_function(X_ij)
        diff = weight * (prediction - target)
        
        # Compute gradients
        grad_W_i = diff * self.W_context[j]
        grad_W_context_j = diff * self.W[i]
        grad_b_i = diff
        grad_b_context_j = diff
        
        # Update parameters
        self.W[i] -= learning_rate * grad_W_i
        self.W_context[j] -= learning_rate * grad_W_context_j
        self.b[i] -= learning_rate * grad_b_i
        self.b_context[j] -= learning_rate * grad_b_context_j
    
    def train(self, epochs=50, learning_rate=0.05, lr_decay=0.99, verbose=True):
        """Train GloVe embeddings"""
        # Extract non-zero co-occurrences
        nonzero_entries = []
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                if self.matrix[i, j] > 0:
                    nonzero_entries.append((i, j, self.matrix[i, j]))
        
        n_entries = len(nonzero_entries)
        
        if n_entries == 0:
            print("❌ Error: No co-occurrences found! Check your data.")
            return []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING GLOVE")
            print(f"{'='*60}")
            print(f"Non-zero co-occurrences: {n_entries}")
            print(f"Starting training...\n")
        
        cost_history = []
        
        for epoch in range(epochs):
            total_cost = 0
            np.random.shuffle(nonzero_entries)
            
            for i, j, X_ij in nonzero_entries:
                cost = self.compute_cost(i, j, X_ij)
                total_cost += cost
                self.update_parameters(i, j, X_ij, learning_rate)
            
            avg_cost = total_cost / n_entries
            cost_history.append(avg_cost)
            learning_rate *= lr_decay
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} | Cost: {avg_cost:.6f} | LR: {learning_rate:.6f}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✓ Training complete!")
            print(f"  - Initial cost: {cost_history[0]:.6f}")
            print(f"  - Final cost: {cost_history[-1]:.6f}")
            print(f"  - Improvement: {cost_history[0] - cost_history[-1]:.6f}")
            print(f"{'='*60}\n")
        
        return cost_history
    
    def get_embeddings(self):
        """Get final word embeddings (average of W and W_context)"""
        return (self.W + self.W_context) / 2
    
    def get_vector(self, word):
        """Get embedding vector for a specific word"""
        if word in self.word2idx:
            idx = self.word2idx[word]
            embeddings = self.get_embeddings()
            return embeddings[idx]
        return None
    
    def most_similar(self, word, top_n=5):
        """Find most similar words using cosine similarity"""
        if word not in self.word2idx:
            print(f"⚠ Warning: '{word}' not in vocabulary")
            return []
        
        word_vec = self.get_vector(word)
        embeddings = self.get_embeddings()
        
        similarities = []
        for other_word in self.vocab:
            if other_word != word:
                other_vec = self.get_vector(other_word)
                cos_sim = np.dot(word_vec, other_vec) / (
                    np.linalg.norm(word_vec) * np.linalg.norm(other_vec) + 1e-10
                )
                similarities.append((other_word, cos_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def analogy(self, word_a, word_b, word_c, top_n=3):
        """Solve word analogies: word_a is to word_b as word_c is to ?"""
        missing_words = [w for w in [word_a, word_b, word_c] if w not in self.word2idx]
        if missing_words:
            print(f"⚠ Warning: Words not in vocabulary: {missing_words}")
            return []
        
        vec_a = self.get_vector(word_a)
        vec_b = self.get_vector(word_b)
        vec_c = self.get_vector(word_c)
        
        target_vec = vec_b - vec_a + vec_c
        
        similarities = []
        for word in self.vocab:
            if word not in [word_a, word_b, word_c]:
                vec = self.get_vector(word)
                cos_sim = np.dot(target_vec, vec) / (
                    np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10
                )
                similarities.append((word, cos_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def visualize_embeddings(self, words=None, method='pca'):
        """
        Visualize embeddings in 2D using PCA or t-SNE
        
        Args:
            words: List of words to visualize (None = all words)
            method: 'pca' or 'tsne'
        """
        embeddings = self.get_embeddings()
        
        if words is None:
            words = self.vocab[:50]  # Limit to 50 words for clarity
        
        indices = [self.word2idx[w] for w in words if w in self.word2idx]
        word_vecs = embeddings[indices]
        word_labels = [self.vocab[i] for i in indices]
        
        # Dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(word_vecs)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            coords = tsne.fit_transform(word_vecs)
        else:
            raise ValueError("method must be 'pca' or 'tsne'")
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
        
        for i, word in enumerate(word_labels):
            plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)
        
        plt.title(f'GloVe Embeddings Visualization ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def train_glove_from_scratch(corpus, window_size=2, embedding_dim=100, 
                            epochs=100, verbose=True):
    """
    Complete pipeline: data → co-occurrence → GloVe training
    
    Args:
        corpus: List of tokenized sentences (list of lists) OR list of strings
        window_size: Context window size
        embedding_dim: Embedding dimension
        epochs: Training epochs
        verbose: Print progress
    
    Returns:
        glove: Trained GloVe model
        matrix: Co-occurrence matrix
        vocab: Vocabulary list
    """
    # Import your utility
    try:
        import sys
        sys.path.append('..')
        from utils.data_preprocessing import build_cooccurrence_matrix
    except ImportError:
        print("❌ Error: Cannot import build_cooccurrence_matrix from utils")
        print("   Make sure utils/data_preprocessing.py exists")
        return None, None, None
    
    # Handle different input formats
    if isinstance(corpus, str):
        corpus = [corpus]
    
    # Check if already tokenized (list of lists)
    if corpus and isinstance(corpus[0], list):
        corpus_strings = [" ".join(sent) for sent in corpus]
    else:
        corpus_strings = corpus
    
    print(f"{'='*60}")
    print(f"GLOVE TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"📊 Dataset Info:")
    print(f"  - Number of sentences: {len(corpus_strings)}")
    print(f"  - Sample: '{corpus_strings[0][:50]}...'")
    
    # Build co-occurrence matrix
    print(f"\n🔨 Building co-occurrence matrix...")
    matrix, vocab, word2idx = build_cooccurrence_matrix(
        corpus=corpus_strings,
        window_size=window_size,
        distance_weighting=True,
        min_count=1
    )
    
    print(f"✓ Matrix shape: {matrix.shape}")
    print(f"✓ Vocabulary size: {len(vocab)}")
    print(f"✓ Non-zero entries: {np.count_nonzero(matrix)}")
    print(f"✓ Sample vocabulary: {vocab[:10]}")
    
    # Initialize and train GloVe
    glove = GloVe(
        cooccurrence_matrix=matrix,
        vocab=vocab,
        word2idx=word2idx,
        embedding_dim=embedding_dim,
        x_max=100,
        alpha=0.75
    )
    
    cost_history = glove.train(
        epochs=epochs,
        learning_rate=0.05,
        lr_decay=0.99,
        verbose=verbose
    )
    
    # Evaluation
    print(f"{'='*60}")
    print(f"EVALUATION")
    print(f"{'='*60}")
    
    # Test with first few words from vocabulary
    test_words = vocab[:min(5, len(vocab))]
    print(f"\n📝 Testing similarity with: {test_words}\n")
    
    for word in test_words:
        print(f"Most similar to '{word}':")
        similar = glove.most_similar(word, top_n=3)
        if similar:
            for sim_word, score in similar:
                print(f"  {sim_word:15s} {score:.4f}")
        print()
    
    # Show sample embeddings
    print(f"\n🔢 Sample embeddings (first 5 dimensions):")
    for word in vocab[:3]:
        vec = glove.get_vector(word)
        print(f"  '{word}': {vec[:5]}")
    
    return glove, matrix, vocab
