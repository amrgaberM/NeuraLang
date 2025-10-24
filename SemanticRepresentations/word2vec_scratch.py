import numpy as np
from collections import Counter

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


class Word2Vec:
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


# Example usage with comprehensive dataset
if __name__ == "__main__":
    # Rich, diverse dataset for better learning
    sentences = [
        # Animals and pets
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "log"],
        ["cats", "and", "dogs", "are", "animals"],
        ["the", "cat", "and", "dog", "played", "together"],
        ["animals", "like", "cats", "are", "cute", "pets"],
        ["dogs", "are", "loyal", "animals", "and", "pets"],
        ["the", "cute", "cat", "is", "a", "pet"],
        ["the", "loyal", "dog", "is", "a", "pet"],
        
        # Locations and objects
        ["the", "mat", "is", "on", "the", "floor"],
        ["the", "floor", "is", "very", "clean"],
        ["the", "log", "is", "on", "the", "ground"],
        ["ground", "and", "floor", "are", "surfaces"],
        
        # Actions and behaviors
        ["the", "cat", "played", "with", "a", "toy"],
        ["the", "dog", "played", "in", "the", "yard"],
        ["they", "sat", "and", "played", "together"],
        ["animals", "played", "on", "the", "ground"],
        
        # Descriptions
        ["the", "cute", "cat", "sat", "quietly"],
        ["the", "loyal", "dog", "sat", "nearby"],
        ["the", "clean", "mat", "and", "floor"],
        ["very", "cute", "pets", "are", "animals"],
        
        # More context for better embeddings
        ["cats", "are", "cute", "and", "quiet"],
        ["dogs", "are", "loyal", "and", "playful"],
        ["the", "yard", "has", "green", "grass"],
        ["pets", "like", "dogs", "and", "cats"],
        ["the", "toy", "is", "on", "the", "mat"],
    ]
    
    print("="*60)
    print("ENHANCED SKIP-GRAM WORD2VEC TRAINING")
    print("="*60)
    print()
    
    # Train model with optimal hyperparameters
    w2v = Word2Vec(
        sentences, 
        window_size=2,              # Consider 2 words on each side
        embedding_dim=30,           # Good balance for this dataset size
        learning_rate=0.025,        # Standard Word2Vec learning rate
        epochs=200,                 # Will early stop if needed
        min_count=1,                # Keep all words
        lr_decay=0.98,              # 2% decay per epoch
        early_stopping_patience=20  # Stop if 20 epochs no improvement
    )
    
    # Train and get loss history
    loss_history = w2v.train(verbose=True)
    
    print("\n" + "="*60)
    print("SEMANTIC SIMILARITY TESTING")
    print("="*60)
    
    # Test similarity for multiple words
    test_words = ["cat", "dog", "mat", "floor", "played", "cute"]
    for word in test_words:
        if word in w2v.word2idx:
            print(f"\n Most similar to '{word}':")
            similar = w2v.most_similar(word, top_n=4)
            for similar_word, sim in similar:
                print(f"   {similar_word:12s} → {sim:.4f}")
    
    # Test word analogies
    print("\n" + "="*60)
    print("WORD ANALOGY TESTING")
    print("="*60)
    
    analogy_tests = [
        ("cat", "cats", "dog"),      # cat:cats :: dog:?
        ("mat", "floor", "log"),     # mat:floor :: log:?
    ]
    
    for word_a, word_b, word_c in analogy_tests:
        result = w2v.analogy(word_a, word_b, word_c, top_n=3)
        if result:
            print(f"\n {word_a}:{word_b} :: {word_c}:?")
            for word, score in result:
                print(f"   {word:12s} → {score:.4f}")
    
    # Show sample embeddings
    print("\n" + "="*60)
    print("SAMPLE WORD EMBEDDINGS")
    print("="*60)
    for word in ["cat", "dog", "pets", "cute"]:
        if word in w2v.word2idx:
            vec = w2v.get_vector(word)
            print(f"\n'{word}' embedding (first 8 dimensions):")
            print(f"  {vec[:8]}")
    
    # Training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"✓ Vocabulary size:     {w2v.vocab_size} words")
    print(f"✓ Embedding dimension: {w2v.embedding_dim}")
    print(f"✓ Epochs completed:    {len(loss_history)}")
    print(f"✓ Initial loss:        {loss_history[0]:.4f}")
    print(f"✓ Final loss:          {loss_history[-1]:.4f}")
    print(f"✓ Total improvement:   {loss_history[0] - loss_history[-1]:.4f}")
    print(f"✓ Improvement rate:    {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.1f}%")
    print("="*60)