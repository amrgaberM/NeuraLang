import numpy as np

class GloVe:
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
        
        print(f"Initialized GloVe model:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  x_max: {x_max}, alpha: {alpha}")
    
    def weighting_function(self, X_ij):
        """
        Weighting function to prevent very frequent co-occurrences from dominating
        
        f(X) = (X/x_max)^alpha  if X < x_max
               1                 otherwise
        
        Args:
            X_ij: Co-occurrence count
        
        Returns:
            weight: Value between 0 and 1
        """
        if X_ij < self.x_max:
            return (X_ij / self.x_max) ** self.alpha
        else:
            return 1.0
    
    def compute_cost(self, i, j, X_ij):
        """
        Compute weighted squared error for single co-occurrence
        
        Cost = f(X_ij) * (w_i·w_j + b_i + b_j - log(X_ij))²
        
        Args:
            i: Index of target word
            j: Index of context word
            X_ij: Co-occurrence count
        
        Returns:
            cost: Weighted squared error
        """
        # Dot product of embeddings + biases
        prediction = np.dot(self.W[i], self.W_context[j]) + self.b[i] + self.b_context[j]
        
        # Log of co-occurrence (add small epsilon for stability)
        target = np.log(X_ij + 1e-10)
        
        # Weighted squared error
        weight = self.weighting_function(X_ij)
        cost = weight * (prediction - target) ** 2
        
        return cost
    
    def update_parameters(self, i, j, X_ij, learning_rate):
        """
        Compute gradients and update parameters using gradient descent
        
        Args:
            i: Index of target word
            j: Index of context word
            X_ij: Co-occurrence count
            learning_rate: Step size for gradient descent
        """
        # Forward pass: compute prediction
        prediction = np.dot(self.W[i], self.W_context[j]) + self.b[i] + self.b_context[j]
        
        # Target value
        target = np.log(X_ij + 1e-10)
        
        # Compute weighted difference
        weight = self.weighting_function(X_ij)
        diff = weight * (prediction - target)
        
        # Compute gradients (chain rule)
        grad_W_i = diff * self.W_context[j]
        grad_W_context_j = diff * self.W[i]
        grad_b_i = diff
        grad_b_context_j = diff
        
        # Update parameters (gradient descent)
        self.W[i] -= learning_rate * grad_W_i
        self.W_context[j] -= learning_rate * grad_W_context_j
        self.b[i] -= learning_rate * grad_b_i
        self.b_context[j] -= learning_rate * grad_b_context_j
    
    def train(self, epochs=50, learning_rate=0.05, lr_decay=0.99, verbose=True):
        """
        Train GloVe embeddings
        
        Args:
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            lr_decay: Learning rate decay factor per epoch
            verbose: Whether to print progress
        
        Returns:
            cost_history: List of average costs per epoch
        """
        # Extract non-zero co-occurrences
        nonzero_entries = []
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                if self.matrix[i, j] > 0:
                    nonzero_entries.append((i, j, self.matrix[i, j]))
        
        n_entries = len(nonzero_entries)
        
        if verbose:
            print(f"\nTraining GloVe on {n_entries} non-zero co-occurrences")
            print("="*60)
        
        cost_history = []
        
        for epoch in range(epochs):
            total_cost = 0
            
            # Shuffle for better convergence
            np.random.shuffle(nonzero_entries)
            
            # Process each non-zero co-occurrence
            for i, j, X_ij in nonzero_entries:
                # Compute cost
                cost = self.compute_cost(i, j, X_ij)
                total_cost += cost
                
                # Update parameters
                self.update_parameters(i, j, X_ij, learning_rate)
            
            # Average cost
            avg_cost = total_cost / n_entries
            cost_history.append(avg_cost)
            
            # Decay learning rate
            learning_rate *= lr_decay
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} | Cost: {avg_cost:.6f} | LR: {learning_rate:.6f}")
        
        if verbose:
            print("="*60)
            print(f"Training complete!")
            print(f"Final cost: {cost_history[-1]:.6f}")
        
        return cost_history
    
    def get_embeddings(self):
        """
        Get final word embeddings
        
        GloVe paper suggests averaging main and context embeddings
        
        Returns:
            embeddings: numpy array [vocab_size, embedding_dim]
        """
        return (self.W + self.W_context) / 2
    
    def get_vector(self, word):
        """
        Get embedding vector for a specific word
        
        Args:
            word: Word string
        
        Returns:
            vector: numpy array of embedding_dim, or None if word not in vocab
        """
        if word in self.word2idx:
            idx = self.word2idx[word]
            embeddings = self.get_embeddings()
            return embeddings[idx]
        return None
    
    def most_similar(self, word, top_n=5):
        """
        Find most similar words using cosine similarity
        
        Args:
            word: Query word
            top_n: Number of similar words to return
        
        Returns:
            List of (word, similarity) tuples
        """
        if word not in self.word2idx:
            return []
        
        word_vec = self.get_vector(word)
        embeddings = self.get_embeddings()
        
        similarities = []
        for other_word in self.vocab:
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
        
        Args:
            word_a, word_b, word_c: Words in analogy
            top_n: Number of results to return
        
        Returns:
            List of (word, similarity) tuples
        """
        if not all(w in self.word2idx for w in [word_a, word_b, word_c]):
            return []
        
        vec_a = self.get_vector(word_a)
        vec_b = self.get_vector(word_b)
        vec_c = self.get_vector(word_c)
        
        # Compute target: vec_b - vec_a + vec_c
        target_vec = vec_b - vec_a + vec_c
        
        embeddings = self.get_embeddings()
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


# Example usage
if __name__ == "__main__":
    # Sample data
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "log"],
        ["cats", "and", "dogs", "are", "animals"],
        ["the", "cat", "and", "dog", "played"],
        ["animals", "like", "cats", "are", "pets"],
        ["dogs", "are", "loyal", "animals"],
        ["the", "mat", "is", "on", "the", "floor"],
        ["the", "log", "is", "on", "the", "ground"],
    ]
    
    # Build vocabulary
    vocab = sorted(set(word for sentence in sentences for word in sentence))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)
    
    # Build co-occurrence matrix (simplified for demo)
    matrix = np.zeros((vocab_size, vocab_size))
    window_size = 2
    
    for sentence in sentences:
        for i, target in enumerate(sentence):
            target_idx = word2idx[target]
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                context_pos = i + offset
                if 0 <= context_pos < len(sentence):
                    context = sentence[context_pos]
                    context_idx = word2idx[context]
                    distance = abs(offset)
                    weight = 1.0 / distance
                    matrix[target_idx][context_idx] += weight
    
    print("="*60)
    print("GloVe IMPLEMENTATION DEMO")
    print("="*60)
    
    # Initialize and train GloVe
    glove = GloVe(
        cooccurrence_matrix=matrix,
        vocab=vocab,
        word2idx=word2idx,
        embedding_dim=20,
        x_max=100,
        alpha=0.75
    )
    
    # Train
    cost_history = glove.train(epochs=100, learning_rate=0.05, lr_decay=0.99)
    
    # Test similarity
    print("\n" + "="*60)
    print("SIMILARITY TESTS")
    print("="*60)
    
    test_words = ["cat", "dog", "mat", "animals"]
    for word in test_words:
        if word in word2idx:
            print(f"\nMost similar to '{word}':")
            similar = glove.most_similar(word, top_n=3)
            for sim_word, score in similar:
                print(f"  {sim_word:12s} → {score:.4f}")
    
    # Show embeddings
    print("\n" + "="*60)
    print("SAMPLE EMBEDDINGS")
    print("="*60)
    for word in ["cat", "dog", "animals"]:
        vec = glove.get_vector(word)
        if vec is not None:
            print(f"\n'{word}' (first 5 dims): {vec[:5]}")
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Initial cost: {cost_history[0]:.6f}")
    print(f"Final cost:   {cost_history[-1]:.6f}")
    print(f"Improvement:  {cost_history[0] - cost_history[-1]:.6f}")
    print("="*60)