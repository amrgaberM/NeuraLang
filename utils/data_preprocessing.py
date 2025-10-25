import re
import string
import numpy as np

# --- Dependency for clean_text_v3 ---
# This map is our "domain knowledge" for English contractions.
# It is kept global here as it's a constant.
CONTRACTION_MAP = {
    "can't": "cannot",
    "won't": "will not",
    "it's": "it is",
    "let's": "let us",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
}

def clean_text(text: str, expand_contractions: bool = True) -> str:
    """
    Cleans and normalizes a single string with optional policy flags.
    
    Logical Order:
    1. Normalize special entities (URLs, emails, numbers, emojis).
    2. Convert to lowercase.
    3. Optionally expand contractions.
    4. Remove remaining punctuation.
    5. Collapse and strip whitespace.
    """
    
    # --- Robustness Check ---
    if not isinstance(text, str):
        return ""

    # --- Step 1: Normalize Special Entities ---
    # We run this *before* lowercasing or punctuation removal
    # to correctly capture patterns like 'example.COM' or 'https://...'
    
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', '_URL_', text, flags=re.IGNORECASE)
    # Replace Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '_EMAIL_', text)
    # Replace Numbers (integers, decimals)
    text = re.sub(r'\b\d+[\.,\']?\d*\b', '_NUM_', text)
    # Replace Emojis (basic range)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', '_EMOJI_', text)

    # --- Step 2: Convert to lowercase ---
    text = text.lower()

    # --- Step 3: Optionally Expand Contractions ---
    if expand_contractions:
        cleaned_words = []
        # Split on whitespace to process word by word
        for word in text.split():
            # Use .get() to lookup the word; if not found, return the word itself
            expanded_word = CONTRACTION_MAP.get(word, word)
            cleaned_words.append(expanded_word)
        # Re-join the words into a string
        text = ' '.join(cleaned_words)

    # --- Step 4: Remove Remaining Punctuation ---
    # Now that entities and contractions are handled, we can safely remove
    # all remaining punctuation marks.
    text = text.translate(str.maketrans('', '', string.punctuation))

    # --- Step 5: Collapse Whitespace and Strip ---
    # Replace one or more whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize(text: str) -> list[str]:
    """
    Splits a cleaned string into a list of tokens.
    Assumes text is already cleaned and normalized by clean_text_v3.
    """
    # Python's .split() with no arguments is highly efficient.
    # It splits on any whitespace and automatically handles
    # empty strings or strings with only spaces, returning [].
    return text.split()


def build_cooccurrence_matrix(corpus, window_size=2, distance_weighting=False, min_count=1):
    """
    Build co-occurrence matrix using existing utility functions

    Args:
        corpus: List of sentences or single text string
        window_size: Context window size (±n words)
        distance_weighting: If True, weight by 1/distance
        min_count: Minimum word frequency to include (NEW!)

    Returns:
        matrix: numpy array of shape (vocab_size, vocab_size)
        vocab: sorted list of unique words
        word2idx: dictionary mapping words to indices
    """
    # Handle both string and list inputs
    if isinstance(corpus, str):
        corpus = corpus.split('\n')

    # Use YOUR clean_text and tokenize functions
    tokens = []
    word_counts = {}  # NEW: Track word frequencies
    
    for sentence in corpus:
        cleaned = clean_text(sentence)
        words = tokenize(cleaned)
        
        if words:
            tokens.append(words)
            # Count word frequencies
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

    # Filter by min_count (NEW!)
    vocab = sorted([word for word, count in word_counts.items() 
                    if count >= min_count])
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)

    # Initialize co-occurrence matrix
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    # Populate matrix with co-occurrence counts
    for sentence in tokens:
        # Filter sentence to only include vocab words (NEW!)
        sentence = [w for w in sentence if w in word2idx]
        
        for i, target_word in enumerate(sentence):
            target_idx = word2idx[target_word]

            # Look at context window
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue

                context_pos = i + offset

                if 0 <= context_pos < len(sentence):
                    context_word = sentence[context_pos]
                    context_idx = word2idx[context_word]

                    distance = abs(offset)
                    weight = 1.0 / distance if distance_weighting else 1.0

                    matrix[target_idx][context_idx] += weight

    return matrix, vocab, word2idx