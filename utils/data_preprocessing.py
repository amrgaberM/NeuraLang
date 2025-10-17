import re
import string

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