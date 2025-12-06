import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Load the data2vec-text-base model and tokenizer
MODEL_NAME = "facebook/data2vec-text-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


# Function to process text input and generate aligned embeddings
def generate_text_embeddings(word_timings, sampling_rate=30):
    """
    Generate text embeddings aligned with audio and motion data.

    Args:
        word_timings (list of tuples): List of words with start and end times [(word, start, end), ...].
        sampling_rate (int): Rate in Hz to replicate embeddings.

    Returns:
        np.ndarray: Text embeddings aligned at the specified sampling rate.
    """
    # Extract the text from the word timings
    text = " ".join([word for word, _, _ in word_timings])

    # Tokenize the text input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Pass the tokenized text through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the last hidden layer
    hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
    token_embeddings = hidden_states.squeeze(0).numpy()  # Shape: (seq_len, hidden_dim)

    # Align embeddings with word timings
    aligned_embeddings = []
    for word_idx, (word, start_time, end_time) in enumerate(word_timings):
        duration = end_time - start_time
        n_samples = int(duration * sampling_rate)
        embedding = token_embeddings[word_idx]
        aligned_embeddings.append(np.tile(embedding, (n_samples, 1)))  # Replicate embedding

    # Concatenate all aligned embeddings to form a single sequence
    return np.vstack(aligned_embeddings)


# Example usage
if __name__ == "__main__":
    word_timings = [
        ("This", 0.0, 0.3),
        ("is", 0.3, 0.5),
        ("a", 0.5, 0.6),
        ("sample", 0.6, 1.0),
        ("sentence", 1.0, 1.5)
    ]

    aligned_text_embeddings = generate_text_embeddings(word_timings)
    print("Aligned text embeddings shape:", aligned_text_embeddings.shape)
