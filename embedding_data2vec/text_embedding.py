import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import csv


# Function to process text input and generate aligned embeddings
def generate_text_embeddings(data2vec_model, tokenizer, word_timings, sampling_rate=30):
    """
    Generate text embeddings aligned with audio and motion data.

    Args:
        data2vec_model: Data2Vec model for text embeddings.
        tokenizer: Tokenizer object for the model.
        word_timings (list of tuples): List of words with start and end times [(word, start, end), ...].
        sampling_rate (int): Rate in Hz to replicate embeddings.

    Returns:
        np.ndarray: Text embeddings aligned at the specified sampling rate.
    """
    # Extract the text from the word timings
    text = " ".join([word for word, _, _ in word_timings])
    print("text:", text)

    # Tokenize the text input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(inputs)
    print("inputs:", inputs.input_ids.shape)

    # Pass the tokenized text through the model
    with torch.no_grad():
        outputs = data2vec_model(**inputs)
        # token_embeddings Shape: (seq_len, hidden_dim)
        token_embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    print("token_embeddings", token_embeddings.shape)
    # Extract the last hidden layer
    # hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
    # token_embeddings = hidden_states.squeeze(0).numpy()  # Shape: (seq_len, hidden_dim)

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
    custom_cache_dir = "./data2vec-text-base"
    csv_file = "./001_Neutral_0_x_1_0.csv"
    # Load the data2vec-text-base model and tokenizer
    MODEL_NAME = "facebook/data2vec-text-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=custom_cache_dir)
    data2vec_model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=custom_cache_dir)
    word_timings = []
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Extract data from each row
            word = row["Label"]
            begin = float(row["Begin"])
            end = float(row["End"])
            wtype = row["Type"]

            if wtype == "words":
                word_timings.append((word, begin, end))

    # word_timings = [
    #     ("This", 0.0, 0.3),
    #     ("is", 0.3, 0.5),
    #     ("a", 0.5, 0.6),
    #     ("sample", 0.6, 1.0),
    #     ("sentence", 1.0, 1.5)
    # ]
    # print(word_timings)

    aligned_text_embeddings = generate_text_embeddings(data2vec_model, tokenizer, word_timings)
    print("Aligned text embeddings shape:", aligned_text_embeddings.shape)

    # Original
    # shape: (6071, 768)
    # Resampled
    # shape: (3643, 768)
    # Aligned text embeddings
    # shape: (1293, 768)
    # (1979, 768)