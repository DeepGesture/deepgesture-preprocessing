import torch
import torchaudio
from transformers import (
    Data2VecTextModel,
    Data2VecAudioModel,
    AutoTokenizer,
    AutoFeatureExtractor
)
import numpy as np
from scipy.signal import resample_poly
from typing import Dict, List, Optional, Tuple
import csv
import logging


class AudioEmbedder:
    def __init__(
            self,
            model_name: str = "facebook/data2vec-audio-base-960h",
            device: str = "cuda:0"
    ):
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Data2VecAudioModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def load_and_resample_audio(
            self,
            audio_path: str,
            target_sample_rate: int = 16000
    ) -> torch.Tensor:
        """Load and resample audio to 16kHz."""
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate,
                target_sample_rate
            )
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def process_audio(
            self,
            audio_path: str,
            target_fps: int = 30
    ) -> np.ndarray:
        """
        Process audio file and extract embeddings.

        Args:
            audio_path: Path to audio file
            target_fps: Target frames per second for output

        Returns:
            numpy.ndarray: Sequence of embeddings at target_fps
        """
        # Load and preprocess audio
        waveform = self.load_and_resample_audio(audio_path)
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden state [batch, sequence_length, hidden_size]
            embeddings = outputs.last_hidden_state[0].cpu().numpy()

        # Resample from 50Hz to target_fps
        resampled_embeddings = align_embeddings_to_motion(
            embeddings,
            source_fps=50,  # data2vec-audio outputs at 50Hz
            target_fps=target_fps
        )

        return resampled_embeddings


class TextEmbedder:
    def __init__(
            self,
            model_name: str = "facebook/data2vec-text-base",
            device: str = "cuda:0"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Data2VecTextModel.from_pretrained(model_name).to(device)
        self.model.eval()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_transcript(
            self,
            word_timings: List[Tuple[str, float, float]],
            target_fps: int = 30
    ) -> np.ndarray:
        """
        Process text with word-level timings to create aligned embeddings.

        Args:
            word_timings: List of tuples (word, start_time, end_time)
            target_fps: Target frames per second for output

        Returns:
            numpy.ndarray: Sequence of embeddings at target_fps
        """
        # Filter out empty words and validate timings
        word_timings = [(word.strip(), start, end) for word, start, end in word_timings
                        if word.strip() and end > start]

        if not word_timings:
            raise ValueError("No valid words found in word_timings")

        # Process each word separately to maintain accurate token mapping
        word_embeddings = []

        for word, start_time, end_time in word_timings:
            # Tokenize single word
            tokens = self.tokenizer(word, return_tensors="pt", padding=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            # Get embeddings for this word
            with torch.no_grad():
                outputs = self.model(**tokens, output_hidden_states=True)
                # Get last hidden state and remove batch dimension
                word_embedding = outputs.last_hidden_state[0].cpu().numpy()

                # Average all token embeddings for this word (excluding special tokens)
                # First token ([CLS]) and last token ([SEP]) are special tokens
                if word_embedding.shape[0] > 2:  # If we have more than just special tokens
                    word_embedding = word_embedding[1:-1].mean(axis=0)
                else:
                    # If we only have special tokens, use the middle position
                    word_embedding = word_embedding[1]

                # Calculate frames needed for this word
                duration = end_time - start_time
                num_frames = max(1, int(duration * target_fps))

                # Repeat embedding for word duration
                repeated_embeddings = np.tile(word_embedding, (num_frames, 1))
                word_embeddings.append(repeated_embeddings)

        if not word_embeddings:
            raise ValueError("No valid embeddings generated")

        # Concatenate all word embeddings
        final_embeddings = np.concatenate(word_embeddings, axis=0)
        return final_embeddings

    def map_tokens_to_words(self, text: str, encoding, words: List[str]) -> List[List[str]]:
        """
        Maps tokenized pieces back to original words.

        Args:
            text: Original text
            encoding: Tokenizer encoding output
            words: List of original words

        Returns:
            List of lists containing tokens for each word
        """
        offset_mapping = encoding['offset_mapping'][0].numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

        current_word = 0
        current_word_start = 0
        word_tokens = [[] for _ in words]

        for token_idx, (start, end) in enumerate(offset_mapping):
            # Skip special tokens
            if start == end == 0:
                continue

            # Find which word this token belongs to
            while current_word < len(words):
                word = words[current_word]
                word_end = current_word_start + len(word)

                # If token overlaps with current word
                if start >= current_word_start and start < word_end:
                    word_tokens[current_word].append(tokens[token_idx])
                    break

                current_word += 1
                current_word_start = text.find(word, current_word_start) + len(word)

        return word_tokens

    @staticmethod
    def load_word_timings(csv_file: str) -> List[Tuple[str, float, float]]:
        """
        Load word timings from a CSV file.

        Args:
            csv_file: Path to CSV file containing word timing data

        Returns:
            List of tuples containing (word, start_time, end_time)
        """
        word_timings = []
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Extract data from each row
                word = row["Label"]
                begin = float(row["Begin"])
                end = float(row["End"])
                wtype = row["Type"]

                if wtype == "words" and word.strip():  # Skip empty words
                    word_timings.append((word, begin, end))
        return word_timings


def align_embeddings_to_motion(
        embeddings: np.ndarray,
        source_fps: int,
        target_fps: int
) -> np.ndarray:
    """
    Resample embeddings to match motion data frame rate using polyphase resampling.

    Args:
        embeddings: Input embeddings sequence
        source_fps: Original frame rate
        target_fps: Desired frame rate

    Returns:
        numpy.ndarray: Resampled embeddings sequence
    """
    if source_fps == target_fps:
        return embeddings

    # Calculate resampling parameters
    gcd = np.gcd(source_fps, target_fps)
    up = target_fps // gcd
    down = source_fps // gcd

    # Reshape embeddings to handle each dimension separately
    orig_shape = embeddings.shape
    resampled = []

    for dim in range(orig_shape[1]):
        resampled.append(
            resample_poly(embeddings[:, dim], up, down)
        )

    # Stack back together
    return np.stack(resampled, axis=1)


class MultimodalEmbedder:
    def __init__(
            self,
            audio_model: str = "facebook/data2vec-audio-base-960h",
            text_model: str = "facebook/data2vec-text-base",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.audio_embedder = AudioEmbedder(audio_model, device)
        self.text_embedder = TextEmbedder(text_model, device)

    def process_input(
            self,
            audio_path: str,
            text: str,
            word_timings: List[Dict[str, float]],
            target_fps: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process both audio and text inputs to get aligned embeddings.

        Args:
            audio_path: Path to audio file
            text: Input text
            word_timings: Word-level timing information
            target_fps: Target frame rate for outputs

        Returns:
            Tuple[np.ndarray, np.ndarray]: Audio and text embeddings aligned at target_fps
        """
        audio_embeddings = self.audio_embedder.process_audio(
            audio_path,
            target_fps=target_fps
        )

        text_embeddings = self.text_embedder.process_transcript(
            text,
            word_timings,
            target_fps=target_fps
        )

        return audio_embeddings, text_embeddings


# Example usage
if __name__ == "__main__":
    # # Initialize embedder
    # embedder = MultimodalEmbedder()
    #
    # # Example data
    # audio_path = "./001_Neutral_0_x_1_0.wav"
    #
    # # Process both modalities
    # audio_embeddings, text_embeddings = embedder.process_input(
    #     audio_path,
    #     text,
    #     word_timings,
    #     target_fps=30
    # )
    #
    # print(f"Audio embeddings shape: {audio_embeddings.shape}")
    # print(f"Text embeddings shape: {text_embeddings.shape}")
    # Initialize embedder
    embedder = TextEmbedder(device="mps")

    # # Example 1: Using direct word timings
    word_timings = [
        ("This", 0.0, 0.3),
        ("is", 0.3, 0.5),
        ("a", 0.5, 0.6),
        ("sample", 0.6, 1.0),
        ("sentence", 1.0, 1.5)
    ]
    #
    # embeddings = embedder.process_transcript(word_timings, target_fps=30)
    # print(f"Direct timing embeddings shape: {embeddings.shape}")
    csv_file = "./001_Neutral_0_x_1_0.csv"

    try:
        # word_timings = embedder.load_word_timings(csv_file)
        # print("word_timings", word_timings)
        embeddings = embedder.process_transcript(word_timings, target_fps=30)
        print(f"CSV timing embeddings shape: {embeddings.shape}")
    except FileNotFoundError:
        print(f"CSV file {csv_file} not found")
