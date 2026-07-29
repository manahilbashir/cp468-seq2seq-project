from collections import Counter
import json


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class Vocabulary:
    def __init__(self, token_to_id, id_to_token):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    @property
    def pad_id(self):
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self):
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self):
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self):
        return self.token_to_id["<eos>"]

    def __len__(self):
        return len(self.id_to_token)

    def encode(self, tokens):
        """
        Convert tokens into numerical IDs.
        Adds beginning-of-sentence and end-of-sentence tokens.
        """

        token_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]

        return [self.bos_id] + token_ids + [self.eos_id]

    def decode(self, token_ids):
        """
        Convert numerical IDs back into tokens.
        """

        return [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

    def save(self, file_path):
        """
        Save the vocabulary to a JSON file.
        """

        vocabulary_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                vocabulary_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, file_path):
        """
        Load a vocabulary from a JSON file.
        """

        with open(file_path, "r", encoding="utf-8") as file:
            vocabulary_data = json.load(file)

        return cls(
            vocabulary_data["token_to_id"],
            vocabulary_data["id_to_token"],
        )


def build_vocabulary(
    tokenized_sentences,
    min_frequency=2,
    max_size=30000,
):
    """
    Build a vocabulary from tokenized training sentences.

    Important:
    This function must only receive training data.
    """

    token_counter = Counter(
        token
        for sentence in tokenized_sentences
        for token in sentence
    )

    sorted_tokens = sorted(
        token_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )

    selected_tokens = [
        token
        for token, frequency in sorted_tokens
        if frequency >= min_frequency
        and token not in SPECIAL_TOKENS
    ]

    selected_tokens = selected_tokens[
        : max_size - len(SPECIAL_TOKENS)
    ]

    id_to_token = SPECIAL_TOKENS + selected_tokens

    token_to_id = {
        token: index
        for index, token in enumerate(id_to_token)
    }

    return Vocabulary(token_to_id, id_to_token)