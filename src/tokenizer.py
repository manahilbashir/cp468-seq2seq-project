import re
import unicodedata


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def clean_text(text: str, lowercase: bool = True):
    """
    Clean and normalize text.
    """

    text = unicodedata.normalize("NFKC", str(text))

    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def tokenize(text: str, lowercase: bool = True):
    """
    Convert text into tokens.
    """

    text = clean_text(text, lowercase)

    return TOKEN_PATTERN.findall(text)