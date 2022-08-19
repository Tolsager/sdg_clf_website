import os
import transformers
import re


def get_tokenizer(tokenizer_type: str):
    path_tokenizers = "sdg_clf/tokenizers"
    path_tokenizer = os.path.join(path_tokenizers, tokenizer_type)
    if not os.path.exists(path_tokenizer):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        tokenizer.save_pretrained(path_tokenizer)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(path_tokenizer)
    return tokenizer


def remove_with_regex(text: str, patterns: list[str]) -> str:
    """
    Remove text with regex patterns.
    Args:
        text: text to be edited
        patterns: patterns to indicate what to substitute

    Returns:


    """
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    return text


def process_text(text: str) -> str:
    """
    process the text by removing labels and noise.

    Args:
        text: text to process

    Returns:
        processed text
    """
    # lower text
    text = text.lower()

    # remove labels from the tweet
    sdg_prog1 = r"#(?:sdg)s?(\s+)?(\d+)?"
    sdg_prog2 = r"(?:sdg)s?(\s?)(\d+)?"
    sdg_prog3 = r"(sustainable development goals?\s?)(\d+)?"
    copyright_prog = r"© \d\d(\d?)\d"
    elsevier_prog = r"elsevier\s+Ltd"
    url_prog = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    patterns = [sdg_prog1, sdg_prog2, sdg_prog3, copyright_prog, elsevier_prog, url_prog]

    text = remove_with_regex(text, patterns)
    return text
