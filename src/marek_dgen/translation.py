from openai import OpenAI
import re
import unicodedata

from num2words import num2words


_CZECH_DIGITS = {
    "0": "nula",
    "1": "jedna",
    "2": "dva",
    "3": "tři",
    "4": "čtyři",
    "5": "pět",
    "6": "šest",
    "7": "sedm",
    "8": "osm",
    "9": "devět",
}


def _digits_to_words(match: re.Match) -> str:
    """Convert dotted numbers like 2.0 to 'dva nula'."""
    return " ".join(_CZECH_DIGITS[ch] for ch in match.group(0) if ch in _CZECH_DIGITS)


def _ordinal_to_words(match: re.Match) -> str:
    """Convert an ordinal like 19. to a Czech ordinal word."""
    return num2words(int(match.group(1)), lang="cz", to="ordinal")


def _cardinal_to_words(match: re.Match) -> str:
    """Convert a cardinal number to a Czech word."""
    return num2words(int(match.group(0)), lang="cz")


def post_process_numbers(text: str) -> str:
    """Convert remaining numeric patterns in translated text to Czech words."""
    # 1. Dotted versions / decimals: 2.0, 3.0 -> dva nula, tři nula
    text = re.sub(r"(?<![\d.])\d+\.\d+(?![\d.])", _digits_to_words, text)
    # 2. Ordinals ending with a dot: 19. -> devatenáctý
    text = re.sub(r"(?<!\d)(\d+)\.(?!\d)", _ordinal_to_words, text)
    # 3. Remaining integers
    text = re.sub(r"(?<!\d)\d+(?!\d)", _cardinal_to_words, text)
    return text


def translation_request(
    content_desc: str,
    base_url: str,
    einfra_api_key: str,
    translating_model: str,
    temperature: float,
    max_tokens: int,
    text_segment: str,
):
    # 120 s total timeout + 2 retries on transient failures.
    # Lower the timeout if you want faster failure detection.
    client = OpenAI(
        base_url=base_url,
        api_key=einfra_api_key,
        timeout=120.0,
        max_retries=2,
    )

    system_prompt = f"""
  You are an expert English to Czech translator.
  CONTEXT: The following text is a direct transcription of a {content_desc}.
  TASK: Translate the dialogue directly into Czech, maintaining the conversational yet scientific tone. Use appropriate terminology.
  STRICT CONSTRAINTS:
  - Translate ONLY the exact words provided.
  - Do not complete unfinished sentences or add missing information, even if you know the context.
  - Do not add any extra explanations, notes, or conversational filler.
  - Return ONLY the translated Czech text.
  - Every digit or number has to be written as word, not as numeric symbol. For example, not x:y but always transform for example '19.' to the textual 'devatenáctého'" 
  """

    prompt = {
        # e.g. "aya-expanse" or "glm-4.7", "thinker", "llama-4-scout-17b-16e-instruct",
        # "qwen3-reranker-4b", "qwen3-embedding-4b", "mxbai-embed-large:latest",
        # "multilingual-e5-large-instruct", "deepseek-v3.2", "deepseek-v3.2-thinking",
        # "qwen3.5", "gpt-oss-120b", "glm-5", "kimi-k2.5", "qwen3.5-122b"
        "model": translating_model,  # "aya-expanse"
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_segment},
        ],
        "temperature": temperature,  # 0.1, # adjust as needed
        "max_tokens": max_tokens,  # lower max tokens for shorter translations, adjust as needed
    }

    try:
        response = client.chat.completions.create(**prompt)
    except Exception as exc:
        raise RuntimeError(
            f"Translation API call failed for segment: {text_segment[:80]}..."
        ) from exc

    raw_response = response.choices[0].message.content.strip()
    raw_response = post_process_numbers(raw_response)
    normalized_response = unicodedata.normalize("NFC", raw_response)

    return normalized_response
