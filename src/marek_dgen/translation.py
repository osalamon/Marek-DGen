from openai import OpenAI
import unicodedata


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
    normalized_response = unicodedata.normalize("NFC", raw_response)

    return normalized_response
