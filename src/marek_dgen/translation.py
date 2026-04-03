from openai import OpenAI

def translation_request(base_url: str, einfra_api_key: str, translating_model: str, text_segment: str, content_desc: str, temperature: float, max_tokens: int):
  client = OpenAI(
    base_url = base_url,
    api_key = einfra_api_key
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
  	"model": "aya-expanse", # translating_model,
  	"messages": [
  		{
  			"role": "system",
  			"content": system_prompt
  		},
  		{
  			"role": "user",
  			"content": text_segment
  		}
  	],
  	"temperature": temperature, 	# 0.2, # adjust as needed
  	"max_tokens": max_tokens 			# lower max tokens for shorter translations, adjust as needed
  }
  response = client.chat.completions.create(**prompt)

  return response.choices[0].message.content.strip()
