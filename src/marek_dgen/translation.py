from openai import OpenAI

def translation_request(translating_model: str, whole_text: str, rec_desc: str, temp: float, max_tokens: int):
  client = OpenAI()
  base = "https://llm.ai.e-infra.cz/v1/"
  model = translating_model
  target_lang = "cs"


  prompt_sys = f"You are a professional translator. Translate the following English text into natural, fluent Czech. The text is recording of {rec_desc}. Do not add any explanations, introductions, markdown, or any additional characters except newlines. Output ONLY the translated text."



	# {
	# 	"model": model, # "aya-expanse",
	# 	"messages": [
	# 		{
	# 			"role": "system",
	# 			"content": "You are a professional translator. Translate the following English text into natural, fluent Czech. Do not add any explanations, introductions, or markdown. Output ONLY the translated text."
	# 		},
	# 		{
	# 			"role": "user",
	# 			"content": "Hello everyone, welcome to the meeting.\nThanks, Jan. Let's start with the agenda."
	# 		}
	# 	],
	# 	"temperature": 0.0,
	# 	"max_tokens": max_tokens # lower max tokens for shorter translations, adjust as needed
	# }




# THIS PROBS>>>>
# Streaming responses

# We provide support for streaming responses using Server Side Events (SSE).

# from openai import OpenAI

# client = OpenAI()

# stream = client.responses.create(
#     model="gpt-5.2",
#     input="Write a one-sentence bedtime story about a unicorn.",
#     stream=True,
# )

# for event in stream:
#     print(event)





# Realtime API

# The Realtime API enables you to build low-latency, multi-modal conversational experiences. It currently supports text and audio as both input and output, as well as function calling through a WebSocket connection.

# Under the hood the SDK uses the websockets library to manage connections.

# The Realtime API works through a combination of client-sent events and server-sent events. Clients can send events to do things like update session configuration or send text and audio inputs. Server events confirm when audio responses have completed, or when a text response from the model has been received. A full event reference can be found here and a guide can be found here.

# Basic text based example:

# import asyncio
# from openai import AsyncOpenAI

# async def main():
#     client = AsyncOpenAI()

#     async with client.realtime.connect(model="gpt-realtime") as connection:
#         await connection.session.update(
#             session={"type": "realtime", "output_modalities": ["text"]}
#         )

#         await connection.conversation.item.create(
#             item={
#                 "type": "message",
#                 "role": "user",
#                 "content": [{"type": "input_text", "text": "Say hello!"}],
#             }
#         )
#         await connection.response.create()

#         async for event in connection:
#             if event.type == "response.output_text.delta":
#                 print(event.delta, flush=True, end="")

#             elif event.type == "response.output_text.done":
#                 print()

#             elif event.type == "response.done":
#                 break

# asyncio.run(main())

# However the real magic of the Realtime API is handling audio inputs / outputs, see this example TUI script for a fully fledged example.
# Realtime error handling

# Whenever an error occurs, the Realtime API will send an error event and the connection will stay open and remain usable. This means you need to handle it yourself, as no errors are raised directly by the SDK when an error event comes in.

# client = AsyncOpenAI()

# async with client.realtime.connect(model="gpt-realtime") as connection:
#     ...
#     async for event in connection:
#         if event.type == 'error':
#             print(event.error.type)
#             print(event.error.code)
#             print(event.error.event_id)
#             print(event.error.message)
