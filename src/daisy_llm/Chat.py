import openai
import logging
import nltk.data
import threading
import time
import yaml
import json
import requests
import re

from .ChatSpeechProcessor import ChatSpeechProcessor
from .SoundManager import SoundManager
from .Text import print_text, delete_last_lines
import pprint




class Chat:
	description = "Implements a chatbot using OpenAI's GPT-3 language model and allows for interaction with the user through speech or text."

	def __init__(self, ml=None, ch=None):
		self.ml = ml
		self.ch = ch
		self.csp = ChatSpeechProcessor()
		self.sounds = SoundManager()

		with open("configs.yaml", "r") as f:
			self.configs = yaml.safe_load(f)
		openai.api_key = self.configs["keys"]["openai"]

		nltk.data.load('tokenizers/punkt/english.pickle')

	def request(self,
	     messages,
		 stop_event=None,
		 sound_stop_event=None,
		 tts=None,
		 tool_check=True,
		 model="gpt-3.5-turbo",
		 silent=False,
		 response_label=True
		 ):
		#Handle LLM request. Optionally convert to sentences and queue for tts, if needed.

		#Queues for handling chunks, sentences, and tts sounds
		sentences = [[]]  # create a queue to hold the sentences



		if not stop_event:
			stop_event = threading.Event()
		if not sound_stop_event:
			sound_stop_event = threading.Event()

		#Flags for handling chunks, sentences, and tts sounds
		sentence_queue_canceled = [False]  # use a list to make response_canceled mutable
		sentence_queue_complete = [False]	# use a list to make response_complete mutable

		threads = []  # keep track of all threads created
		text_stream = [""]
		return_text = [""]

		#If tool check enabled, find the appropriate tool, and append the output to the messages list.
		if tool_check:
			response = self.toolform_checker(
				messages=messages, 
				stop_event=stop_event,
				response_label=response_label,
				model='gpt-3.5-turbo'
				)
			if response:

				print_text("SYSTEM:", "red", "", "bold")
				print_text(response, None, "\n")

				#Append to CH_context, and the local messages list.
				self.ch.add_message_object("system", response)
				messages.append(self.ch.single_message_context("system", response, False))

		#pprint.pprint(messages)
		try:
			logging.info("Sending request to OpenAI model...")
			response = openai.ChatCompletion.create(
				model=model,
				messages=messages,
				temperature=0.7,
				stream=True,
				request_timeout=5,
			)

			#Handle chunks. Optionally convert to sentences for sentence_queue, if needed.
			arguments = {
				'response': response,
				'text_stream': text_stream,
				'sentences': sentences,
				'sentence_queue_canceled': sentence_queue_canceled,
				'sentence_queue_complete': sentence_queue_complete,
				'return_text': return_text,
				'stop_event': stop_event,
				'sound_stop_event': sound_stop_event,
				'silent': silent,
				'model': model,
				'response_label': response_label
			}
			t = threading.Thread(target=self.stream_queue_sentences, args=(arguments,))
			t.start()
			threads.append(t)

			if tts:
				self.csp.queue_and_tts_sentences(
					tts=tts, 
					sentences=sentences, 
					sentence_queue_canceled=sentence_queue_canceled, 
					sentence_queue_complete=sentence_queue_complete, 
					stop_event=stop_event, 
					sound_stop_event=sound_stop_event
					)

			while not return_text[0] and not stop_event.is_set():
				time.sleep(0.1)  # wait a bit before checking again

			# return response_complete and return_text[0] when return_text is set

			t.join()

			return return_text[0]
		
		# Handle different types of errors that may occur when sending request to OpenAI model
		except openai.error.Timeout as e:
			logging.error(f"Timeout: {e}")
			#self.csp.tts("TimeoutError Error. Check your internet connection.")
			return False  
		except openai.error.APIConnectionError as e:
			logging.error(f"APIConnectionError: {e}")
			#self.csp.tts("APIConnectionError. Sorry, I can't talk right now.")
			return False  
		except openai.error.InvalidRequestError as e:
			logging.error(f"Invalid Request Error: {e}")
			#self.csp.tts("Invalid Request Error. Sorry, I can't talk right now.")
			return False        
		except openai.APIError as e:
			logging.error(f"API Error: {e}")
			#self.csp.tts("API Error. Sorry, I can't talk right now.")
			return False
		except openai.error.RateLimitError as e:
			logging.error(f"RateLimitError: {e}")
			#self.csp.tts("Rate Limit Error. Sorry, I can't talk right now.")
			return False
		except ValueError as e:
			logging.error(f"Value Error: {e}")
			#self.csp.tts("Value Error. Sorry, I can't talk right now.")
			return False    
		except TypeError as e:
			logging.error(f"Type Error: {e}")
			#self.csp.tts("Type Error. Sorry, I can't talk right now.")
			return False  

	def toolform_checker(
			self, 
			messages, 
			stop_event=None, 
			response_label=True,
			model='gpt-3.5-turbo'
			):
		logging.info("Checking for tool forms...")

		#HOOK: Chat_request_inner
		#Right now, only one hook can be run at a time. If a hook returns a value, the rest of the hooks are skipped.
		#I may update this soon to allow for inline responses (For example: "5+5 is [Calculator: 5+5]")
		if self.ml:
			hook_instances = self.ml.get_hook_instances()
			logging.debug(hook_instances)

			if "Chat_request_inner" in hook_instances:

				#Create a tool-chooser prompt
				prompt = """1. "Tools" contains a list of available tools for you to use.
2. Choose one, or more, or None of the tools that are most useful given the context of the "Conversation".
4. Format your response using JSON, like this: [{"name":"tool_form_name", "arg":"tool_form_argument"}]
5. If you choose more than one tool, create an list. Like this: [{"name":"tool_form_name", "arg":"tool_form_argument"}, {"name":"tool_form_name", "arg":"tool_form_argument"}]
6. If you choose no tools, respond with ["None"].
7. Your response will be parsed in a computer program. Do not include any additional text in your response.
8. "Conversation" starts with the earliest message and ends with the most recent message.
9. If the latest message changes the subject of the conversation, even if an earlier message is still relevant, you may respond with ["None"].

Tools:
"""
				tools_list = self.ml.build_tools_list_from_available_modules()
				prompt += tools_list

				#Get the last three messages and add them to the prompt
				prompt += "Conversation:\n"
				last_three_messages = messages[-3:]
				for message in last_three_messages:
					prompt += str(message)+"\n"
				logging.info(prompt)
				message = [{'role': 'system', 'content': prompt}]
				logging.info(prompt)


				response = self.request(
					messages=message, 
					stop_event=stop_event, 
					tool_check=False,
					model=model,
					silent=True,
					response_label=response_label
					)

				logging.info("Tool form response: "+str(response))

				tools = self.handle_tool_checker_response(response)
				result = self.run_tools(tools, stop_event)

				if result:
					return_prompt = "Below is the response from the user's request. It is the information the user is requesting. Don't mention the existence of this information. Use it to continue the conversation. If the information is irrelevant to the conversation, then ignore it.\n\n"
					return_prompt += "\n"+result+"\n"

					return return_prompt
				
				else:
					logging.info("No data found.")
					return False

		return False

	def handle_tool_checker_response(self, response):
			#Parse JSON response
		data = None
		start_index = response.find('[')
		if start_index >= 0:
			end_index = response.find(']', start_index) + 1
			json_data = response[start_index:end_index]
			try:
				# Check if the input string matches the expected JSON format
				if not re.fullmatch(r'\[.*\]', json_data):
					# Input string does not match expected format
					logging.warning('Input is not valid JSON')
				else:
					# Attempt to load the input string as JSON
					try:
						data = json.loads(json_data)
						logging.info('Data:' + str(data))
					except json.decoder.JSONDecodeError as e:
						# Input string contains errors, attempt to fix them
						logging.error('JSONDecodeError:', e)
						
						# Search for keys with missing values
						match = json_data.search(json_data)
						if match:
							# Replace missing values with empty strings
							fixed_str = json_data[:match.end()] + '""' + json_data[match.end()+1:]
							logging.warning('Fixed input:', fixed_str)
							try:
								data = json.loads(fixed_str)
								logging.info('Data:'+ str(data))
							except json.decoder.JSONDecodeError:
								logging.error('Could not fix input')
						else:
							logging.error('Could not fix input')

				#data = json.loads(json_data)


			except json.decoder.JSONDecodeError as e:
				logging.error("JSONDecodeError: "+str(e))
				data = None
			if data and data[0] == "None":
				data = None
		else:
			logging.warning("No JSON data found in string.")

		logging.info("Tool form chosen: "+str(data))
		return data
	
	def run_tools(self, tools, stop_event):
		result = ""
		if tools:
			for tool in tools:
				for module in self.ml.get_available_modules():
					if "tool_form_name" in module:
						if module["tool_form_name"] == tool["name"]:
							print_text("Tool: ", "green")
							print_text(module["tool_form_name"] + " (" + tool['arg']+")", None, "\n\n")

							class_name = module["class_name"]
							chat_request_inner_hook_instances = self.ml.get_hook_instances()["Chat_request_inner"]
							for instance in chat_request_inner_hook_instances:
								if instance.__class__.__name__ == class_name.split(".")[-1]:
									logging.info("Found instance: "+instance.__class__.__name__)
									result += "["+instance.__class__.__name__+" response]\n"
									result += instance.main(tool['arg'], stop_event)+"\n\n"
		return result


	def stream_queue_sentences(self, arguments_dict):
		response = arguments_dict['response']
		text_stream = arguments_dict['text_stream']
		sentences = arguments_dict['sentences']
		sentence_queue_canceled = arguments_dict.get('sentence_queue_canceled', [False])
		sentence_queue_complete = arguments_dict.get('sentence_queue_complete', [False])
		return_text = arguments_dict['return_text']
		stop_event = arguments_dict['stop_event']
		sound_stop_event = arguments_dict['sound_stop_event']
		silent = arguments_dict['silent']
		model = arguments_dict['model']
		response_label = arguments_dict['response_label']

		collected_chunks = []
		collected_messages = []

		try:
			if not silent and response_label:
				print_text("Daisy ("+model+"): ", "blue", "", "bold")

			for chunk in response:
				if not sentence_queue_canceled[0]:
					if not stop_event.is_set():
						temp_sentences = []
						collected_chunks.append(chunk)
						chunk_message = chunk['choices'][0]['delta']
						collected_messages.append(chunk_message)
						text_stream[0] = ''.join([m.get('content', '') for m in collected_messages])
						logging.debug(text_stream[0])

						if not silent:
							if 'content' in chunk_message:
								print_text(chunk_message['content'])
						
						#Tokenize the text into sentences
						temp_sentences = self.csp.nltk_sentence_tokenize(text_stream[0])
						sentences[0] = temp_sentences  # put the sentences into the queue
					else:
						sentence_queue_canceled[0] = True
						logging.info("Sentence queue canceled")
						return
			print_text("\n\n")
		except requests.exceptions.ConnectionError as e:
			logging.error("stream_queue_sentences(): Request timeout. Check your internet connection.")
			sentence_queue_canceled[0] = True

		time.sleep(0.01)
		self.complete_sentence_queue(
			sentence_queue_complete, 
			text_stream, 
			return_text, 
			sound_stop_event)
		return


	def complete_sentence_queue(self, 
				 sentence_queue_complete, 
				 text_stream, 
				 return_text, 
				 sound_stop_event):
		sentence_queue_complete[0] = True
		return_text[0] = text_stream[0]
		sound_stop_event.set()
		logging.info("Sentence queue complete")