import time
import re
import string
import pyttsx3
import requests
import logging
import yaml
import nltk.data
import queue
import time
import queue
import requests
import threading
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor

from .SoundManager import SoundManager
from .Text import print_text, delete_last_lines
from .LoadTts import LoadTts



class ChatSpeechProcessor:
	description = "A class that handles speech recognition and text-to-speech processing for a chatbot."

	def __init__(self, ml):
		# Set up AssemblyAI API key and websocket endpoint
		self.uri = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

		# Define global variables
		self.tts_speed = 1.0
		self.tts = None

		self.ml = ml
		self.result_str = ""
		self.new_result_str = ""
		self.result_received = False
		self.sounds = SoundManager()
		self.engine = pyttsx3.init()

		# Download the tokenizer resource if it's not already done.
		resource_url = 'tokenizers/punkt/english.pickle'
		try:
			nltk.data.find(resource_url)
		except LookupError:
			nltk.download('punkt')
		self.tokenizer = nltk.data.load(resource_url)
		
		self.elapsed_time = 0
		self.timeout_seconds = 0

		self.tts_queue = queue.Queue()  # create a queue to hold the tts_sounds
		self.tts_queue_complete = [False]	# use a list to make response_complete mutable

		self.threads = []  # keep track of all threads created

		self.initialize_tts(self.ml)



	def initialize_tts(self, ml):
		t = LoadTts(self, ml)
		t.start()
		#t.join()

	def speak_tts(self, arguments_dict):
		text = arguments_dict.get('text')
		stop_event = arguments_dict.get('stop_event', None)
		sound_stop_event = arguments_dict.get('sound_stop_event', None)

		with open("configs.yaml", "r") as f:
			configs = yaml.safe_load(f)
			self.tts_speed = configs["TTS"]["speed"]

		audio = self.create_tts_audio(text)
		self.sounds.play_sound(audio, 1.0, stop_event, sound_stop_event, self.tts_speed)

	def create_tts_audio(self, text):
		audio = self.tts.create_tts_audio(text)
		return audio

	def queue_and_tts_sentences(
			self, 
			tts, 
			sentences, 
			sentence_queue_canceled, 
			sentence_queue_complete, 
			stop_event, 
			sound_stop_event=None
			):

		with ThreadPoolExecutor(max_workers=2) as executor:
			arguments = {
				'tts':tts, 
				'sentences':sentences, 
				'sentence_queue_complete':sentence_queue_complete, 
				'sentence_queue_canceled':sentence_queue_canceled, 
				'tts_queue_complete':self.tts_queue_complete,
				'tts_queue':self.tts_queue,
				'stop_event':stop_event, 
			}
			executor.submit(self.queue_tts_from_sentences, arguments)

			arguments = {
				'play_tts_queue':self.play_tts_queue, 
				'tts_queue':self.tts_queue, 
				'sentence_queue_complete':sentence_queue_complete, 
				'sentence_queue_canceled':sentence_queue_canceled, 
				'tts_queue_complete':self.tts_queue_complete,
				'stop_event':stop_event,
				'sound_stop_event':sound_stop_event, 
			}
			executor.submit(self.play_tts_queue, arguments)
		return
		  
	def queue_tts_from_sentences(self, arguments_dict):
		tts = arguments_dict['tts']
		sentences = arguments_dict['sentences']
		sentence_queue_canceled = arguments_dict.get('sentence_queue_canceled', [False])
		sentence_queue_complete = arguments_dict.get('sentence_queue_complete', [False])
		tts_queue_complete = arguments_dict['tts_queue_complete']
		tts_queue = arguments_dict['tts_queue']
		stop_event = arguments_dict['stop_event']

		tts_queue_complete[0] = False
		sentences_length = 1

		def queue_tts_items(index):
			queued_sentence = temp_sentences[index]
			logging.info("Queued sentence: " + queued_sentence)

			try:
				tts_queue.put(self.create_tts_audio(queued_sentence))
			except requests.exceptions.HTTPError as e:
				self.tts("HTTP Error. Error creating TTS audio. Please check your TTS account.")
				logging.error(f"HTTP Error: {e}")
			except requests.exceptions.ConnectionError as e:
				self.tts("Connection Error. Error creating TTS audio. Please check your TTS account.")
				logging.error(f"Connection Error: {e}")


		while not stop_event.is_set():
			temp_sentences = sentences[0]
			index = 0

			if not sentences[0]:
				continue

			if len(temp_sentences) > sentences_length:
				logging.debug("Sentences: " + str(sentences))
				sentence_length_difference = len(temp_sentences) - sentences_length

				sentences_length = len(temp_sentences)
				


				for i in range(sentence_length_difference):
					if not sentence_queue_canceled[0]:
						index = (sentence_length_difference-i+1) * -1
						queue_tts_items(index)

			elif sentence_queue_complete[0]:

				#Play a single sentence response
				if len(sentences[0]) == 1:
					logging.info("Single sentence response")
					queued_sentence = sentences[0][0]
					logging.info("Queued sentence: "+queued_sentence)

					try:
						if not sentence_queue_canceled[0]:
							tts_queue.put(self.create_tts_audio(queued_sentence))
					except requests.exceptions.HTTPError as e:
						self.tts("HTTP Error. Error creating TTS audio. Please check your TTS account.")
						logging.error(f"HTTP Error: {e}")

					tts_queue_complete[0] = True
					logging.info("TTS queue complete: single sentence response")
					return

				#Play the very last sentence
				elif len(sentences[0]) == len(temp_sentences):
					if len(sentences[0][-1]) == len(temp_sentences[-1]):
							logging.debug("last sentence...")

							queue_tts_items(-1)

							tts_queue_complete[0] = True
							logging.info("TTS queue complete")
							return

				#All tts items used
				if tts_queue.empty():
					tts_queue_complete[0] = True
					logging.info("TTS queue complete")
					return

			if sentence_queue_canceled[0] or stop_event.is_set():
				tts_queue_complete[0] = True
				while not tts_queue.empty(): #Empty out the TTS queue so no sounds linger
					tts_queue.get()
				logging.info("TTS queue canceled")
				return
			time.sleep(0.5) #Wait juuuust a bit to prevent sentence overlap
				



	def play_tts_queue(self, arguments_dict):
		tts_queue = arguments_dict['tts_queue']
		sentence_queue_canceled = arguments_dict.get('sentence_queue_canceled', [False])
		tts_queue_complete = arguments_dict['tts_queue_complete']
		stop_event = arguments_dict['stop_event']
		sound_stop_event = arguments_dict['sound_stop_event']
		
		tts = []

		# Wait for tts to be generated
		while tts_queue.empty() and not sentence_queue_canceled[0]:
			time.sleep(0.01)

		# Play tts
		while not stop_event.is_set():
			tts = None


			if tts_queue.qsize():
				tts = tts_queue.get(block=True, timeout=0.01)  # get tts from the queue

				#Stop voice assistant "waiting" sound
				if sound_stop_event:
					sound_stop_event.set()
			
				if tts:
					
					# Define global variables
					with open("configs.yaml", "r") as f:
						configs = yaml.safe_load(f)
						if "TTS" in configs:
							if "speed" in configs["TTS"]:
								self.tts_speed = configs["TTS"]["speed"]
					self.sounds.play_sound(tts, 1.0, stop_event, None, self.tts_speed)
			elif tts_queue_complete[0]:
				logging.info("TTS play queue complete")
				return
			
	def stt(self, stop_event, timeout=30):
		# Create a recognizer object
		recognizer = sr.Recognizer()

		# Use the default system microphone as the audio source
		with sr.Microphone() as source:
			# Adjust for ambient noise levels
			recognizer.adjust_for_ambient_noise(source)

			# Listen for the user's speech input until the stop event is set
			while not stop_event.is_set():
				self.sounds.play_sound_with_thread('alert')

				# Create the stop event for the timer thread
				timer_stop_event = threading.Event()

				# Start the timer thread
				timer_thread = threading.Thread(target=self.display_timer, args=(timeout, timer_stop_event))
				timer_thread.start()

				try:
					# Listen for the user's speech
					audio = recognizer.listen(source=source, timeout=timeout)

					# Stop the timer thread
					timer_stop_event.set()
					timer_thread.join()

					# Use the recognizer to transcribe the speech
					text = recognizer.recognize_google(audio)
					print_text(text, None, "\n")
					return text
				except sr.UnknownValueError:
					print("Unable to transcribe the audio")
				except sr.RequestError as e:
					print("Error occurred during transcription: {}".format(e))

			return None


	def display_timer(self, timeout):
		start_time = time.time()
		while time.time() - start_time < timeout:
			if stop_event.is_set():
				break
			time.sleep(1)

	def display_timer(self, timeout, stop_event):
		for seconds_remaining in range(timeout, -1, -1):
			
			if stop_event.is_set():
				return
			
			print_text(f"\r("+str(seconds_remaining)+"/"+str(timeout)+" seconds) You: ", "blue", end="")
			time.sleep(1)



	def remove_non_alpha(self, text):
		"""Removes all non-alphabetic characters (including punctuation and numbers) from a string and returns the modified string in lowercase."""
		if text:
			# Log a debug message with the input string
			logging.debug(f'Removing non-alpha characters from string: {text}')

			# Use regular expression to replace non-alphanumeric characters with empty string
			text = re.sub(r'[^a-zA-Z]+', '', text)

			# Log a debug message with the modified string
			logging.debug(f'Filtered text: {text}')

			# Return the modified string
			return text.lower()
		else:
			return False
		
	def nltk_sentence_tokenize(self, text, language="english"):
		"""Splits a string into sentences using the NLTK sentence tokenizer."""
		# Log a debug message with the input string
		logging.debug(f'Tokenizing string: {text}')

		# Split the string into sentences
		sentences = nltk.sent_tokenize(text, language)

		# Log a debug message with the modified string
		logging.debug(f'Tokenized sentences: {sentences}')

		# Return the modified string
		return sentences

