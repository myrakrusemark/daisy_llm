import asyncio
import sys
import logging
import platform
import pvporcupine
import threading
import os
import system_modules.ChatSpeechProcessor as csp
import system_modules.ConnectionStatus as cs
import system_modules.ContextHandlers as ch
import system_modules.SoundManager as sm
import system_modules.Chat as chat
import system_modules.LoadTts as loadtts
import modules.Porcupine as porcupine
import modules.DaisyMethods as dm
import ModuleLoader as ml


import modules.RgbLed as led

class Daisy:
	description = "Provides a user flow for Chat"
	module_hook = "Main_start"

	def __init__(self):
		self.daisy_stop_event = threading.Event()
		self.awake_stop_event = threading.Event()

		self.csp = csp.instance
		self.cs = cs.instance
		self.ch = ch.instance
		self.sounds = sm.instance
		self.chat = chat.instance
		self.dm = dm.instance
		self.ml = ml.instance

		self.led = led.instance

		self.internet_warning_logged = False
		self.tts = None



	def close(self):
		self.daisy_stop_event.set()


	def main(self):
		self.sounds.play_sound("beep", 0.5)
		print("DAISY")

		# Create the TtsThread instance and start it in time for when its needed
		tts_thread = loadtts.LoadTts(self)
		tts_thread.start()

		print("TTS LOADED", self.tts)

		while not self.daisy_stop_event.is_set():
			self.awake_stop_event.clear()

			if self.cs.check_internet():
				# If internet connection is restored, log a message
				if self.internet_warning_logged:
					logging.info('Internet connection restored!')
					self.internet_warning_logged = False

				# Detect a wake word before listening for a prompt
				awoken = False

				self.led.turn_on_color(0, 100, 0)  # Solid Green

				try:
					# Listen for Porcupine wake word
					awoken = self.csp.listen_for_wake_word(self.daisy_stop_event)
				except Exception as e:
					# Catch the exception and handle it
					logging.error(f"Error initializing Porcupine: {e}")
					continue

				if awoken:
					self.led.breathe_color(100, 100, 100)  # Breathe Blue

					sleep_word_detected = False

					#HOOK: Daisy_wake
					try:
						import ModuleLoader as ml
						hook_instances = ml.instance.hook_instances
						if "Daisy_wake" in hook_instances:
							Daisy_wake_instances = hook_instances["Daisy_wake"]
							for instance in Daisy_wake_instances:
								logging.info("Running Daisy_start module: "+type(instance).__name__)
								response_text = instance.main()
					except Exception as e:
						logging.warning("Daisy_wake Hook: "+str(e))

					daisy_cancel_thread = threading.Thread(target=self.dm.daisy_cancel, args=(self.daisy_stop_event, self.awake_stop_event))
					daisy_cancel_thread.start()
					#self.dm.set_cancel_loop(False)

					while not self.daisy_stop_event.is_set():
						if not self.awake_stop_event.is_set():
								self.led.breathe_color(0, 0, 100)  # Breathe Blue
								stt_text = self.csp.stt(self.awake_stop_event, 30) #30s timeout

								self.led.breathe_color(100,0,100)  # Breathe Blue #NEEDS CANCEL LOOP

								self.ch.add_message_object('user', stt_text)

								if self.awake_stop_event.is_set():
									self.sounds.play_sound_with_thread('end', 1.0)
									break

								sound_stop_event = threading.Event()
								self.sounds.play_sound_with_thread('waiting', 0.2, self.awake_stop_event, sound_stop_event)
								print("TTS:", self.tts)
								text = self.chat.request(self.ch.get_context_without_timestamp(), self.awake_stop_event, sound_stop_event, self.tts)
								if not text:
									break

								self.ch.add_message_object('assistant', text)

								self.chat.display_messages()
								if self.awake_stop_event.is_set():
									self.sounds.play_sound_with_thread('end', 1.0)
									break

								self.led.breathe_color(100, 100, 100)  # Breathe White

						else:
							daisy_cancel_thread.join()
							break
			else:
				# Log a warning message if there is no internet connection and the warning hasn't been logged yet
				if not self.internet_warning_logged:
					self.led.turn_on_color(100, 0, 0)  # Solid Red
					logging.warning('No Internet connection. When a connection is available the script will automatically re-activate.')
					self.internet_warning_logged = True