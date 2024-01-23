import os
import importlib.util
import logging
import yaml
import time
import threading
import concurrent.futures
import pubsub as pub
import queue
import json
import numpy as np
#import torch
#from scipy.spatial.distance import cosine

from ruamel.yaml import YAML
yaml = YAML()
yaml.allow_duplicate_keys = True

import daisy_llm.InputManager as im
import daisy_llm.CommandHandlers as commh
from .Text import print_text



class ModuleLoader:
	initialized = False
	
	def __init__(self, ch, 
		  configs_yaml="configs.yaml", 
		  modules=[]):
		self.ch = ch
		self.commh = commh.CommandHandlers(self)
		self.im = im.InputManager()
		self.configs_yaml = configs_yaml
		self.passed_modules = modules

		self.messaging_system = pub	

		if not ModuleLoader.initialized:
			ModuleLoader.initialized = True

			self.start_prompts = []
			self.hook_instances = {}
			self.loaded = False

			self.stop_event = threading.Event()
			self.thread = threading.Thread(target=self.update_configs_loop)

			# Load modules
			self.available_modules = []
			self.enabled_modules = []


			# Load enabled modules from config file
			with open(self.configs_yaml, 'r') as f:
				self.configs = yaml.load(f)


			

	def close(self):
		self.stop_event.set()
		self.thread.join()

	def start(self):
		self.thread.start()
		
	def get_hook_instances(self):
		return self.hook_instances
	
	def get_enabled_modules(self):
		return self.enabled_modules
		
	def get_available_modules(self):
		if not self.loaded:
			self.loaded = True
			logging.info("Updating modules...")

			self.enabled_modules = list(self.passed_modules) #Create a copy so that it does not get updated when appended to.

			try:
				with open(self.configs_yaml, 'r') as f:
					self.configs = yaml.load(f)
			except Exception as e:
				logging.warning(f"Failed to load configs.yaml: {str(e)}")

			if "enabled_modules" in self.configs:
				if self.configs["enabled_modules"]:

					#Add modules in configs.yaml to enabled_modules
					for enabled_module in self.configs["enabled_modules"]:
						if enabled_module not in self.enabled_modules: #Eliminates duplicates
							self.enabled_modules.append(enabled_module)

			#Set all modules as disabled and enable them as they are found
			for available_module in self.available_modules:
				available_module["enabled"] = False

			for module_name in self.enabled_modules:
				module_in_available = False

				# If module is in configs.yaml and available_modules, it is enabled.
				for available_module in self.available_modules:
					if module_name == available_module["class_name"]:
						available_module["enabled"] = True
						module_in_available = True


				# Add the module if it's NOT in available modules
				if not module_in_available:
					# Attempt to import the module, and handle exceptions.
					try:
						module = importlib.import_module(module_name, package=None)
					except ModuleNotFoundError as e:
						logging.warning(f"Failed to load {module_name} due to missing dependency: {str(e)}")
						continue  # Skip this module and proceed with the next one

					# Find all classes in the module, and extract their methods and initialization parameters.
					for name in dir(module):

						if name == module.__name__.split(".")[-1]:
							obj = getattr(module, name)
							if isinstance(obj, type):
								module_hook = getattr(obj, "module_hook", "")

								if module_hook and module_name:
									class_description = getattr(obj, "description", "No description.")
									tool_form_name = getattr(obj, "tool_form_name", None)
									tool_form_description = getattr(obj, "tool_form_description", None)
									tool_form_argument = getattr(obj, "tool_form_argument", None)

									# Add module_hook and enabled attributes to module dictionary
									module_dict = {}
									module_dict["class_name"] = module_name
									module_dict["description"] = class_description
									module_dict["module_hook"] = module_hook
									module_dict["enabled"] = True

									self.available_modules.append(module_dict)


			# Notify status of modules
			for available_module in self.available_modules:
				if available_module['enabled']:
					print_text("MODULE LOADED: ", "green", "", "italic")
					print_text(available_module['class_name']+" to "+available_module['module_hook'], None, "\n")

				else:
					print_text("MODULE REMOVED: ", "red", "", "italic")
					print_text(available_module['class_name']+" from "+available_module['module_hook'], None, "\n")

			#If config.yaml item is no longer available, set enabled to False
			for available_module in self.available_modules:
				if available_module["class_name"] not in self.enabled_modules:
					available_module["enabled"] = False
			self.build_hook_instances()
		return self.available_modules

		
	def build_hook_instances(self):
		
		# Create a new dictionary to keep track of updated hook instances
		updated_hook_instances = {}

		# Iterate over the modules in the order they appear
		for module_name in self.enabled_modules:
			for module in self.available_modules:
				
				if module['class_name'] == module_name:
					if module['module_hook'] not in updated_hook_instances:
						updated_hook_instances[module['module_hook']] = []

					# Check if the instance already exists in hook_instances and use it if found
					existing_instance = None

					#Get the module object
					module_class = importlib.import_module(module_name)
					obj = getattr(module_class, module_name.split(".")[-1])

					#Check if the module already exists
					if module['module_hook'] in self.hook_instances:
						for instance in self.hook_instances[module['module_hook']]:
							if isinstance(obj, type) and instance.__class__.__name__ == module_name.split(".")[-1]:
								existing_instance = instance
								break

					#If so, use it instead
					if existing_instance:
						instance = existing_instance
					else:
						for name in dir(module_class):
							if name == module_class.__name__.split(".")[-1]:
								if isinstance(obj, type):
									instance = obj(self)
									instance.ch = self.ch  # Add self.ch to the instance
									instance.ml = self  # Add self to the instance
								if hasattr(instance, "start") and callable(getattr(instance, "start")):
									instance.start()

					# Add the updated instance to the updated_hook_instances
					updated_hook_instances[module['module_hook']].append(instance)
					

					break



		# Close removed instances from hook_instances
		for hook in self.hook_instances:
			for instance in self.hook_instances[hook]:
				if hook not in updated_hook_instances or instance not in updated_hook_instances[hook]:


					if hasattr(instance, 'close') and callable(getattr(instance, 'close')):
						instance.close()

		#Replace existing object with the new one
		self.hook_instances = updated_hook_instances

	def update_configs_loop(self):
		last_modified_time = 0
		while True:
			current_modified_time = os.path.getmtime("configs.yaml")
			if current_modified_time > last_modified_time:
				self.loaded = False
				self.get_available_modules()

				last_modified_time = current_modified_time

			time.sleep(1)

	def start_update_configs_loop_thread(self):
		self.update_configs_loop_thread = threading.Thread(target=self.update_configs_loop)
		self.update_configs_loop_thread.start()

	def stop_update_configs_loop_thread(self):
		self.update_configs_loop_thread.stop()

	def enable_module(self, module_name):
		logging.info("Enabling module: " + module_name)
		with open(self.configs_yaml, 'r') as f:
			config = yaml.load(f)

		if module_name not in config['enabled_modules']:
			config['enabled_modules'].append(module_name)
			with open(self.configs_yaml, 'w') as f:
				yaml.dump(config, f)

			self.loaded = False
		else:
			logging.warning(module_name + " is already enabled.")
		time.sleep(0.5)
		return self.get_available_modules()

	def disable_module(self, module_name):
		logging.info("Disabling module: " + module_name)
		with open(self.configs_yaml, 'r') as f:
			config = yaml.load(f)

		if module_name in config['enabled_modules']:
			config['enabled_modules'].remove(module_name)
			with open(self.configs_yaml, 'w') as f:
				yaml.dump(config, f)

			self.loaded = False
		else:
			logging.warning(module_name + " is already disabled.")
		time.sleep(0.5)
		return self.get_available_modules()

	def process_main_start_instances(self):

		# Define a function that starts a new thread for a given hook instance
		def start_instance(instance):
			logging.info("Main_start: Running module: "+instance.__class__.__name__+" "+type(instance).__name__)
			future = executor.submit(instance.main)
			return future

		# Define a dictionary to keep track of running threads
		running_threads = {}
		stop_event = threading.Event()
		# Create the ThreadPoolExecutor outside the while loop
		with concurrent.futures.ThreadPoolExecutor() as executor:
			# Main loop that watches for changes to hook_instances["Main_start"]
			while True:
				logging.debug("Main_start: Checking for changes...")
				if list(running_threads.keys()):
					future_object = list(running_threads.values())[0]  # get the Future object from the dictionary
					if future_object.exception() is not None:  # check if the Future object has a raised exception
						runtime_error = future_object.exception()  # get the raised exception from the Future object
						logging.error("An error occurred: %s "+str(future_object.exception()))


				hook_instances = self.get_hook_instances()
				# Check if any new hook instances have been added or removed
				if "Main_start" in hook_instances:
					for instance in hook_instances["Main_start"]:
						for module in self.get_available_modules():
							if module['class_name'] == instance.__module__ and instance not in running_threads:
								if module['enabled']:
									future = executor.submit(start_instance, instance)
									running_threads[instance] = future
								else:
									future = running_threads[instance]
									future.cancel()
									del running_threads[instance]


				# Wait for some time before checking for updates again
				time.sleep(1)
		