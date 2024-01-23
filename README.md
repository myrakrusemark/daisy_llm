## 🌼 Daisy LLM Tools 🌼
Daisy is a Python platform designed to work with language model APIs such as OpenAI's GPT-3 and GPT-4. It includes a suite of classes and methods that can be used to converse with, extend capabilities, and augment the reasoning capacities of large language models.

### 🏁 Getting Started

NOTES:
Pytorch does not yet support python 3.12. This library uses pytorch in its commands process. You may modify the code to omit this step.
FFPROBE is required. You can install FFMPEG in Windows using this guide: https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

Install package
```
pip install daisy_llm
```

Import into your project and use. See main-example.py


Create ```config.py``` in your project with necessary information and enable desired modules to be loaded. See [Daisy-openai-chat](https://github.com/myrakrusemark/Daisy-openAI-chat) for sample project and a collection of modules to get started

### 🧰 Capabilities
Daisy accepts different types of user-developed "modules". A voice assistant module comes with [Daisy-openai-chat](https://github.com/myrakrusemark/Daisy-openAI-chat) as a "proof-of-concept". Possible configurations and apps built using Daisy could include:
  - Web apps
  - Conversational processing APIs
  - Computer vision interpretation
  - Autonomous initiation (with time awareness, Daisy could send a message or tool-form without user input)
  - Customer service IVR (which could be powerful with API tool-forms on the back-end).
  - Possibilities are endless, you only need to create what you want to see in the world. This platform makes that easier.

Keep in mind: Daisy is still in development. It has, and will, evolve significantly in the coming months as contributors enhance functinality by improving platform code, ading module hooks, and developing their own modules.


### 🛎️ Services
Daisy uses the following APIs for conversation processing:
  - Language model: OpenAI chatGPT
  - Speech-to-text (STT): AssemblyAI
  - Text-To-Speech(TTS): ElevenLabs (Quality), Google Cloud TTS (Cheap), Google Translate TTS (Free!) (Modules in [Daisy-openai-chat](https://github.com/myrakrusemark/Daisy-openAI-chat). Uses pyttsx3 by default)
  - Wake word (Local): Picovice Porcupine
  - Alternative local APIs are available and should be easily interchangeable if you choose to use them. In some cases, they can be switched out as modules. In every other case, a code hook can be added to make it interchangeable.


### 🌇 Background
I have been eager to have a conversation with chatGPT using my voice. I used to search daily for a program that could exchange between speech recognition and TTS for a real human-like conversation, but it was not until recently that I discovered one.

So of course I began making what I wanted in the world. I started working on a voice recognition script for chatGPT. It began with simple requests, such as incorporating a request to openAI API and routing the speech recognition output. Since then the project evolved into a platform for building applications, opening the door for infinite potential.

Some people argue that text models and AI are not thinking, but just using heuristics. However, when we examine ourselves, we too are simply a collection of learned behavior and responses. Although GPT may not be perfect, it is important to reflect on ourselves and determine how much better we truly are.

### 🤝 Compatibility
This software is designed to run on Windows and Linux.


### ✅ To-Do
- LLMs (API or local) as modules