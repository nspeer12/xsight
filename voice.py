import pyttsx3


def speak(input):
	engine = pyttsx3.init()
	engine.say(input)
	engine.runAndWait()	
