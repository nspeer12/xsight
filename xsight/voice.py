import pyttsx3
import speech_recognition as sr
import pyaudio
import wave
from playsound import playsound
import numpy as np

def speak(input):
	engine = pyttsx3.init()
	engine.say(input)
	engine.runAndWait()


def listen(filename='tmp.wav'):
	r = sr.Recognizer()
	rec = sr.AudioFile(filename)
	with rec as source:
		audio = r.record(source)

	return r.recognize_google(audio)

def record_audio(seconds=5, filename='tmp.wav'):
	chunk = 1024  # Record in chunks of 1024 samples
	sample_format = pyaudio.paInt16  # 16 bits per sample
	channels = 2
	fs = 44100  # Record at 44100 samples per seconds

	p = pyaudio.PyAudio()  # Create an interface to PortAudio

	print('Recording')

	stream = p.open(format=sample_format,
			channels=channels,
			rate=fs,
			frames_per_buffer=chunk,
			input=True)

	frames = []  # Initialize array to store frames

	# Store data in chunks for 3 seconds
	for i in range(0, int(fs / chunk * seconds)):
	    data = stream.read(chunk)
	    frames.append(data)

	# Stop and close the stream 
	stream.stop_stream()
	stream.close()
	# Terminate the PortAudio interface
	p.terminate()

	print('Finished recording')

	# Save the recorded data as a WAV file
	wf = wave.open(filename, 'wb')
	wf.setnchannels(channels)
	wf.setsampwidth(p.get_sample_size(sample_format))
	wf.setframerate(fs)
	wf.writeframes(b''.join(frames))
	wf.close()


def increase_pitch(input_file='tmp.wav', output_file='output.wav'):
	wr = wave.open(input_file, 'r')
	# Set the parameters for the output file.
	par = list(wr.getparams())
	par[3] = 0  # The number of samples will be set by writeframes.
	par = tuple(par)
	ww = wave.open(output_file, 'w')
	ww.setparams(par)

	fr = 20
	sz = wr.getframerate()//fr  # Read and process 1/fr second at a time.
	# A larger number for fr means less reverb.
	c = int(wr.getnframes()/sz)  # count of the whole file
	shift = 100//fr  # shifting 100 Hz
	for num in range(c):	
		da = np.fromstring(wr.readframes(sz), dtype=np.int16)
		left, right = da[0::2], da[1::2]  # left and right channel
		lf, rf = np.fft.rfft(left), np.fft.rfft(right)
		lf, rf = np.roll(lf, shift), np.roll(rf, shift)
		lf[0:shift], rf[0:shift] = 0, 0
		nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
		ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
		ww.writeframes(ns.tostring())
	
	wr.close()
	ww.close()
		

def play_audio(filename='output.wav'):
	playsound(filename)


if __name__=='__main__':
	#record_audio()
	increase_pitch()
	play_audio()
	#speak(listen(filename='output.wav'))
