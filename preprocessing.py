import numpy as np
import os
import random
import librosa
import matplotlib.pyplot as plt

class Preprocessing():

	def __init__(self,sound_path="./codes" , output_path="./output", sr=44100, fr=22050, fs=22, n_chroma=12, threshold=20):
		self.sound_path = sound_path
		self.x_output_path = output_path + "/x/"
		self.y_output_path = output_path + "/y/"
		self.sr = sr
		self.fr = fr
		self.fs = fs
		self.n_chroma = n_chroma
		self.threshold = threshold
		self.notes = [
			"C",
			"C#",
			"D",
			"D#",
			"E",
			"F",
			"F#",
			"G",
			"G#",
			"A",
			"A#",
			"B"
		]
		self.codeFiles = [
			"c",
			"c#",
			"d",
			"d#",
			"e",
			"f",
			"f#",
			"g",
			"g#",
			"a",
			"a#",
			"b",
			"cm",
			"c#m",
			"dm",
			"d#m",
			"em",
			"fm",
			"f#m",
			"gm",
			"g#m",
			"am",
			"a#m",
			"bm"
		]
	
	def compile_sound(self, train_range=[1,2,3,4,5,6,7,8], test_range=[9,10], close=True):
		for i, cf in enumerate(self.codeFiles):
			folder_path = self.sound_path + cf + "/"
			folder = os.listdir(folder_path)
			for j, f in enumerate(folder):
				x_output = self.x_output_path + cf + "_" + str(j + 1) + ".txt"
				y_output = self.y_output_path + cf + "_" + str(j + 1) + ".txt"
				file_path = folder_path + f

				y, sr = librosa.load(file_path, sr=sr)

				#無音のトリミング
				y, index = librosa.effects.trim(y=y, top_db=self.threshold)

				#特徴検出
				Chromas = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=self.fr, hop_length=self.fs, n_chroma=self.n_chroma)

				trans_Chromas = Chromas.T

				#教師データを定義
				y_data = np.full(trans_Chromas.shape[0], i)

				print("=============================================")
				print(file_path)
				print("Chromas :" + str(Chromas.shape))
				print("trans_Chromas" + str(trans_Chromas.shape))
				print("y_data:")
				print(y_data)
				print("=============================================")

				# if plot == True:
				# 	fl = len(Chromas[0])

				# 	fig = plt.figure()
				# 	x = np.arange(0, fl, 1)
				# 	for i, note in enumerate(self.notes) :
				# 		ax = fig.add_subplot(4,3,i + 1)
				# 		ax.plot(x, Chromas[i])
				# 		ax.set_title(note)
				# 		ax.set_ylim(0,1.1)

				# 	fig.subplots_adjust(hspace=0.4)
				# 	plt.show()

				np.savetxt(fname=x_output, X=trans_Chromas, fmt="%.18e")
				np.savetxt(fname=y_output, X=y_data, fmt="%s")
	
	def get_chromas(self):
		x_train = np.zeros(12)
		x_test = np.zeros(12)
		y_train = np.empty(1)
		y_test = np.empty(1)

		if close == True:
			test_range = train_range + test_range

		for c in self.codeFiles:
			for i in train_range:
				x_train = np.vstack([x_train, np.loadtxt(self.x_output_path + c + "_" + str(i) + ".txt")])
				y_train = np.hstack([y_train, np.loadtxt(self.y_output_path + c + "_" + str(i) + ".txt", dtype=str)])
				print("loadhing " + self.x_output_path + c + "_" + str(i) + ".txt" + " and " + self.y_output_path + c + "_" + str(i) + ".txt")
			for i in test_range:
				x_test = np.vstack([x_test, np.loadtxt(self.x_output_path + c + "_" + str(i) + ".txt")])
				y_test = np.hstack([y_test, np.loadtxt(self.y_output_path + c + "_" + str(i) + ".txt", dtype=str)])
				print("loadhing " + self.x_output_path + c + "_" + str(i) + ".txt" + " and " + self.y_output_path + c + "_" + str(i) + ".txt")

		x_train = np.delete(x_train, 0, 0)
		x_test = np.delete(x_test, 0, 0)
		y_train = np.delete(y_train, 0)
		y_test = np.delete(y_test, 0)

		return x_train, y_train, x_test, y_test