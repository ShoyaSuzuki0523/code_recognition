import numpy as np
import os
import random
import librosa
import matplotlib.pyplot as plt

x_output_path = "./output/x/"
y_output_path = "./output/y/"
sr = 44100 #ファイルのサンプリングレート
fr = 22050 #フレーム長
fs = int(fr/100) #フレームシフト
n_chroma = 12 #1オクターブ間の音階数
threshold = 20 #無音とみなすしきい値(db)

notes = [
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

codeFiles = [
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

for i, cf in enumerate(codeFiles):
  folder_path = "./codes/" + cf + "/"
  folder = os.listdir(folder_path)
  
  for j, f in enumerate(folder):
    x_output = x_output_path + cf + "_" + str(j + 1) + ".txt"
    y_output = y_output_path + cf + "_" + str(j + 1) + ".txt"
    file_path = folder_path + f

    y, sr = librosa.load(file_path, sr=sr)

    #無音のトリミング
    y, index = librosa.effects.trim(y=y, top_db=threshold)

    #特徴検出
    Chromas = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=fr, hop_length=fs, n_chroma=n_chroma)

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

    # fl = len(Chromas[0])

    # fig = plt.figure()
    # x = np.arange(0, fl, 1)
    # for i, note in enumerate(notes) :
    #   ax = fig.add_subplot(4,3,i + 1)
    #   ax.plot(x, Chromas[i])
    #   ax.set_title(note)
    #   ax.set_ylim(0,1.1)

    # fig.subplots_adjust(hspace=0.4)
    # plt.show()

    np.savetxt(fname=x_output, X=trans_Chromas, fmt="%.18e")
    np.savetxt(fname=y_output, X=y_data, fmt="%s")