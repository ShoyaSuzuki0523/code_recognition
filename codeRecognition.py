import numpy as np
import random
import librosa
import matplotlib.pyplot as plt

sr = 44100 #ファイルのサンプリングレート
fr = 22050 #フレーム長
fs = int(fr/100) #フレームシフト
n_chroma = 12 #1オクターブ間の音階数
threshold = 20 #無音とみなすしきい値(db)
sample_length = 1520 #音声サンプルの長さ

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

for cf in codeFiles:
  file_path = "./codes/" + cf + "_4.wav"
  output = "./output/" + cf + "_4.txt"

  y, sr = librosa.load(file_path, sr=sr)

  #無音のトリミング
  y, index = librosa.effects.trim(y=y, top_db=threshold)

  #特徴検出
  Chromas = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=fr, hop_length=fs, n_chroma=n_chroma)
  cut_Chromas = np.empty((12,1520))

  #音声サンプルの数をそろえる
  for i, c in enumerate(Chromas):
    if len(c) <= sample_length:
      #padding
      pad = sample_length - len(c)
      cut_Chromas[i] = np.append(c, np.zeros(pad, dtype=np.float64))
    else:
      #trimming
      cut_Chromas[i] = c[0:sample_length]

  print(cut_Chromas)
  print(cut_Chromas.shape)

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

  np.savetxt(fname=output, X=cut_Chromas, fmt="%.18e")

  # csvファイルとして出力
  # np.savetxt(frame=output, x=Chromas, fmt="%.18e", delimiter=",")