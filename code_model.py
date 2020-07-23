import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#自作パッケージ
from data_manager import DataManager as dm

#各種パラメータ

batch_size = 128  # 訓練データを128ずつのデータに分けて学習させる
num_classes = 24 # 分類させる数。今回のコードの総数は24種類
epochs = 20 # 訓練データを繰り返し学習させる数
x_output_path = "./output/x/"
y_output_path = "./output/y/"

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

# クロマベクトルのデータとその教師データを取得する
dm = dm()
train_range = [1,2,3,4,5,6,7,8,9,10]
test_range = [1,2,3,4,5,6,7,8,9,10]
x_train, y_train, x_test, y_test = dm.get_chromas(train_range=train_range, test_range=test_range)

# y(教師データ)にはコードの名前が文字列として入っているが、Kerasで扱いやすい形(0 or 1)に変換する
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# モデルの作成
model = Sequential()
model.add(Dense(48, activation='sigmoid', input_shape=(None,12)))
model.add(Dense(48, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
 optimizer=RMSprop(),
 metrics=['accuracy'])

model.summary()

# 学習
history = model.fit(x_train, y_train,
 batch_size=batch_size,
 epochs=epochs,
 verbose=1,
 validation_data=(x_test, y_test))

# 評価はevaluateで行う
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 結果をプロットする
fig, (loss_plot, accuracy_plot) = plt.subplots(ncols=2)

accuracy_plot.plot(history.history['accuracy'])
accuracy_plot.plot(history.history['val_accuracy'])
accuracy_plot.set_title('Model accuracy')
accuracy_plot.set_ylabel('Accuracy')
accuracy_plot.set_xlabel('Epoch')
accuracy_plot.set_ylim(0, 1.00)
accuracy_plot.legend(['Train', 'Test'], loc='upper left')

loss_plot.plot(history.history['loss'])
loss_plot.plot(history.history['val_loss'])
loss_plot.set_title('Model loss')
loss_plot.set_ylabel('Loss')
loss_plot.set_xlabel('Epoch')
loss_plot.legend(['Train', 'Test'], loc='upper left')

fig.tight_layout()
plt.show()