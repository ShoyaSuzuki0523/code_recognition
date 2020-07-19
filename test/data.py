import numpy as np
import os

x_output_path = "../output/x/"
y_output_path = "../output/y/"
x_list = os.listdir(x_output_path)
y_list = os.listdir(y_output_path)

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

x_train = np.zeros(12)
x_test = np.zeros(12)
y_train = np.full(1, 3)
y_test = np.full(1, 3)

for c in codeFiles:
  for i in range(1,8):
    x_train = np.vstack([x_train, np.loadtxt(x_output_path + c + "_" + str(i) + ".txt")])
    y_train = np.hstack([y_train, np.loadtxt(y_output_path + c + "_" + str(i) + ".txt", dtype=str)])
    print("loadhing " + x_output_path + c + "_" + str(i) + ".txt" + " and " + y_output_path + c + "_" + str(i) + ".txt")
  for i in range(9,10):
    x_test = np.vstack([x_test, np.loadtxt(x_output_path + c + "_" + str(i) + ".txt")])
    y_test = np.hstack([y_test, np.loadtxt(y_output_path + c + "_" + str(i) + ".txt", dtype=str)])
    print("loadhing " + x_output_path + c + "_" + str(i) + ".txt" + " and " + y_output_path + c + "_" + str(i) + ".txt")

x_train = np.delete(x_train, 0, 0)
x_test = np.delete(x_test, 0, 0)
y_train = np.delete(y_train, 0)
y_test = np.delete(y_test, 0)

print("x_test:" + str(x_test.shape))
print(x_test)
print("y_test:" + str(y_test.shape))
print(y_test)
print("x_train:" + str(x_train.shape))
print(x_train)
print("y_train:" + str(y_train.shape))
print(y_train)