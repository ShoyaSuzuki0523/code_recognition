import numpy as np

def code_to_int(code):
  codelist = [
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

  y = np.empty(len(code))
  for i, c in enumerate(code):
    for j, l in enumerate(codelist):
      if l == c :
        y[i] = j

  return y

def int_to_code(int_list):
  codelist = [
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

  y = np.empty(len(int_list), dtype=str)
  for i, il in enumerate(int_list):
    for j, l in enumerate(codelist):
      if j == il :
        y[i] = l
  
  return y

y_output_path = "../output/y/"

codelist = [
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

for c in codelist:
  for i in range(1,10):
    test = np.loadtxt(y_output_path + c + "_" + str(i) + ".txt", dtype=str)
    test = code_to_int(test)
    print(test)
    test = int_to_code(test)
    print(test)