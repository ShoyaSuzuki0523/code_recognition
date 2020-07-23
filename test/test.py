import numpy as np

class testClass():
  def test(self):
    print("引数ナシ")

  def test(self, num1):
    print("引数１つ")

  def test(self, num1, num2):
    print("引数２つ")

testClass = testClass()
testClass.test()
testClass.test(1)
testClass.test(1,2)