import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

with open('data_batch_1','rb') as f:
    batch_1 = pickle.load(f,encoding='bytes')

xs = batch_1[b'data']
ys = batch_1[b'labels']

while True:
    sample_num = random.randint(1,10000)
    x = xs[sample_num].reshape(3,32,32)
    y = ys[sample_num]

    categories = ['飞机','汽车','鸟','猫','鹿','狗','青蛙','马','船','卡车']
    name = categories[y]
    
    print(name)

    x_r = x[0]
    x_g = x[1]  #array
    x_b = x[2]
    
    ir = Image.fromarray(x_r)
    ig = Image.fromarray(x_g)  #from array to img
    ib = Image.fromarray(x_b)

    img = Image.merge('RGB',(ir,ig,ib)) #merge,合并
    img.show()
    input()

