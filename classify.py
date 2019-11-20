from PIL import Image
import json
import os
import numpy as np
import tensorflow as tf
from train_model import *

with open('weights.json') as f:
    weights = json.load(f)
    for key,value in weights.items():
        weights[key] = tf.cast(np.array(value),dtype=tf.float32)

with open('biases.json') as f:
    biases = json.load(f)
    for key,value in biases.items():
        biases[key] = tf.cast(np.array(value),dtype=tf.float32)

def color2gray(img_array):
    img = Image.fromarray(img_array)
    gray_img = img.convert('L')
    gray_img_array = np.array(gray_img)
    return (gray_img_array.reshape(1,32,32,1)).astype('float32')

def predict(img):
    categories = ['飞机','汽车','鸟','猫','鹿','狗','青蛙','马','船','卡车']
    use_bn = True
    use_dropout = False

    pred,_ = conv_net(img,weights,biases,use_bn,use_dropout)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        out = sess.run(pred)
    index = np.argmax(out)
    name = categories[index]

    return name

if __name__ == '__main__':
    while True:
        img_name = input('请把图片移动到pictures文件夹，并输入图片名字(不用包含扩展名):\n')
        file = r'C:\Users\tensorflow\Desktop\tensorflow\cifar-10\pictures\\' + img_name
        if not os.path.exists(file+'.jpg') and not os.path.exists(file+'.png'):
            print('你输入的图片不存在，请重新输入')
            continue
        try:
            image = Image.open(file+'.jpg')
        except:
            image = Image.open(file+'.png')

        image = image.resize((32,32),Image.ANTIALIAS)
        small_img = np.array(image).reshape(32,32,-1)
        img = color2gray(small_img)
        name = predict(img)

        print('你输入的图片是:  ',name)

        
    
