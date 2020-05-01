import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import json
import random
from sklearn.utils import shuffle

#模型精度：70%

#超参数
lr = 0.001
batch_size = 96
iters = 2000
view_1 = 3
view_2 = 5
view_3 = 5
#view_4 = 3
#view_5 = 3
num_filter_1 = 48
num_filter_2 = 64
num_filter_3 = 64
#num_filter_4 = 64
#num_filter_5 = 64
fc_neuron_num_1 = 512
#fc_neuron_num_out = 512
#------
use_bn = True
use_dropout = False
dropout = 0.2#每个元素被保留的概率
keep_prob = tf.placeholder(tf.float32)


n = 0
t = 0
def get_train_batch():
    global n
    global t
    m = n%5 +1
    s = t%10000 
    file_name = 'data_batch_' + str(m)
    with open(file_name,'rb') as f:
        batch = pickle.load(f,encoding='bytes')
        
    t += 100
    if s == 9900:
        n += 1
    try:
        x_batch = batch[b'data'][s:s+batch_size]
        y_batch = batch[b'labels'][s:s+batch_size]
    except:
        x_batch = batch[b'data'][s:-1]
        y_batch = batch[b'labels'][s:-1]
    return x_batch,y_batch

def get_test_batch():
    with open('test_batch','rb') as f:
        batch = pickle.load(f,encoding='bytes')

    return batch[b'data'],batch[b'labels']

def roll_pics(x_batch,y_batch):
    n =0
    img_batch = x_batch
    y__batch = y_batch
    for index,img in enumerate(x_batch):
        if n%15 == 0:
            img_batch = np.append(img_batch,x_batch[index][:,::-1])
            y__batch = np.append(y__batch,y_batch[index])
        n+=1
    img_batch = img_batch.reshape(-1,32,32,3)
    y__batch = y__batch.reshape(-1,10)
    #运用sklearn.utils的shuffle能够同时打乱两个数组，并且保持label对应关系
    x,y = shuffle(img_batch,y__batch)
    return x,y

def data_batch_pretreatment(img_batch,y_batch):
    img_batch = img_batch.reshape(-1,3,32,32)
    target_batch = img_batch.reshape(-1,32,32,3)
    for index,img in enumerate(img_batch):

        _r = img[0]
        _g = img[1]
        _b = img[2]
        
        img_r = Image.fromarray(_r)  
        img_g = Image.fromarray(_g)
        img_b = Image.fromarray(_b)

        img = Image.merge('RGB',(img_r,img_g,img_b))
        img = np.array(img)
        
        target_batch[index] = img     
    with tf.Session() as sess:
        y_batch = sess.run(tf.one_hot(y_batch,depth=10))
    return target_batch,y_batch

def conv2d(x,w,b,use_bn,strides=1):
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    x = tf.layers.batch_normalization(x,training=use_bn)

    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,k,k,1],padding='SAME')
    return x

def conv_net(x,weights,biases,use_bn,use_dropout):
    convs = []
    #activations = []
    #x = tf.reshape(x,[-1,32,32,3])
    
    conv_1 = conv2d(x,weights['wc1'],biases['bc1'],use_bn)
    convs.append(conv_1)
    #activations.append(conv_1)
    conv_1 = maxpool2d(conv_1)
    #lrn1 = tf.nn.lrn(conv_1,4,bias=1,alpha=0.001/9.0, beta=0.75)
    
    conv_2 = conv2d(conv_1,weights['wc2'],biases['bc2'],use_bn)
    convs.append(conv_2)
    #activations.append(conv_2)
    conv_3 = conv2d(conv_2,weights['wc3'],biases['bc3'],use_bn)
    convs.append(conv_3)
    #activations.append(conv_3)
    conv_3 = maxpool2d(conv_3)
    #lrn2 = tf.nn.lrn(conv_2,4,bias=1,alpha=0.001/9.0, beta=0.75)
    
    #conv_3 = conv2d(conv_2,weights['wc3'],biases['bc3'])
    '''
    conv_4 = conv2d(conv_3,weights['wc4'],biases['bc4'])
    conv_4 = maxpool2d(conv_4)

    conv_5 = conv2d(conv_4,weights['wc5'],biases['bc5'])
    conv_5 = maxpool2d(conv_5)
    '''
    flatten = tf.reshape(conv_3,[-1,8*8*num_filter_3])
    
    fc1 = tf.nn.relu(tf.matmul(flatten,weights['wf1'])+biases['bf1'])
    fc1= tf.layers.batch_normalization(fc1)
    #activations.append(fc1)
    if use_dropout:
        fc1 = tf.nn.dropout(fc1,keep_prob)
    '''
    fc2 = tf.nn.relu(tf.matmul(fc1,weights['wf2']) + biases['bf2'])
    if use_dropout:
        fc2 = tf.nn.dropout(fc2,keep_prob)
    '''
    out = tf.matmul(fc1,weights['out']) + biases['out']

    return out,convs

weights={'wc1':tf.Variable(tf.random.truncated_normal([view_1,view_1,1,num_filter_1],stddev=0.02)),
         'wc2':tf.Variable(tf.random.truncated_normal([view_2,view_2,num_filter_1,num_filter_2],stddev=0.02)/np.sqrt(num_filter_1/2)),
         'wc3':tf.Variable(tf.random.truncated_normal([view_3,view_3,num_filter_2,num_filter_3],stddev=0.02)/np.sqrt(num_filter_2/2)),
         #'wc4':tf.Variable(tf.random.truncated_normal([view_4,view_4,num_filter_3,num_filter_4],stddev=0.05)/np.sqrt(num_filter_3/2)),
         #'wc5':tf.Variable(tf.random.truncated_normal([view_5,view_5,num_filter_4,num_filter_5],stddev=0.05)/np.sqrt(num_filter_4/2)),
         'wf1':tf.Variable(tf.random.truncated_normal([8*8*num_filter_3,fc_neuron_num_1],stddev=0.04)/np.sqrt(num_filter_3/2)),
         #'wf2':tf.Variable(tf.random.truncated_normal([fc_neuron_num_1,fc_neuron_num_out],stddev=0.04)/np.sqrt(fc_neuron_num_1/2)),
         'out':tf.Variable(tf.random.truncated_normal([fc_neuron_num_1,10],stddev=1/192)/np.sqrt(192/2))
         }

biases={'bc1':tf.Variable(tf.zeros([num_filter_1])),
        'bc2':tf.Variable(tf.zeros([num_filter_2])+0.1),
        'bc3':tf.Variable(tf.zeros([num_filter_3])),
        #'bc4':tf.Variable(tf.zeros([num_filter_4])),
        #'bc5':tf.Variable(tf.zeros([num_filter_5])),
        'bf1':tf.Variable(tf.zeros([fc_neuron_num_1])+0.1),
        #'bf2':tf.Variable(tf.zeros([fc_neuron_num_out])+0.1),
        'out':tf.Variable(tf.zeros([10])),
        } 
x = tf.placeholder(tf.float32,[None,32,32,1])
y = tf.placeholder(tf.float32,[None,10])

pred,convs = conv_net(x,weights,biases,use_bn,use_dropout)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1)),dtype=tf.float32))
init = tf.global_variables_initializer()

if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(init)
        #Train
        for step in range(iters):
            x_batch,y_batch = get_train_batch()
            x_batch,y_batch = data_batch_pretreatment(x_batch,y_batch)
            
            
            #三通道变一通道
            #方法一（三通道中取一个通道)
            '''
            tmp_batch = np.zeros((batch_size,1,x_batch.shape[1],x_batch.shape[2]))
            for idx,img in enumerate(x_batch):
                tmp_batch[idx] = img[:,:,1].reshape(1,32,32)
            x_batch = tmp_batch.reshape(batch_size,32,32,1)
            '''
            #方法二
            tmp_batch = np.zeros((batch_size,1,32,32))
            for index,each in enumerate(x_batch):
                img = Image.fromarray(each)
                gray_img = img.convert('L')
                img_array = np.array(gray_img).reshape(1,32,32)
                tmp_batch[index] = img_array
            x_batch = tmp_batch.reshape(batch_size,32,32,1)
            
            #归一化
            #x_batch = x_batch/255
            
            #正则化(数据增强)
            #x_batch,y_batch = roll_pics(x_batch,y_batch)
            sess.run(optimizer,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            loss = sess.run(cost,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            acc = sess.run(accuracy,feed_dict={x:x_batch,y:y_batch,keep_prob:1.0})
            print('loss:',loss)
            print('accuracy:',acc,'\n')

            
            '''
            for idx,activation in enumerate(activations):
                plt.subplot(1,len(activations),idx+1)
                plt.title(str(idx+1) + "-layer")
                plt.hist(sess.run(activation,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout}).flatten())
                plt.pause(0.1)
            '''    
        #Evaluate
        use_dropout = False
        use_bn = False
        avg_acc_test = 0
        indx = 0
        test_x,test_y = get_test_batch()
        test_x,test_y = data_batch_pretreatment(test_x,test_y)
        
        '''
        tmp_batch = np.zeros((test_x.shape[0],1,test_x.shape[1],test_x.shape[2]))
        for idx,img in enumerate(test_x):
            tmp_batch[idx] = img[:,:,1].reshape(1,32,32)
        test_x = tmp_batch.reshape(test_x.shape[0],32,32,1)
        '''
        tmp_batch = np.zeros((test_x.shape[0],1,test_x.shape[1],test_x.shape[2]))
        for index,each in enumerate(test_x):
            img = Image.fromarray(each)
            gray_img = img.convert('L')
            img_array = np.array(gray_img).reshape(1,32,32)
            tmp_batch[index] = img_array
        test_x = tmp_batch.reshape(test_x.shape[0],32,32,1)
        
        for i in range(100):
            acc_test = sess.run(accuracy,feed_dict={x:test_x[indx:indx+99],y:test_y[indx:indx+99],keep_prob:dropout})
            indx += 100
            print('Test accuracy: ',acc_test)
            avg_acc_test += acc_test
            
        avg_acc_test = avg_acc_test/100
        print('Done! Average accuracy of test data is: ',avg_acc_test)
        
        #探索特征图
        plt.subplot(2,1,1)
        plt.axis('off')
        plt.imshow(test_x[10].reshape(32,32))

        plt.subplot(2,1,2)
        for idx,conv in enumerate(convs):
            maps_batches = sess.run(conv,feed_dict={x:test_x,y:test_y,keep_prob:1.0})
            maps = maps_batches[10]
            for chanel in range(maps.shape[-1]):
                plt.imshow(maps[:,:,chanel])
                plt.axis('off')
                plt.title('conv:'+str(idx+1)+'    chanel:'+str(chanel+1))
                plt.pause(0.1)
        

           
    








         
    
    
