from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, Add, Subtract
from keras.layers import Activation
import datagenrator as dg
import bicubic_int_layer as btl
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import numpy as np
import math
import csv
import time
from skimage.io import imread, imsave
from skimage.measure import compare_psnr, compare_ssim
import matplotlib.pyplot as plt
import cv2
from imagecodecs import jpeg8_encode,jpeg8_decode
import keras
from tqdm import tqdm
import keras.backend as K
from keras import layers
from keras import optimizers
import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,Add
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, advanced_activations
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add, Conv2DTranspose
from keras.layers.advanced_activations import PReLU
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import os, glob, sys, threading
import scipy.io
from scipy import ndimage, misc
import numpy as np
import re
import math
from keras.models import load_model
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave, imshow
import tensorflow as tf
import tensorflow.compat.v2 as tf_v2
import tensorflow.keras as keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
session = InteractiveSession(config=config)








parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ebrcnn_expdecay_same_likecom_qf5', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
#parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
#parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


#save_dir = os.path.join('VDSR_sus',args.model+'_') 

save_dir = os.path.join('result_analysis',args.model)

def fsrcnn_enhancement():
#    adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.08798916064, amsgrad=False)
    inpt = Input(shape=(None,None,1),name='input_com')

#comcnnlayer 1
    model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_uniform',name='layer1')(inpt)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL1_fsr')(model)

#    model=Activation('PReLU',name='reluL1_com')(model)

    model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform',name='layer2',)(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL2_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer3')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL3_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer4')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL4_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer5')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL5_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer6')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL6_fsr')(model)

    model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_uniform',name='layer7')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL7_fsr')(model)

    model1 = Conv2DTranspose(1, (9, 9), strides=(2, 2), padding='same',name='layer8')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer9')(model1)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer9_bn')(model) 
    model=Activation('relu',name='reluL9_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer10')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer10_bn')(model) 
    model=Activation('relu',name='reluL10_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer11')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer11_bn')(model) 
    model=Activation('relu',name='reluL11_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer12')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer12_bn')(model) 
    model=Activation('relu',name='reluL12_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer13')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer13_bn')(model) 
    model=Activation('relu',name='reluL13_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer14')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer14_bn')(model) 
    model=Activation('relu',name='reluL14_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer15')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer915_bn')(model) 
    model=Activation('relu',name='reluL15_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer16')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer16_bn')(model) 
    model=Activation('relu',name='reluL16_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer17')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer17_bn')(model) 
    model=Activation('relu',name='reluL17_fsr')(model)
    model = Conv2D(1, (3,3), padding='same', kernel_initializer='he_uniform',name='layer18')(model)
    model= Add(name = 'add_layer')([model1, model])

    output_img = model
#    comcnn_upscale=btl.UpSampling2DBilinear(40,40)(comcnn_layer3)
    model = Model(inpt, output_img)
#    model_comcnn.compile(optimizer=adam, loss = 'mean_squared_error')
    return model



def train_datagen(epoch_iter=3200,epoch_num=50,batch_size=64):
    while(True):
#        n_count = 0
#        if n_count == 0:
#            #print(n_count)
        [xs1,xs2]=dg.jpegqf5_same_bpp_fsrcnn_enhancement()
        #Standarize the data to mean 0 and variance 1
        #We take mean & variance of the data set
        #This is called feature wise scaling
#        xs_n=xs-xs.mean()
#        xs_n=xs/xs.std()
        
#            assert len(xs)%args.batch_size ==0, \
#            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
#            xs = xs.astype('float32')/255.0
        indices = list(range(xs1.shape[0]))
#            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs2[indices[i:i+batch_size]]
                batch_y = xs1[indices[i:i+batch_size]]
                
                yield batch_x, batch_y
                




def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf5_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*ebrcnn_expdecay_same_likecom_qf5_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)
     
def exp_decay(epoch):
   initial_lrate = args.lr
   k = 0.088
#   k = 0.0183
   lr = initial_lrate * math.exp(-k*epoch)
   log('current learning rate is %2.8f' %lr)
   return lr

if __name__ == '__main__':
    # model selection
    model_fsrcnn=fsrcnn_enhancement()
    model_fsrcnn.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf5_%03d.hdf5'%initial_epoch), compile=False)
    
    # compile the model
    model_fsrcnn.compile(optimizer=Adam(0.001), loss='mean_squared_error')
#    adam_rec=optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=False)
#    sgd = SGD(lr=0.1, momentum=0.9,clipvalue=0.005, nesterov=False)
#    model_rec.compile(optimizer=adam_rec, loss='mean_squared_error')
#    model_rec.compile(optimizer=adam_rec, loss='mean_squared_error')
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf5_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log_ebrcnn_expdecay_same_likecom_qf5.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(exp_decay)
    
    history = model_fsrcnn.fit_generator(train_datagen(),
                steps_per_epoch=3200, epochs=50, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])
    
    
    
#########################trained for qf15####################################################################  



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ebrcnn_expdecay_same_likecom_qf15', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
#parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
#parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


#save_dir = os.path.join('VDSR_sus',args.model+'_') 

save_dir = os.path.join('result_analysis',args.model)

def fsrcnn_enhancement():
#    adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.08798916064, amsgrad=False)
    inpt = Input(shape=(None,None,1),name='input_com')

#comcnnlayer 1
    model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_uniform',name='layer1')(inpt)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL1_fsr')(model)

#    model=Activation('PReLU',name='reluL1_com')(model)

    model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform',name='layer2',)(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL2_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer3')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL3_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer4')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL4_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer5')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL5_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer6')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL6_fsr')(model)

    model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_uniform',name='layer7')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL7_fsr')(model)

    model1 = Conv2DTranspose(1, (9, 9), strides=(2, 2), padding='same',name='layer8')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer9')(model1)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer9_bn')(model) 
    model=Activation('relu',name='reluL9_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer10')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer10_bn')(model) 
    model=Activation('relu',name='reluL10_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer11')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer11_bn')(model) 
    model=Activation('relu',name='reluL11_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer12')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer12_bn')(model) 
    model=Activation('relu',name='reluL12_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer13')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer13_bn')(model) 
    model=Activation('relu',name='reluL13_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer14')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer14_bn')(model) 
    model=Activation('relu',name='reluL14_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer15')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer915_bn')(model) 
    model=Activation('relu',name='reluL15_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer16')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer16_bn')(model) 
    model=Activation('relu',name='reluL16_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer17')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer17_bn')(model) 
    model=Activation('relu',name='reluL17_fsr')(model)
    model = Conv2D(1, (3,3), padding='same', kernel_initializer='he_uniform',name='layer18')(model)
    model= Add(name = 'add_layer')([model1, model])

    output_img = model
#    comcnn_upscale=btl.UpSampling2DBilinear(40,40)(comcnn_layer3)
    model = Model(inpt, output_img)
#    model_comcnn.compile(optimizer=adam, loss = 'mean_squared_error')
    return model



def train_datagen(epoch_iter=3200,epoch_num=50,batch_size=64):
    while(True):
#        n_count = 0
#        if n_count == 0:
#            #print(n_count)
        [xs1,xs2]=dg.jpegqf15_same_bpp_fsrcnn_enhancement()
        #Standarize the data to mean 0 and variance 1
        #We take mean & variance of the data set
        #This is called feature wise scaling
#        xs_n=xs-xs.mean()
#        xs_n=xs/xs.std()
        
#            assert len(xs)%args.batch_size ==0, \
#            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
#            xs = xs.astype('float32')/255.0
        indices = list(range(xs1.shape[0]))
#            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs2[indices[i:i+batch_size]]
                batch_y = xs1[indices[i:i+batch_size]]
                
                yield batch_x, batch_y
                




def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf15_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*ebrcnn_expdecay_same_likecom_qf15_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)
     
def exp_decay(epoch):
   initial_lrate = args.lr
   k = 0.088
#   k = 0.0183
   lr = initial_lrate * math.exp(-k*epoch)
   log('current learning rate is %2.8f' %lr)
   return lr

if __name__ == '__main__':
    # model selection
    model_fsrcnn=fsrcnn_enhancement()
    model_fsrcnn.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf15_%03d.hdf5'%initial_epoch), compile=False)
    
    # compile the model
    model_fsrcnn.compile(optimizer=Adam(0.001), loss='mean_squared_error')
#    adam_rec=optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=False)
#    sgd = SGD(lr=0.1, momentum=0.9,clipvalue=0.005, nesterov=False)
#    model_rec.compile(optimizer=adam_rec, loss='mean_squared_error')
#    model_rec.compile(optimizer=adam_rec, loss='mean_squared_error')
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf15_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log_ebrcnn_expdecay_same_likecom_qf15.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(exp_decay)
    
    history = model_fsrcnn.fit_generator(train_datagen(),
                steps_per_epoch=3200, epochs=50, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])




##############forqf 10######################
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ebrcnn_expdecay_same_likecom', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
#parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
#parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


#save_dir = os.path.join('VDSR_sus',args.model+'_') 

save_dir = os.path.join('com_2020',args.model)

def fsrcnn_enhancement():
#    adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.08798916064, amsgrad=False)
    inpt = Input(shape=(None,None,1),name='input_com')

#comcnnlayer 1
    model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_uniform',name='layer1')(inpt)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL1_fsr')(model)

#    model=Activation('PReLU',name='reluL1_com')(model)

    model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform',name='layer2',)(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL2_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer3')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL3_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer4')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL4_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer5')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL5_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer6')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL6_fsr')(model)

    model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_uniform',name='layer7')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL7_fsr')(model)

    model1 = Conv2DTranspose(1, (9, 9), strides=(2, 2), padding='same',name='layer8')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer9')(model1)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer9_bn')(model) 
    model=Activation('relu',name='reluL9_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer10')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer10_bn')(model) 
    model=Activation('relu',name='reluL10_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer11')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer11_bn')(model) 
    model=Activation('relu',name='reluL11_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer12')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer12_bn')(model) 
    model=Activation('relu',name='reluL12_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer13')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer13_bn')(model) 
    model=Activation('relu',name='reluL13_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer14')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer14_bn')(model) 
    model=Activation('relu',name='reluL14_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer15')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer915_bn')(model) 
    model=Activation('relu',name='reluL15_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer16')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer16_bn')(model) 
    model=Activation('relu',name='reluL16_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer17')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer17_bn')(model) 
    model=Activation('relu',name='reluL17_fsr')(model)
    model = Conv2D(1, (3,3), padding='same', kernel_initializer='he_uniform',name='layer18')(model)
    model= Add(name = 'add_layer')([model1, model])

    output_img = model
#    comcnn_upscale=btl.UpSampling2DBilinear(40,40)(comcnn_layer3)
    model = Model(inpt, output_img)
#    model_comcnn.compile(optimizer=adam, loss = 'mean_squared_error')
    return model



def train_datagen(epoch_iter=3200,epoch_num=50,batch_size=64):
    while(True):
#        n_count = 0
#        if n_count == 0:
#            #print(n_count)
        [xs1,xs2]=dg.jpeg_same_bpp_fsrcnn_enhancement()
        #Standarize the data to mean 0 and variance 1
        #We take mean & variance of the data set
        #This is called feature wise scaling
#        xs_n=xs-xs.mean()
#        xs_n=xs/xs.std()
        
#            assert len(xs)%args.batch_size ==0, \
#            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
#            xs = xs.astype('float32')/255.0
        indices = list(range(xs1.shape[0]))
#            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs2[indices[i:i+batch_size]]
                batch_y = xs1[indices[i:i+batch_size]]
                
                yield batch_x, batch_y
                




def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*ebrcnn_expdecay_same_likecom_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)
     
def exp_decay(epoch):
   initial_lrate = args.lr
   k = 0.088
#   k = 0.0183
   lr = initial_lrate * math.exp(-k*epoch)
   log('current learning rate is %2.8f' %lr)
   return lr

if __name__ == '__main__':
    # model selection
    model_fsrcnn=fsrcnn_enhancement()
    model_fsrcnn.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_%03d.hdf5'%initial_epoch), compile=False)
    
    # compile the model
    model_fsrcnn.compile(optimizer=Adam(0.001), loss='mean_squared_error')
#    adam_rec=optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=False)
#    sgd = SGD(lr=0.1, momentum=0.9,clipvalue=0.005, nesterov=False)
#    model_rec.compile(optimizer=adam_rec, loss='mean_squared_error')
#    model_rec.compile(optimizer=adam_rec, loss='mean_squared_error')
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log_ebrcnn_expdecay_same_likecom.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(exp_decay)
    
    history = model_fsrcnn.fit_generator(train_datagen(),
                steps_per_epoch=3200, epochs=50, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])


###########Test###################################################################################
##############for qf5#######################################
model_comcnn=load_model('comcnn_v3_expdecay_lr0.001_050.hdf5',custom_objects= {"tf_v2":tf_v2})



intermediate_layer_model = Model(inputs=model_comcnn.get_layer('input_com').input,
                                 outputs=model_comcnn.get_layer('convL3_com').output)
tst=np.array(imread('Dataset/Testset/01.png'), dtype=np.float32) / 255.0

tst2=tst[np.newaxis,...,np.newaxis]


opt=intermediate_layer_model.predict(tst2)
opt_n=np.squeeze(np.moveaxis(opt[...,0],0,-1))



h,w=opt_n.shape
for j in range(0,h):
    idx=np.where((opt_n[j,0:w])>1.0)
    opt_n[j,idx]=1
    idx=np.where((opt_n[j,0:w])<0.0)
    opt_n[j,idx]=0


opt_n=opt_n*255
opt_new = np.array(opt_n, dtype='uint8')
opt_enc=jpeg8_encode(opt_new,37)
#############JPEG DECODER############
opt_dec=jpeg8_decode(opt_enc)
imshow(opt_dec)
com_output2=np.array(opt_dec,dtype=np.float32)/255.0
fsrcnn_adding_axis=com_output2[np.newaxis,...,np.newaxis]

model_fsrcnn=load_model('ebrcnn_expdecay_same_likecom_qf5_050.hdf5')
predict_fsrcnn=model_fsrcnn.predict(fsrcnn_adding_axis)
squizing_fsrcnn=np.squeeze(np.moveaxis(predict_fsrcnn[...,0],0,-1))


h,w=squizing_fsrcnn.shape
for j in range(0,h):
    idx=np.where((squizing_fsrcnn[j,0:w])>1.0)
    squizing_fsrcnn[j,idx]=1
    idx=np.where((squizing_fsrcnn[j,0:w])<0.0)
    squizing_fsrcnn[j,idx]=0
    
fsrcnn_output=squizing_fsrcnn*255.0
fsrcnn_output2=np.array(fsrcnn_output,dtype='uint8')
imshow(fsrcnn_output2)
tst=np.array(imread('Dataset/Testset/01.png'))
psnr=compare_psnr(tst,fsrcnn_output2)
ssim=compare_ssim(tst,fsrcnn_output2)

############to compute jpeg psnr ansd ssim#################

jpeg_enc=jpeg8_encode(tst,8)
#############JPEG DECODER############
jpeg_dec=jpeg8_decode(jpeg_enc)
psnr_jpeg8=compare_psnr(tst,jpeg_dec)
ssim_jpeg8=compare_ssim(tst,jpeg_dec)

####################training to compute trainning time##################
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ebrcnn_expdecay_same_likecom_qf5_compute_time', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
#parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
#parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


#save_dir = os.path.join('VDSR_sus',args.model+'_') 

save_dir = os.path.join('result_analysis',args.model)

def fsrcnn_enhancement():
#    adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.08798916064, amsgrad=False)
    inpt = Input(shape=(None,None,1),name='input_com')

#comcnnlayer 1
    model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_uniform',name='layer1')(inpt)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL1_fsr')(model)

#    model=Activation('PReLU',name='reluL1_com')(model)

    model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform',name='layer2',)(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL2_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer3')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL3_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer4')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL4_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer5')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL5_fsr')(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_uniform',name='layer6')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL6_fsr')(model)

    model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_uniform',name='layer7')(model)
#    model = PReLU()(model)
    model=Activation('relu',name='reluL7_fsr')(model)

    model1 = Conv2DTranspose(1, (9, 9), strides=(2, 2), padding='same',name='layer8')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer9')(model1)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer9_bn')(model) 
    model=Activation('relu',name='reluL9_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer10')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer10_bn')(model) 
    model=Activation('relu',name='reluL10_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer11')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer11_bn')(model) 
    model=Activation('relu',name='reluL11_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer12')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer12_bn')(model) 
    model=Activation('relu',name='reluL12_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer13')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer13_bn')(model) 
    model=Activation('relu',name='reluL13_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer14')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer14_bn')(model) 
    model=Activation('relu',name='reluL14_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer15')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer915_bn')(model) 
    model=Activation('relu',name='reluL15_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer16')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer16_bn')(model) 
    model=Activation('relu',name='reluL16_fsr')(model)
    model = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform',name='layer17')(model)
    model=  BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'layer17_bn')(model) 
    model=Activation('relu',name='reluL17_fsr')(model)
    model = Conv2D(1, (3,3), padding='same', kernel_initializer='he_uniform',name='layer18')(model)
    model= Add(name = 'add_layer')([model1, model])

    output_img = model
#    comcnn_upscale=btl.UpSampling2DBilinear(40,40)(comcnn_layer3)
    model = Model(inpt, output_img)
#    model_comcnn.compile(optimizer=adam, loss = 'mean_squared_error')
    return model



def train_datagen(epoch_iter=3200,epoch_num=50,batch_size=64):
    while(True):
#        n_count = 0
#        if n_count == 0:
#            #print(n_count)
        [xs1,xs2]=dg.jpegqf5_same_bpp_fsrcnn_enhancement()
        #Standarize the data to mean 0 and variance 1
        #We take mean & variance of the data set
        #This is called feature wise scaling
#        xs_n=xs-xs.mean()
#        xs_n=xs/xs.std()
        
#            assert len(xs)%args.batch_size ==0, \
#            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
#            xs = xs.astype('float32')/255.0
        indices = list(range(xs1.shape[0]))
#            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs2[indices[i:i+batch_size]]
                batch_y = xs1[indices[i:i+batch_size]]
                
                yield batch_x, batch_y
                




def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf5_compute_time_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*ebrcnn_expdecay_same_likecom_qf5_compute_time_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)
     
def exp_decay(epoch):
   initial_lrate = args.lr
   k = 0.088
#   k = 0.0183
   lr = initial_lrate * math.exp(-k*epoch)
   log('current learning rate is %2.8f' %lr)
   return lr


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

if __name__ == '__main__':
    # model selection
    model_fsrcnn=fsrcnn_enhancement()
    model_fsrcnn.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf5_compute_time_%03d.hdf5'%initial_epoch), compile=False)
    
    # compile the model
    model_fsrcnn.compile(optimizer=Adam(0.001), loss='mean_squared_error')
#    adam_rec=optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=False)
#    sgd = SGD(lr=0.1, momentum=0.9,clipvalue=0.005, nesterov=False)
#    model_rec.compile(optimizer=adam_rec, loss='mean_squared_error')
#    model_rec.compile(optimizer=adam_rec, loss='mean_squared_error')
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'ebrcnn_expdecay_same_likecom_qf5_compute_time_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log_ebrcnn_expdecay_same_likecom_qf5_compute_time.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(exp_decay)
    time_callback = TimeHistory()
    history = model_fsrcnn.fit_generator(train_datagen(),
                steps_per_epoch=3200, epochs=50, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler,time_callback])
    
    times = time_callback.times
    j=np.asarray(times)
    j=np.transpose(j)
    csvfile = open('epoch_time_ebrcnn.csv','w', newline='')
    obj = csv.writer(csvfile)
    obj.writerows(j)
    csvfile.close()            
    