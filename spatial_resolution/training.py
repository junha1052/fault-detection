import numpy as np
import pandas as pd
#from model_sup import *
from Unet import *
from module_io import * 
import tensorflow as tf
import os
import random
import sys


os.environ["CUDA_VISIBLE_DEVICES"]='0'

datadir='R256/'

s_train=np.load(datadir+'s2.npy')
e_train=np.load(datadir+'e2.npy')

case='case1'

model=unet()

#model_checkpoint = ModelCheckpoint('unet_fault_new.hdf5', monitor='loss',verbose=1, save_best_only=True)
model_checkpoint = ModelCheckpoint('check_'+case+'/unet_fault_{epoch:03d}.hdf5', monitor='val_loss', 
      verbose=1, save_best_only=True, mode='min')

#result=model.fit(s_train,(e_train,e_train,e_train,e_train,e_train),validation_split=0.2,epochs=100,batch_size=200,callbacks=[model_checkpoint])
result=model.fit(s_train,e_train,validation_split=0.2,epochs=100,batch_size=50,callbacks=[model_checkpoint])

#model=tf.keras.models.load_model('./check/unet_fault_027.hdf5')

history=pd.DataFrame(result.history)
history.to_csv('check_'+case+"/history.csv")


preds_train=model.predict(s_val,batch_size=100)
preds_train_t=preds_train
print(preds_train.size)

#preds_train_t = (preds_train > 0.5).astype(np.uint8)

ntr=len(s_train)
savefolder='./train_result_'+case
createFolder(savefolder)
lfolder=savefolder+'/label'
pfolder=savefolder+'/prediction'
sfolder=savefolder+'/seismo'
createFolder(lfolder)
createFolder(pfolder)
createFolder(sfolder)
for i in range(0,len(s_val),100):
    to_bin(lfolder+'/label.'+str(i),e_val[i])
    to_bin(pfolder+'/predict.'+str(i),preds_train_t[i])
    to_bin(sfolder+'/seismo.'+str(i),s_val[i])
 

