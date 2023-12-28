import os
import tensorflow as tf 
from tensorflow.python.client import device_lib
import pickle
import numpy as np
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from tdc.single_pred import Tox
from Module import RDK
from Module import custom_layers


def predict(model,results,len_list):
    index = 0
    res = model.predict(results,verbose=0)
    x_val = []
    for i in len_list:
        temp = res[index:index+i]
        x_val.append(np.average(temp,axis=0))
        index = index+i
    return np.array(x_val)

def similar(a, b):    return SequenceMatcher(None, a, b).ratio()
def most_similar(query,word2idx):

    max = 0
    tokken = ''
    query = query.split(';')
    for i in word2idx.keys():
        key = i.split(';')
        temp2 = 0
        temp3 = 0
        temp1 = similar(query[0],key[0])*10
        try:
            temp2 = similar(query[1],key[1])*2
            temp3 = similar(query[2],key[2])*1
        except:
            pass
        temp = temp1+temp2+temp3
        if temp>max:
            max = temp
            tokken = i
    return tokken

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val,y_true,len_20):
        super().__init__()
        self.x_val = x_val
        index = 0
        res = []
        for i in len_20:
            res.append(np.average(y_true[index:index+i]))
            index = index+i
        self.len20 = len_20
        self.counts = []
        self.max = 0
        self.y_true = np.array(res)
        self.history = {}
        self.epoch = 0
    def on_epoch_end(self, epoch, logs=None):
        # 에포크가 끝날 때마다 validation 데이터로 모델 평가
        result = predict(self.model,self.x_val,self.len20)
        acc = Accuracy()(self.y_true,np.round(result))
        auc_res = (AUC()(self.y_true,result)).numpy()
        loss = tf.keras.metrics.BinaryCrossentropy()(self.y_true,result)
        auc_res = auc_res
        print(f"     val_acc : {acc},    val_auc : {auc_res} val_loss : {loss}")
        if 'val_acc' not in self.history:
            self.history['val_acc'] = [acc]
        else:
            self.history['val_acc'] += [acc]
        if 'val_auc' not in self.history:
            self.history['val_auc'] = [auc_res]
        else:
            self.history['val_auc'] += [auc_res]
        
        if 'val_loss' not in self.history:
            self.history['val_loss'] = [loss]
        else:
            self.history['val_loss'] += [loss]
        
        self.max = np.max(self.history['val_auc'])
            
        if self.history['val_auc'][-1]<self.max:
            self.counts.append(1)
        else:
            self.counts = []

        self.epoch += 1
        """
        if self.epoch>10 and len(self.counts)>2:
            self.model.stop_training = True"""

inputs = Input(shape = (200,),dtype=tf.int32)
outputs = custom_layers.BERT_tensor(256,6,1024,732)(inputs,None)

model = Model(inputs = [inputs], outputs = [outputs])

#model.load_weights('./BERT/atomInSmile/F_Random_ZINC_L_model_weights.h5')


#BERT_parameters = model.get_weights()[:130]

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

def BERT_model():
    bert_layer = custom_layers.BERT_tensor(256,6,1024)
    inputs = Input(200,)
    hidden = bert_layer(inputs)
    hidden = hidden[:,0]
    hidden = tf.keras.layers.Dense(256,activation = 'gelu')(hidden)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid')(hidden)
    result = Model(inputs = [inputs],outputs = [output])
    return result

def Bit_model():
    inputs = Input(2048,)
    hidden = tf.keras.layers.Dense(250,activation = 'relu')(inputs)
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    hidden = tf.keras.layers.Dense(40,activation = 'relu')(hidden)
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    hidden = tf.keras.layers.Dense(10,activation = 'relu')(hidden)
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid')(hidden)
    result = Model(inputs = [inputs],outputs = [output])
    return result

from execute import tox_process

class execute():
    def __init__(self,tox,test_size,split_seed,epoch = 20,batch=32*20,tokens = ['AIS'],att_mask = None):
        super().__init__()
        self.tox = tox
        self.size = test_size
        self.seed = split_seed
        self.epoch = epoch
        self.BERTs = []
        self.batch_size = batch
        Bit_Classifier = Bit_model()
        self.Bit = Bit_Classifier
        #model.load_weights('./BERT/atomInSmile/F_Random_ZINC_L_model_weights.h5')
        self.tokens = tokens
        self.BERT_parameters = []
        
        for token in tokens:
            if token == 'AIS':
                model.load_weights('./BERT/atomInSmile/Pre_BERT')
            elif token == 'SMILE':
                model.load_weights('./BERT/SMILE/small_tensor_Pre_BERT')
            else:
                raise
            self.BERTs.append(BERT_model())
            self.BERT_parameters.append(model.get_weights())
    def forward(self,set_weights=True):
        if set_weights:
            for index,token in enumerate(self.tokens):
                BERT_parameter = self.BERT_parameters[index]
                self.BERTs[index].layers[1].set_weights(BERT_parameter)
        

        for index,token in enumerate(self.tokens): 
            process = tox_process(self.tox, self.size, self.seed)
            x_train, x_val, y_train, y_val,len_20 = process.AIS_process(token = token)
            
            temp_BERT = self.BERTs[index]
            
            val_call = CustomCallback(x_val,y_val,len_20)
            
            temp_BERT.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),loss = 'binary_crossentropy',metrics=['acc',AUC(name='auc')])
            
            hist1 = temp_BERT.fit(x_train,y_train,batch_size=self.batch_size,epochs=self.epoch,callbacks=[val_call])
            
        
            temp_BERT.history.history['val_loss'] = val_call.history['val_loss']
            temp_BERT.history.history['val_acc'] = val_call.history['val_acc']
            temp_BERT.history.history['val_auc'] = val_call.history['val_auc']
            
        x_train_NN,x_val_NN,y_train_NN,y_val_NN = process.bit_precess()
        self.Bit.compile(optimizer = 'Adam',loss = 'binary_crossentropy',metrics=['acc',AUC(name='auc')])
        hist2 = self.Bit.fit(x_train_NN,y_train_NN,batch_size=32,epochs=self.epoch,validation_data=(x_val_NN,y_val_NN))
        
import matplotlib.pyplot as plt
def plot_history(models,tox_name,token=['AIS']):
    plt.figure(figsize=(16,8))
    plt.subplot(2, 3, 1)
    for model in models.keys():
        plt.plot([i for i in range(len(model.history.history['loss']))],model.history.history['loss'],label=models[model])
    #plt.plot([i for i in range(len(val_call2.history.history['val_loss']))],val_call2.history.history['val_loss'],label = 'BERT_Norm acc data')
    plt.title(tox_name + ' train Loss')
    plt.xlabel('epoch')
    plt.ylabel('score')

    plt.legend()
    plt.subplot(2, 3, 2)
    for model in models.keys():
        plt.plot([i for i in range(len(model.history.history['acc']))],model.history.history['acc'],label=models[model])
    #plt.plot([i for i in range(len(val_call2.history.history['val_acc']))],val_call2.history.history['val_acc'],label = 'BERT_Norm acc data')
    plt.title(tox_name + ' train ACC')
    plt.xlabel('epoch')
    plt.ylabel('score')

    plt.legend()
    plt.subplot(2, 3, 3)
    for model in models.keys():
        plt.plot([i for i in range(len(model.history.history['auc']))],model.history.history['auc'],label=models[model])
    plt.title(tox_name + ' train AUC')
    plt.xlabel('epoch')
    plt.ylabel('score')

    plt.subplot(2, 3, 4)
    for model in models.keys():
        plt.plot([i for i in range(len(model.history.history['val_loss']))],model.history.history['val_loss'],label=models[model])
    #plt.plot([i for i in range(len(val_call2.history.history['val_loss']))],val_call2.history.history['val_loss'],label = 'BERT_Norm acc data')
    plt.title(tox_name + ' val Loss')
    plt.xlabel('epoch')
    plt.ylabel('score')

    plt.legend()
    plt.subplot(2, 3, 5)
    for model in models.keys():
        plt.plot([i for i in range(len(model.history.history['val_acc']))],model.history.history['val_acc'],label=models[model])
    #plt.plot([i for i in range(len(val_call2.history.history['val_acc']))],val_call2.history.history['val_acc'],label = 'BERT_Norm acc data')
    plt.title(tox_name + ' val ACC')
    plt.xlabel('epoch')
    plt.ylabel('score')

    plt.legend('epoch')
    plt.ylabel('score')

    plt.legend()
    plt.tight_layout() 
    plt.subplot(2, 3, 6)
    for model in models.keys():
        plt.plot([i for i in range(len(model.history.history['val_auc']))],model.history.history['val_auc'],label=models[model])
    plt.title(tox_name + ' val AUC')
    plt.xlabel('epoch')
    plt.ylabel('score')
    
    plt.savefig(f'./Results/tensor/{token[0]}_Tox_result/'+tox_name+'.png')
    
    plt.show()