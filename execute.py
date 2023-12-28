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


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,mask_zero=True)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim,mask_zero = True)

    def call(self, x):
        positions = np.array([0]+[0]+[i+2 for i in range(198)])
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions    
    
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        l2_reg = tf.keras.regularizers.l2(0.01)
        
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,kernel_regularizer=l2_reg)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs,attention_mask=tf.cast(np.array([[1]+[0]+[1]*198] + [[0]+[1]*199] + [[0]*2+[1]*198]*198),bool))
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
    
    
    
class BERT(tf.keras.layers.Layer):
    def __init__(self,emb_dim,num_heads,ff_dim):
        super(BERT, self).__init__()
        self.encoder = tf.keras.Sequential([TransformerBlock(emb_dim,num_heads,ff_dim) for i in range(8)])
        
        self.embedding = TokenAndPositionEmbedding(200,3500,256)
        self.dense = tf.keras.layers.Dense(250,activation = 'gelu')
        self.classify = tf.keras.layers.Dense(71,activation = 'softmax')
    def call(self, inputs, mask_index=None,pretrain = False):
        if pretrain:
            mask_index = tf.one_hot(mask_index,200)
            boolean_mask = tf.cast(tf.reduce_sum(mask_index,axis=1),bool)
            inputs = tf.cast(inputs,dtype=tf.int32)
        
        inputs = tf.reshape(inputs,[-1,200])
        hidden = self.embedding(inputs)
        
        
        hidden = self.encoder(hidden)
    
        if pretrain:
            output = tf.reshape(hidden,[-1,200,256])
            output = self.dense(output)
            output = layers.Dropout(0.1)(output)
            output = self.classify(output)
            output = tf.boolean_mask(output,boolean_mask)
            return output
        else:
            return hidden


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
outputs = BERT(256,6,1024)(inputs,None)

model = Model(inputs = [inputs], outputs = [outputs])

#model.load_weights('./BERT/atomInSmile/F_Random_ZINC_L_model_weights.h5')


#BERT_parameters = model.get_weights()[:130]

bert_layer = BERT(256,6,1024)

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

def BERT_model():
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


    
class tox_process():
    def __init__(self,tox,test_size=0.2,random_state = 1024):
        self.tox_name = tox
        self.size = test_size
        self.seed = random_state
        
        
    def AIS_process(self,plot=False,token = 'AIS',number_of_task = 2):
        if token == 'AIS':
            with open('./Tox_data/AIS_Tox_data/'+self.tox_name,'rb') as file:
                train,label,len_20 = pickle.load(file)[0]
            with open('./BERT/atomInSmile/1M_random_ZINC_word2index.pkl','rb') as file:
                word2idx = pickle.load(file)
        elif token == 'SMILE':
            with open('./Tox_data/SMILE_Tox_data/'+self.tox_name,'rb') as file:
                train,label,len_20 = pickle.load(file)[0]
            with open('./BERT/SMILE/1M_random_ZINC_word2index.pkl','rb') as file:
                word2idx = pickle.load(file)
                
            """elif token == 'SmiletoPE':
                with open('./Tox_data/SmiletoPE/'+self.tox_name,'rb') as file:
                    train,label,len_20 = pickle.load(file)[0]
                with open('./BERT/SmiletoPE/1M_random_ZINC_word2index.pkl','rb') as file:
                    word2idx = pickle.load(file)"""
        else:
            raise
        
        
        
        
        
        if plot:
            temp_dict = {}

            for i in train:
                try:
                    temp_dict[len(i)] = temp_dict[len(i)] + 1
                except:
                    temp_dict[len(i)] = 1
                    
            plt.bar(temp_dict.keys(),temp_dict.values())
        
        except_dict = {}

        for i in train:
            for j in i:
                try:
                    word2idx[j]
                except:
                    try:
                        except_dict[j]
                    except:
                        except_dict[j] = len(except_dict) + 1
        
        similar_dict = {}

        for i in except_dict.keys():
            similar_dict[i] = most_similar(i,word2idx)
            
            
        AIS_train = []
        temp_label = []
        temp_index = 0
        sign = len_20[0]
        remove_temp = []
        for index,i in enumerate(train):
            
            if index > sign:
                sign+= len_20[temp_index]
                temp_index+=1
            if len(i)>180:
                remove_temp.append(temp_index)
                continue
            
            temp = []
            if number_of_task == 2:
                temp.append(3)
                temp.append(4)
            else:
                for i in range(number_of_task):
                    temp.append(len(word2idx)+1+i)
            temp.append(1)
            for j in i:
                try:
                    temp.append(word2idx[j])
                except:
                    print('Unexpected : ',j)
                    word2idx[j] = len(word2idx)+1
                    temp.append(word2idx[j])
                """    
                except:
                    if j != '/[H]':
                        print(j,i)
                        word_sim = similar_dict[j]
                        if word_sim != '':
                            temp.append(word2idx[word_sim])
                    else:
                        pass"""
            if len(temp)>1:
                AIS_train.append(temp)
                temp_label.append(label[index])
                
        for j in remove_temp:
            len_20[j] = len_20[j]-1
        label = np.array(temp_label)

        AIS_train = tf.keras.preprocessing.sequence.pad_sequences(AIS_train, padding='post', maxlen=200)
        temp_x = []
        temp_y = []
        index = 0
        for i in len_20:
            temp_x.append(AIS_train[index:index+i])
            temp_y.append(label[index:index+i])
            index = index+i



        x_train, x_val, y_train, y_val,_,len_20 = train_test_split(temp_x,temp_y,len_20, test_size=self.size,random_state=self.seed)
        
        def flatten(data):
            temp = []
            for i in data:
                temp+=list(i)
            data = np.array(temp)
            return data
        x_train = flatten(x_train)
        x_val = flatten(x_val)
        y_val = flatten(y_val)
        y_train = flatten(y_train)
        return x_train, x_val, y_train, y_val,len_20
    
    def bit_precess(self):
        train,tox_info = Tox(name=self.tox_name).get_data(format='DeepPurpose')
        bit_string = RDK.smile_to_RDkit(train,2048)
        x_train_NN,x_val_NN,y_train_NN,y_val_NN = train_test_split(np.array(bit_string)/1.,np.array(tox_info)/1.,test_size=self.size,random_state=self.seed)
        return x_train_NN,x_val_NN,y_train_NN,y_val_NN
    


class execute():
    def __init__(self,tox,test_size,split_seed,epoch = 20,batch=32*20,tokens = ['AIS']):
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
                model.load_weights('./BERT/atomInSmile/Pre_BERT.h5')
            elif token == 'SMILE':
                model.load_weights('./BERT/SMILE/Pre_BERT.h5')
            elif token == 'SmiletoPE':
                model.load_weights('./BERT/SmiletoPE/F_Random_ZINC_L_model_weights.h5')
            else:
                raise
            self.BERTs.append(BERT_model())
            self.BERT_parameters.append(model.get_weights()[:130])
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
    
    plt.savefig(f'./Results/{token[0]}_{token[1]}_Tox_result/'+tox_name+'.png')
    
    plt.show()
    
    
    
    