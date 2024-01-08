from SmilesPE import tokenizer
from rdkit import Chem
from tqdm import tqdm
from tdc.single_pred import Tox
import pickle
import tensorflow as tf 
import numpy as np
from Module import custom_layers
import csv
from sklearn.model_selection import KFold
import os 
from SmilesPE import tokenizer






with open('./BERT/SMILE/1M_random_ZINC_word2index.pkl','rb') as file:
    word2idx = pickle.load(file)


## Make Randomized smiles from canonical form
def Chem_generator(smiles,label):
    res_list = []
    len_20 = []
    train_label = []
    index1 = 0
    for i in tqdm(smiles):
        mol = Chem.MolFromSmiles(i)
        temp = []
        index = 0
        while(len(set(temp))!=20 and index != 100):
            index+=1
            temp.append(Chem.MolToSmiles(mol,doRandom=True))
        temp_res = list(set(temp))
        res_list.append(temp_res)
        len_20.append(len(temp_res))
        train_label.append([label[index1]]*len(temp_res))
        index1+=1
    return res_list,len_20,train_label

def flatten(input_list):
    res = []
    for seq in input_list:
        for single in seq:
            res.append(single)
    return np.array(res)
    
    

## Preprocess for SMILE
def Preprocess(smiles,label):
    smiles,len_20,train_label = Chem_generator(smiles,label)
    image = []
    for smile in smiles:
        part_image = []
        for single_smile in smile:
            temp = []
            single_smile = tokenizer.atomwise_tokenizer(single_smile)
            for token in single_smile:
                try:
                    temp.append(word2idx[token])
                except:
                    print(token,' is not in the word2idx')
                    word2idx[token] = len(word2idx)+1
            part_image.append(temp)
        image.append(part_image)
    train_image = []
    for single_image in image:
        temp = tf.keras.preprocessing.sequence.pad_sequences(single_image,200,padding='post')
        temp = temp.astype(np.int32)
        train_image.append(temp)
        
    
    return train_image, train_label, len_20

def Preprocess_task(smiles,label,task):
    smiles,len_20,train_label = Chem_generator(smiles,label)
    image = []
    for smile in smiles:
        part_image = []
        for single_smile in smile:
            temp = []
            single_smile = tokenizer.atomwise_tokenizer(single_smile)
            for token in single_smile:
                try:
                    temp.append(word2idx[token])
                except:
                    print(token,' is not in the word2idx')
                    word2idx[token] = len(word2idx)+1
            part_image.append(temp)
        image.append(part_image)
    train_image = []
    for single_image in image:
        temp = tf.keras.preprocessing.sequence.pad_sequences(single_image,200,padding='post')
        temp = temp.astype(np.int32)
        train_image.append(temp)
    train_image = flatten(train_image)    
    train_label = flatten(train_label)
    task_label = [task]*len(train_image)
    
    batch_size = 32
    train_image = [train_image[i:i+batch_size] for i in range(0, len(train_image), batch_size)]
    train_label = [train_label[i:i+batch_size] for i in range(0, len(train_label), batch_size)]
    task_label = [task_label[i:i+batch_size] for i in range(0, len(task_label), batch_size)]
    return train_image, train_label,task_label

def read_folder(folder_path):
    file_list = os.listdir(folder_path)
    csv_files = [file for file in file_list if file.endswith('.csv')]
    dataframes = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    return dataframes


def Task_mask(num_task):
    result = np.zeros([200,200])
    for i in range(num_task):
        for j in range(200):
            if j == i:
                continue
            else:
                result[j][i] = 1
    return result

import pandas as pd
import random

    
def tensor_BERT_small(number_of_task,task_index = 0):
    with open('./BERT/SMILE/small_Pre_BERT.pkl','rb') as file:
        paras = pickle.load(file)
    if number_of_task == 0:
        mask = Task_mask(number_of_task+1)
    else:
        mask = Task_mask(number_of_task)
    bert_layer = custom_layers.BERT_tensor_small(256,8,1024,strat_index=number_of_task)
    inputs = tf.keras.layers.Input(200,dtype=tf.int32)
    hidden = bert_layer(inputs,att_mask = mask)
    hidden = hidden[:,task_index]
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid')(hidden)
    result = tf.keras.Model(inputs = [inputs],outputs = [output])
    result.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),loss = 'binary_crossentropy',metrics=['acc',tf.keras.metrics.AUC(name='auc')])
    result.layers[1].set_weights(paras)
    return result
    
    
def tensor_BERT(number_of_task,task_index = 0):
    with open('./BERT/SMILE/Pre_BERT.pkl','rb') as file:
        paras = pickle.load(file)
    if number_of_task == 0:
        mask = Task_mask(number_of_task+1)
    else:
        mask = Task_mask(number_of_task)
    bert_layer = custom_layers.BERT_tensor(256,6,1024,strat_index=number_of_task)
    inputs = tf.keras.layers.Input(200,dtype=tf.int32)
    hidden = bert_layer(inputs,att_mask = mask)
    hidden = hidden[:,task_index]
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid')(hidden)
    result = tf.keras.Model(inputs = [inputs],outputs = [output])
    result.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),loss = 'binary_crossentropy',metrics=['acc',tf.keras.metrics.AUC(name='auc')])
    result.layers[1].set_weights(paras)
    return result

def predict(model,data_set,model_name,name,write = False):
    logits = model.predict(flatten(data_set[0]),verbose = 0)
    logits = np.reshape(logits,[-1])

    index = 0
    y_true = []
    temp_logits = []
    for label in data_set[1]:
        y_true.append(label[0])

    for seq_len in data_set[2]:
        temp_logits.append(np.mean(logits[index:index+seq_len]))
        index = index + seq_len
    if write == False:
        return np.array(temp_logits)
    
    logits = np.array(temp_logits)
    csv_file_path = "./result_csv/result.csv"

    ACC = tf.keras.metrics.Accuracy()(y_true,np.round(logits))
    AUC = tf.keras.metrics.AUC()(y_true,logits)
    loss = tf.keras.metrics.binary_crossentropy(y_true,logits)
    with open(csv_file_path, mode='a', newline='') as file:
        data = {'model':model_name,'tox':name,'auc':AUC.numpy(),'acc':ACC.numpy(),'loss':loss.numpy()}
        fieldnames = ["model","tox", "auc", "acc", "loss"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(data)



class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self,data_set):
        super().__init__()
        self.x_val = data_set[0]
        index = 0
        res = []
        for i in data_set[1]:
            res.append(i[0])
        self.len20 = data_set[2]
        self.counts = []
        self.max = 0
        self.y_true = np.array(res)
        self.data_set = data_set
        self.history = {}
        self.epoch = 0
    def on_epoch_end(self, epoch, logs=None):
        # 에포크가 끝날 때마다 validation 데이터로 모델 평가
        result = predict(self.model,self.data_set,None,None)
        result = np.reshape(result,[-1])
        acc = tf.keras.metrics.Accuracy()(self.y_true,np.round(result))
        auc_res = (tf.keras.metrics.AUC()(self.y_true,result)).numpy()
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
            
        if self.history['val_auc'][-1]<self.max-0.01:
            self.counts.append(1)
        else:
            self.counts = []

        self.epoch += 1

        if len(self.counts)>=1 and self.epoch>=3:
            self.model.stop_training = True



    
def main():
    folder_path = './RAW_data'
    file_list = os.listdir(folder_path)
    csv_files = [file for file in file_list if file.endswith('.csv')]
    data_set = {}
    val_set = {}
    test_set = {}
    train_set = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        print(csv_file[:-4])
        task = csv_file[:-4]
        data = pd.read_csv(file_path)
        data = data.sample(frac=1).reset_index(drop=True)

        # K-Fold 교차 검증을 위한 K 값 설정
        k = 5
        kf = KFold(n_splits=k)
        data_set[task] = {}
        for fold, (train_index, test_index) in enumerate(kf.split(data)):
            # 각 fold에 대한 데이터를 나눕니다.
            fold_data = data.iloc[test_index]

            # 데이터를 50:50 비율로 validation과 test로 나눕니다.
            data_set[task]['train'] = data.iloc[train_index]
            split_index = len(fold_data) // 2
            data_set[task]['val'] = fold_data.iloc[:split_index]
            data_set[task]['test'] = fold_data.iloc[split_index:]
            if len(train_set) == 0:
                train_set = Preprocess_task(data_set[task]['train']['Drug'],data_set[task]['train']['Y'],task)
            else:
                temp_train = Preprocess_task(data_set[task]['train']['Drug'],data_set[task]['train']['Y'],task)
                train_set[0] += temp_train[0]
                train_set[1] += temp_train[1]
                train_set[2] += temp_train[2]
            val_set[task] = Preprocess(data_set[task]['val']['Drug'].to_numpy(),data_set[task]['val']['Y'].to_numpy())
            test_set[task] = Preprocess(data_set[task]['test']['Drug'].to_numpy(),data_set[task]['test']['Y'].to_numpy())
            train_shuffle = list(zip(flatten(train_set[0]),flatten(train_set[1])))
            random.shuffle(train_shuffle)
            train_shuffle = list(zip(*train_shuffle))
            train_shuffle[0] = np.array(train_shuffle[0])
            train_shuffle[1] = np.array(train_shuffle[1])

            call_back = CustomCallback(val_set)



            model = tensor_BERT(2)
            model.fit(train_shuffle[0],train_shuffle[1],callbacks=[call_back],batch_size=32,epochs=10)
            predict(model,test_set,'tensor',csv_file[:-4],True)
            predict(model, val_set,'tensor',csv_file[:-4]+'_val',True)
            
            call_back = CustomCallback(val_set)
            model = tensor_BERT_small(2)
            model.fit(train_shuffle[0],train_shuffle[1],callbacks=[call_back],batch_size=32,epochs=10)
            predict(model,test_set,'small_tensor',csv_file[:-4],True)
            predict(model, val_set,'small_tensor',csv_file[:-4]+'_val',True)


main()
    
