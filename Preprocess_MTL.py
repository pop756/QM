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
import sys





with open('./BERT/SMILE/1M_random_ZINC_word2index.pkl','rb') as file:
    word2idx = pickle.load(file)


## Make Randomized smiles from canonical form
def Chem_generator(smiles,label):
    res_list = []
    len_20 = []
    train_label = []
    index1 = 0
    for i in tqdm(smiles):
        try:
            mol = Chem.MolFromSmiles(i)
            if mol == None:
                index1+=1
                continue
                
        except:
            index1+=1
            continue
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
def Preprocess(smiles,label,number_of_task):
    smiles,_,train_label = Chem_generator(smiles,label)
    image = []
    for smile in smiles:
        part_image = []
        for single_smile in smile:
            temp = []
            for i in range(number_of_task):
                temp.append(250+i)
            temp.append(1)
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
        
    
    return train_image, train_label

def Preprocess_task(smiles,label,task,number_of_task):
    smiles,len_20,train_label = Chem_generator(smiles,label)
    image = []
    for smile in smiles:
        part_image = []
        for single_smile in smile:
            temp = []
            single_smile = tokenizer.atomwise_tokenizer(single_smile)
            for i in range(number_of_task):
                temp.append(250+i)
            temp.append(1)
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

    


def predict(model,data_set,model_name = None,name = None,write = False):
    logits = model.predict(flatten(data_set[0]),verbose = 0)
    logits = np.reshape(logits,[-1])

    index = 0
    y_true = []
    temp_logits = []
    for label in data_set[1]:
        y_true.append(label[0])

    for label in data_set[1]:
        seq_len = len(label)
        temp_logits.append(np.mean(logits[index:index+seq_len]))
        index = index + seq_len
    if write == False:
        return np.array(temp_logits)
    
    logits = np.array(temp_logits)
    csv_file_path = "./result_csv/MTL_result.csv"

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

def batch_make(image,label,task):
    batch_size = 32
    image = [image[i:i+batch_size] for i in range(0, len(image), batch_size)]
    label = [label[i:i+batch_size] for i in range(0, len(label), batch_size)]
    task = [task]*len(label)
    return image,label,task

def main():
    folder_path = './MTL_RAW'
    file_list = os.listdir(folder_path)
    csv_files = [file for file in file_list if file.endswith('.csv')]
    data_set = {}

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        task = csv_file[:-4]
        data = pd.read_csv(file_path)
        data_set[task] = Preprocess(data['Drug'],data['Y'],number_of_task=len(csv_files))
    k = 5
    kf = KFold(n_splits=k)

    data = {}
    train_data = {}
    for task in data_set.keys():
        temp = list(zip(*data_set[task]))
        random.shuffle(temp)
        data_set[task] = list(zip(*temp))
        for fold, (train_index, test_index) in enumerate(kf.split(data_set[task][0])):
            try:
                data[fold][task] = {}
            except:
                data[fold] = {}
                data[fold][task] = {}
            try:
                data_temp = list(zip(flatten([data_set[task][0][i] for i in train_index]), flatten([data_set[task][1][i] for i in train_index])))
                random.shuffle(data_temp)
                data_temp = list(zip(*data_temp))
                temp = batch_make(data_temp[0], data_temp[1],task)
                train_data[fold][0] += temp[0]
                train_data[fold][1] += temp[1]
                train_data[fold][2] += temp[2]
            except:
                data_temp = list(zip(flatten([data_set[task][0][i] for i in train_index]), flatten([data_set[task][1][i] for i in train_index])))
                random.shuffle(data_temp)
                data_temp = list(zip(*data_temp))
                temp = batch_make(data_temp[0], data_temp[1],task)
                train_data[fold] = list(temp)
            fold_data = [data_set[task][0][i] for i in test_index]
            fold_data1 = [data_set[task][1][i] for i in test_index]
            split_index = len(fold_data) // 2
            data[fold][task]['val'] = fold_data[:split_index],fold_data1[:split_index]
            data[fold][task]['test'] = fold_data[split_index:],fold_data1[split_index:]




    epochs = 5
    for fold in range(5):
        all_models = []
        models_BERT = {}
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        #bert_layer = custom_layers.BERT(256,8,1024,strat_index=len(data_set.keys()))
        globals()['tensor_bert_layer'] = custom_layers.BERT_tensor(256,6,1024,strat_index=len(data_set.keys()))
        globals()['small_tensor_bert_layer'] = custom_layers.BERT_tensor_small(256,8,1024,strat_index=len(data_set.keys()))
        globals()['bert_layer'] = custom_layers.BERT(256,8,1024,strat_index=len(data_set.keys()))
        globals()['GPU_bert_layer'] = custom_layers.BERT_tensor_small_GPU(256,8,1024,strat_index=len(data_set.keys()))
        for index,task in enumerate(data_set.keys()):
            models_BERT[task] = BERT(len(data_set.keys()),index)
        all_models.append(models_BERT)
        models_GPU = {}
        for index,task in enumerate(data_set.keys()):
            models_GPU[task] = GPU_tensor_BERT(len(data_set.keys()),index)
        all_models.append(models_GPU)
        models_Tensor = {}
        for index,task in enumerate(data_set.keys()):
            models_Tensor[task] = tensor_BERT_small(len(data_set.keys()),index)
        all_models.append(models_Tensor)
        model_names = ['BERT','GPU_BERT','tensor_BERT']
        
        temp = list(zip(*train_data[fold]))
        random.shuffle(temp)
        for model_index,models in enumerate(all_models):
            print('Model : ',model_names[model_index])
            models[list(data_set.keys())[0]].summary()
            avgs = []
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                loss_list = {}
                acc_list = {}
                auc_list = {}
                for temp_task in data_set.keys():
                    loss_list[temp_task] = []
                    acc_list[temp_task] = []
                    auc_list[temp_task] = []
                for i in range(0, len(temp)):
                    batch_images = temp[i][0]
                    batch_images = tf.reshape(batch_images,[-1,200])
                    batch_labels = temp[i][1]
                    batch_labels = tf.reshape(batch_labels,[-1])
                    task = temp[i][2]

                    with tf.GradientTape() as tape:
                        model = models[task]
                        logits = model(batch_images)
                        logits = tf.reshape(logits,[-1])
                        loss_value = loss_fn(batch_labels, logits)
                        acc = tf.keras.metrics.Accuracy()(np.round(logits),batch_labels)
                        auc = tf.keras.metrics.AUC()(batch_labels,logits)
                        auc = auc.numpy()
                        loss_value = tf.reduce_mean(loss_value)
                        loss_list[task].append(loss_value)
                        acc_list[task].append(acc)
                        auc_list[task].append(auc)

                    grads = tape.gradient(loss_value, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    if i % 100 == 0:
                        for index,l in enumerate(data_set.keys()):
                            
                            temp_loss = np.average(loss_list[l][-100:])
                            temp_acc = np.average(acc_list[l][-100:])
                            temp_auc = np.average(auc_list[l][-100:])
                            if index == 0:
                                text = "\rSteps : {} Task : {}, Loss: {:.4f}, acc : {:.4f}, auc : {:.4f}".format(i,l,temp_loss,temp_acc,temp_auc)
                            else:
                                text += "       Task : {}, Loss: {:.4f}, acc : {:.4f}, auc : {:.4f}".format(l,temp_loss,temp_acc,temp_auc)
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        
                        
                # 각 에포크 종료 후 평가

                avg = 0
                for j in data_set.keys():
            
                    x_vals = data[fold][j]['val']
                    y_vals = []
                    for i in x_vals[1]:
                        y_vals.append(i[0])
                    
                    val_res = predict(models[j],x_vals)
                    val_res = val_res.reshape(-1)
                    acc = tf.keras.metrics.Accuracy()(y_vals,np.round(val_res))
                    auc_res = (tf.keras.metrics.AUC()(y_vals,val_res)).numpy()
                    loss = loss_fn(y_vals,val_res)
                    
                    avg += auc_res
                    print(f'\nTask is {j}')
                    print(f"Test accuracy: {acc}")
                    print(f"Test AUC: {auc_res}")
                    print(f"Test loss: {loss}\n")
                avg = auc_res/len(data_set.keys())    
                avgs.append([avg])
                print(f'Loss avg {avg}')
                

                    
                    
            for j in data_set.keys():
                
                x_vals = data[fold][j]['val']
                x_test = data[fold][j]['test']
                
                predict(models[j],x_vals,model_name = model_names[model_index],name = j+'_val',write = True)
                predict(models[j],x_test,model_name = model_names[model_index],name = j,write = True)
        
"""
import pandas as pd

# 원하는 열의 이름을 포함한 빈 데이터프레임 생성
data = {"model": [], "tox": [], 'auc': [],'acc':[],'loss':[]}  # 열 이름을 원하는 대로 수정하세요
df = pd.DataFrame(data)

# 빈 데이터프레임을 CSV 파일로 저장
df.to_csv('./result_csv/MTL_result.csv', index=False)""" 



def tensor_BERT_small(number_of_task,task_index = 0):
    with open('./BERT/SMILE/small_Pre_BERT.pkl','rb') as file:
        paras = pickle.load(file)
    if number_of_task == 0:
        mask = Task_mask(number_of_task+1)
    else:
        mask = Task_mask(number_of_task)
    inputs = tf.keras.layers.Input(200,dtype=tf.int32)
    hidden = small_tensor_bert_layer(inputs,att_mask = mask)
    hidden = hidden[:,task_index]
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    hidden = tf.keras.layers.Dense(100,activation='gelu')(hidden)
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
    inputs = tf.keras.layers.Input(200,dtype=tf.int32)
    hidden = tensor_bert_layer(inputs,att_mask = mask)
    hidden = hidden[:,task_index]
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    hidden = tf.keras.layers.Dense(100,activation='gelu')(hidden)
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid')(hidden)
    result = tf.keras.Model(inputs = [inputs],outputs = [output])
    result.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),loss = 'binary_crossentropy',metrics=['acc',tf.keras.metrics.AUC(name='auc')])
    result.layers[1].set_weights(paras)
    return result


def BERT(number_of_task,task_index = 0):
    with open('./BERT/SMILE/V_Pre_BERT.pkl','rb') as file:
        paras = pickle.load(file)
    if number_of_task == 0:
        mask = 1-Task_mask(number_of_task+1)
    else:
        mask = 1-Task_mask(number_of_task)
    inputs = tf.keras.layers.Input(200,dtype=tf.int32)
    hidden = bert_layer(inputs,None,att_mask = mask)
    hidden = hidden[:,task_index]
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    hidden = tf.keras.layers.Dense(100,activation='gelu')(hidden)
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid')(hidden)
    result = tf.keras.Model(inputs = [inputs],outputs = [output])
    result.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),loss = 'binary_crossentropy',metrics=['acc',tf.keras.metrics.AUC(name='auc')])
    result.layers[1].set_weights(paras)
    return result

def GPU_tensor_BERT(number_of_task,task_index = 0):
    with open('./BERT/SMILE/GPU_small_Pre_BERT.pkl','rb') as file:
        paras = pickle.load(file)
    if number_of_task == 0:
        mask = Task_mask(number_of_task+1)
    else:
        mask = Task_mask(number_of_task)
    inputs = tf.keras.layers.Input(200,dtype=tf.int32)
    hidden = GPU_bert_layer(inputs,None,att_mask = mask)
    hidden = hidden[:,task_index]
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    hidden = tf.keras.layers.Dense(100,activation='gelu')(hidden)
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid')(hidden)
    result = tf.keras.Model(inputs = [inputs],outputs = [output])
    result.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),loss = 'binary_crossentropy',metrics=['acc',tf.keras.metrics.AUC(name='auc')])
    result.layers[1].set_weights(paras)
    return result

main()