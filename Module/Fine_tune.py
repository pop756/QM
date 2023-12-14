from tensorflow.keras.preprocessing.sequence import skipgrams
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def data_process(tokenized_data,word2idx,pad_size):
    
    ## change tokenized data into int sequence.
    encoded = []
    
    for i in tokenized_data:
        temp = []
        
        for _,j in enumerate(i):
            try:
                temp.append(word2idx[j])
            except:
                word2idx[j] = len(word2idx)+1
                temp.append(5)
                print(f'{j} is not in the pretrained data')
        encoded.append(temp)
    train_encoded = []
    for _,i in enumerate(encoded):
        while len(i) != pad_size:
            i.append(0)
        train_encoded.append(i)
    train_encoded = np.array(train_encoded)
    return np.array(train_encoded)

import os
def fine_tune(pretrained_model,encoded,vocab_size,file_name,epochs = 10,batch_size = 10000):
    skip_gram = []
    first_em = []
    second_em = []
    y_123 = []
    if os.path.isdir(file_name):
        print('File exist change the name')
        raise

    
    for i in tqdm(range(len(encoded))):
        skip_gram.append(skipgrams(encoded[i], vocabulary_size=vocab_size, window_size=3))
        
        
    for _, elem in tqdm(enumerate(skip_gram)):
        if len(elem[1]) == 0:
            print(_,'is passed')
            continue
        first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        try:
            labels = np.array(elem[1].get(),dtype='int32')
        except:
            labels = np.array(elem[1],dtype='int32')
        for i in first_elem:
            first_em.append(i)
        for j in second_elem:
            second_em.append(j)
        for k in labels:
            y_123.append(k)
            
    second_em =np.array(second_em,dtype='int32')
    first_em =np.array(first_em,dtype='int32')
    y_123 = np.array(y_123,dtype='int32')


    with tf.device("/device:CPU:0"):
        pretrained_model.fit([first_em,second_em],y_123,epochs=epochs,batch_size=batch_size)
    pretrained_model.save(file_name)
    print(file_name,'is saved')
    
import gensim
def make_w2v(model,word2idx,file_name):
    if os.path.isdir(file_name):
        print('File exist change the name')
        raise
    
    f = open(file_name ,'w')
    f.write('{} {}\n'.format(len(word2idx)-1, 256))
    vectors = model.get_weights()[0]
    for word, i in word2idx.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()

    # 모델 로드
    w2v = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=False)
    return w2v
import pickle

## preprocessing before embedding data
def data_extract(train_set,tox_info,min_num=3,cutoff = False,word2idx = None,start_end = True):
    ## Train set = Molecule string data
    ## tox info = Label
    ## cutoff = if True, remove the molecule that contains atom that is not in the word2idx dictionary
    ## word2idx = dictionary of pre training model
    ## start_end = if True, add the start and end token
    
    
    
    train_set = train_set.copy()
    ## add the start and end
    if start_end:
        for i in train_set:
            i.insert(0,'<start>')
            i.insert(0,'<unknown1>')
        for i in train_set:
            i.append('<end>')
    else:
        for i in train_set:
            i.insert(0,'<unknown1>')
    
    ## remove less number tokens
    
    temp_dict = {}
    remove_list = []
    for i in train_set:
        for j in i:
            try:
                temp_dict[j] = temp_dict[j]+1
            except:
                temp_dict[j] = 1

    for j in temp_dict.keys():
        if temp_dict[j]<min_num:
            for i in range(len(train_set)):
                try:
                    train_set[i].remove(j)
                    remove_list.append(i)
                        
                except:
                    continue
    if cutoff:
        for _,i in enumerate(train_set):
            for j in i:
                try:
                    word2idx[j]
                except:
                    remove_list.append(_)
                    break        

    remove_list.sort(reverse=True)
    tox_info = list(tox_info)
    for _,i in enumerate(remove_list):
        if remove_list[_-1] == i:
            continue
        train_set.pop(i)
        tox_info.pop(i)
    tox_info = np.array(tox_info)    
    print('Deleted sentences :',len(remove_list))
    print('Sentences : ',len(train_set))
    
    return train_set,tox_info,remove_list