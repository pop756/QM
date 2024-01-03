from SmilesPE import tokenizer
from rdkit import Chem
from tqdm import tqdm
from tdc.single_pred import Tox
import pickle
import tensorflow as tf 

with open('./BERT/SMILE/1M_random_ZINC_word2index.pkl','rb') as file:
    word2idx = pickle.load(file)


## Make Randomized smiles from canonical form
def Chem_generator(smiles,label):
    res_list = []
    len_20 = []
    train_label = []
    index = 0
    for i in tqdm(smiles):
        mol = Chem.MolFromSmiles(i)
        temp = []
        index = 0
        while(len(set(temp))!=20 and index != 100):
            index+=1
            temp.append(Chem.MolToSmiles(mol,doRandom=True))
        temp_res = list(set(temp))
        res_list.append(temp_res)
        len_20.append(temp_res)
        train_label.append([label[index]]*len(temp_res))
    return res_list,len_20,train_label

def flatten(input_list):
    res = []
    for seq in input_list:
        res+=seq
    return res
    
    

## Preprocess for SMILE
def Preprocess(smiles,label,train = False):
    smiles,len_20,train_label = Chem_generator(smiles,label)
    image = []
    for smile in smiles:
        part_image = []
        for single_smile in smile:
            temp = []
            single_smile = tokenizer.atomwise_tokenizer(smile)
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
        train_image.append(temp)
    if train:
        train_image = flatten(train_image)
        train_label = flatten(train_label)
        return train_image,train_label
        
    
    return train_image, train_label, len_20
    
def main():
    data = Tox(name = 'herg')
    split = data.get_split(method='scaffold',frac = [0.8,0.1,0.1])
    print(split['test'])
    Preprocess(split['test']['Drug'],split['test']['Y'])
    
    
main()
    