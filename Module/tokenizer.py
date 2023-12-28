import pickle
import atomInSmiles
from Module import RDK as rk
from rdkit import Chem
from tqdm import tqdm
from SmilesPE import tokenizer
import numpy as np



def Chem_generator(smiles):
    res_list = []
    for i in tqdm(smiles):
        mol = Chem.MolFromSmiles(i)
        temp = []
        index = 0
        while(len(set(temp))!=4 and index != 100):
            index+=1
            temp.append(Chem.MolToSmiles(mol,doRandom=True))
        res_list+=list(set(temp))
    return res_list


class Tokenizer():
    
    def __init__(self,SMILE_path=None,Random_number = 1):
        if SMILE_path is not None:
            print('Load SMILE file ....')
            with open(SMILE_path,'rb') as file:
                self.SMILE = pickle.load(file)
            print('Number of SMILEs : ',len(self.SMILE))        
        self.tokens = {'AIS':0,'SMILE':1,'SPE':2}
        
        
        
        if Random_number>1:
            temp = Chem_generator(self.SMILE)
            self.SMILE = temp
            
            
        
    def fit(self,token):
        try:
            self.tokens[token]
        except:
            print(f'Token is not suitable, suitable token : {list(self.tokens.keys())}')
            raise
        
        if self.tokens[token] == 0:
            print('start tokenize')
            result = [] 
            for item in tqdm(self.SMILE):
                result.append(atomInSmiles.encode(item,True).split(' '))
        elif self.tokens[token] == 1:
            print('start tokenize')
            result = []
            for item in tqdm(self.SMILE):
                result.append(tokenizer.atomwise_tokenizer(item))
        elif self.tokens[token] == 2:
            print('start tokenize')
            result = []
            result = rk.smile_tokenize(self.SMILE)
        
        self.tokens = list(result)
        
        return result
    
    def encode(self,dictionary_path,tokenized_path = None,update_dictionary = False):
        
        with open(dictionary_path,'rb') as file:
            word2idx = pickle.load(file)
            
        if tokenized_path is not None:
            with open(tokenized_path,'rb') as file:
                self.tokens = pickle.load(file)
    
        ## data process(remove over 180)
        
        remove_list = []
        for index,token in enumerate(self.tokens):
            if len(token)>180:
                remove_list.append(index)
        remove_list.sort(reverse=True)
        
        for remove in remove_list:
            self.tokens.pop(remove)
        
        
        
        
        
        def word_to_index(train_set,dict):
            result = []
            for molecule in tqdm(train_set):
                temp_list = []
                temp_list.append(1)
                for atom in molecule:
                    try:
                        temp_list.append(dict[atom])
                    except:
                        dict[atom] = len(dict)+1
                        temp_list.append(dict[atom])
                while len(temp_list)!=200:
                    temp_list.append(0)
                result.append(temp_list)
            return result
        print('embedding start')
        embedding_word = word_to_index(self.tokens,word2idx)   
        embedding_word = np.array(embedding_word)
    
        if update_dictionary:
            with open(dictionary_path,'wb') as file:
                pickle.dump(word2idx,file)
                
        return embedding_word
            
            

