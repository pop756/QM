import pickle
from tokenizer import Tokenizer

class Pre_train_model(tf.keras.Model):
    def __init__(self, token,input_data,dictionary_path, tokenized = False):
        super(Pre_train_model, self).__init__()
        
        self.tokens = {'AIS':0,'SMILE':1,'SPE':2}
        try:
            self.tokens[token]
        except:
            print(f'Token is not suitable, suitable token : {list(self.tokens.keys())}')
            raise
            
        if tokenized == False:
            tokenizer = Tokenizer(input_data)
            tokenizer.fit(token=token)
            self.data = tokenizer.encode(dictionary_path=dictionary_path)  
        
        else:
            tokenizer = Tokenizer()
            self.data = tokenizer.encode(dictionary_path=dictionary_path,tokenized_path=input_data)
        
    def MLM_processing(self):
        