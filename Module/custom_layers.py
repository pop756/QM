from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
import pickle
import copy
import tensorflow as tf
import tensornetwork as tn
import numpy as np
import tensornetwork as tn
class TNLayer(tf.keras.layers.Layer):
    def __init__(self,Tensor_dimention = 2):
        super(TNLayer, self).__init__()
        # Create the variables for the layer.


    
        initializer = tf.keras.initializers.RandomNormal(stddev=1/np.sqrt(4+Tensor_dimention-1))
        initializer1 = tf.keras.initializers.RandomNormal(stddev=1/np.sqrt(4+Tensor_dimention+Tensor_dimention-1))
        self.A1 = tf.Variable(initializer(shape=[4,Tensor_dimention,4]),name="a1", trainable=True)
        self.A2 = tf.Variable(initializer1(shape=[4,Tensor_dimention,Tensor_dimention,8]),name="a2", trainable=True)
        self.A3 = tf.Variable(initializer1(shape=[4,Tensor_dimention,Tensor_dimention,8]),name="a3", trainable=True)
        self.A4 = tf.Variable(initializer(shape=[4,Tensor_dimention,4]),name="a4", trainable=True)
        


        self.bias = tf.Variable(tf.zeros(shape=(256*4)), name="bias", trainable=True)
        
    def call(self, inputs):
        # Define the contraction.
        # We break it out so we can parallelize a batch using
        # tf.vectorized_map (see below).
        Nodes = [tn.Node(self.A1,'a1',backend="tensorflow")]
        Nodes+=[tn.Node(self.A2,f'a{2}',backend="tensorflow")]
        Nodes+=[tn.Node(self.A3,f'a{3}',backend="tensorflow")]
        Nodes+=[tn.Node(self.A4,f'a{4}',backend="tensorflow")]
        input_vec = tf.reshape(inputs, [-1,200,4,4,4,4])
        T_node = tn.Node(input_vec , backend="tensorflow",name = 't') 
        for i in range(len(Nodes)-1):
            if i == 0:
                Nodes[i][1]^Nodes[i+1][1]
            else:
                Nodes[i][2]^Nodes[i+1][1]
        for i in range(len(Nodes)):
            Nodes[i][0]^T_node[i+2]
        
        bias_var = self.bias
        contraction = T_node@Nodes[0]
        for i in range(1,len(Nodes)):
            contraction = contraction@Nodes[i]
        result = tf.reshape(contraction.tensor,[-1,200,256*4])
        result = result + bias_var
        return result
class TNLayer_small(tf.keras.layers.Layer):
    def __init__(self,Tensor_dimention = 2):
        super(TNLayer_small, self).__init__()
        # Create the variables for the layer.


    
        initializer = tf.keras.initializers.RandomNormal(stddev=1/np.sqrt(3+Tensor_dimention-1))
        initializer1 = tf.keras.initializers.RandomNormal(stddev=1/np.sqrt(4+Tensor_dimention+Tensor_dimention-1))
        self.A1 = tf.Variable(initializer(shape=[4,Tensor_dimention,4]),name="a1", trainable=True)
        self.A2 = tf.Variable(initializer1(shape=[4,Tensor_dimention,Tensor_dimention,4]),name="a2", trainable=True)
        self.A3 = tf.Variable(initializer1(shape=[4,Tensor_dimention,Tensor_dimention,4]),name="a3", trainable=True)
        self.A4 = tf.Variable(initializer(shape=[4,Tensor_dimention,3]),name="a4", trainable=True)
        


        self.bias = tf.Variable(tf.zeros(shape=(192)), name="bias", trainable=True)
        
    def call(self, inputs):
        # Define the contraction.
        # We break it out so we can parallelize a batch using
        # tf.vectorized_map (see below).
        Nodes = [tn.Node(self.A1,'a1',backend="tensorflow")]
        Nodes+=[tn.Node(self.A2,f'a{2}',backend="tensorflow")]
        Nodes+=[tn.Node(self.A3,f'a{3}',backend="tensorflow")]
        Nodes+=[tn.Node(self.A4,f'a{4}',backend="tensorflow")]
        input_vec = tf.reshape(inputs, [-1,200,4,4,4,4])
        T_node = tn.Node(input_vec , backend="tensorflow",name = 't') 
        for i in range(len(Nodes)-1):
            if i == 0:
                Nodes[i][1]^Nodes[i+1][1]
            else:
                Nodes[i][2]^Nodes[i+1][1]
        for i in range(len(Nodes)):
            Nodes[i][0]^T_node[i+2]
        
        bias_var = self.bias
        contraction = T_node@Nodes[0]
        for i in range(1,len(Nodes)):
            contraction = contraction@Nodes[i]
        result = tf.reshape(contraction.tensor,[-1,200,192])
        result = result + bias_var
        return result

class TokenAndPositionEmbedding_mask(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,start_index = 0):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,mask_zero=True)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim,mask_zero = True)
        
        self.positions = np.array([0]*start_index+[i+1 for i in range(200-start_index)])
    def call(self, x):
        positions = self.pos_emb(self.positions)
        mask = self.token_emb.compute_mask(x)
        x = self.token_emb(x)
        return x + positions,mask

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,start_index = 0):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,mask_zero=True)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim,mask_zero = True)
        self.positions = np.array([0]*start_index+[i+1 for i in range(200-start_index)])
    def call(self, x):
        positions = self.pos_emb(self.positions)
        x = self.token_emb(x)
        return x + positions


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,attention_mask=None):
        super().__init__()
        if attention_mask is not None:
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        else:
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs,attention_mask=None):
        attn_output = self.att(inputs, inputs,attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
class TransformerBlock_Tensor(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,Tensor_dimention=2):
        super().__init__()
        self.att = MultiheadAttention_tensor(d_model=embed_dim,num_heads=num_heads,Tensor_dimention=Tensor_dimention,dropout=0)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs,mask=None):
        attn_output,_ = self.att(inputs,mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
import tensorflow as tf

class TransformerBlock_Tensor_small(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,Tensor_dimention=2):
        super().__init__()
        self.att = MultiheadAttention_tensor_small(d_model=int(embed_dim*1/num_heads),num_heads=num_heads,Tensor_dimention=Tensor_dimention,dropout=0)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(128),tf.keras.layers.Dense(ff_dim, activation="relu"),tf.keras.layers.Dense(128), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs,mask=None):
        attn_output,_ = self.att(inputs,mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
import tensorflow as tf



import tensorflow as tf


class MultiheadAttention_tensor_small(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads,classical_rate = 1/4,Tensor_dimention = 2,dropout=0):
        super(MultiheadAttention_tensor_small, self).__init__()
        #assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model
        self.rate = classical_rate
        assert int(num_heads/classical_rate) == num_heads/classical_rate



        self.wq = tf.keras.layers.Dense(int(d_model)*int(num_heads*classical_rate))
        self.wk = tf.keras.layers.Dense(int(d_model)*int(num_heads*classical_rate))
        self.wv = tf.keras.layers.Dense(int(d_model)*int(num_heads*classical_rate))
        self.dropout = dropout
        self.wq_tensor = TNLayer_small(Tensor_dimention)
        self.wk_tensor = TNLayer_small(Tensor_dimention)
        self.wv_tensor = TNLayer_small(Tensor_dimention)

        self.dense = tf.keras.layers.Dense(d_model*num_heads,kernel_initializer='lecun_normal')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, int(self.num_heads*self.rate), 32))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def split_heads_tensor(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, int(self.num_heads*(1-self.rate)), 32))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, q, mask):
        batch_size = tf.shape(q)[0]
        k = q
        v = q
        
        q_class = self.split_heads(self.wq(q), batch_size)

        
        k_class = self.split_heads(self.wk(k), batch_size)

        
        v_class = self.split_heads(self.wv(v), batch_size)
        
        
        q_tensor = self.split_heads_tensor(self.wq_tensor(q),batch_size)

        
        k_tensor = self.split_heads_tensor(self.wk_tensor(k),batch_size)

        
        v_tensor = self.split_heads_tensor(self.wv_tensor(v),batch_size)

        q,k,v = tf.concat([q_class,q_tensor],axis=1),tf.concat([k_class,k_tensor],axis=1),tf.concat([v_class,v_tensor],axis=1)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.depth*self.num_heads))

        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        scaled_attention_logits = tf.transpose(scaled_attention_logits,perm=[1,0,2,3])
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        scaled_attention_logits = tf.transpose(scaled_attention_logits,perm=[1,0,2,3])
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = tf.keras.layers.Dropout(self.dropout)(attention_weights)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights




class MultiheadAttention_tensor(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads,classical_rate = 1/3,Tensor_dimention = 2,dropout=0):
        super(MultiheadAttention_tensor, self).__init__()
        #assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model
        self.rate = classical_rate
        assert int(num_heads/classical_rate) == num_heads/classical_rate



        self.wq = tf.keras.layers.Dense(int(d_model)*int(num_heads*classical_rate))
        self.wk = tf.keras.layers.Dense(int(d_model)*int(num_heads*classical_rate))
        self.wv = tf.keras.layers.Dense(int(d_model)*int(num_heads*classical_rate))
        self.dropout = dropout
        self.wq_tensor = TNLayer(Tensor_dimention)
        self.wk_tensor = TNLayer(Tensor_dimention)
        self.wv_tensor = TNLayer(Tensor_dimention)

        self.dense = tf.keras.layers.Dense(d_model,kernel_initializer='lecun_normal')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, int(self.num_heads*self.rate), self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def split_heads_tensor(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, int(self.num_heads*(1-self.rate)), self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, q, mask):
        batch_size = tf.shape(q)[0]
        k = q
        v = q
        
        q_class = self.split_heads(self.wq(q), batch_size)

        
        k_class = self.split_heads(self.wk(k), batch_size)

        
        v_class = self.split_heads(self.wv(v), batch_size)
        
        
        q_tensor = self.split_heads_tensor(self.wq_tensor(q),batch_size)

        
        k_tensor = self.split_heads_tensor(self.wk_tensor(k),batch_size)

        
        v_tensor = self.split_heads_tensor(self.wv_tensor(v),batch_size)

        q,k,v = tf.concat([q_class,q_tensor],axis=1),tf.concat([k_class,k_tensor],axis=1),tf.concat([v_class,v_tensor],axis=1)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.depth*self.num_heads))
        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        scaled_attention_logits = tf.transpose(scaled_attention_logits,perm=[1,0,2,3])
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        scaled_attention_logits = tf.transpose(scaled_attention_logits,perm=[1,0,2,3])
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = tf.keras.layers.Dropout(self.dropout)(attention_weights)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    
    
def Attention_mask(embedding_mask):
    embedding_mask = tf.logical_not(embedding_mask)
    embedding_mask = tf.cast(embedding_mask,tf.float32)
    embedding_mask = tf.expand_dims(embedding_mask,axis=-1)
    embedding_mask = tf.tile(embedding_mask,[1,1,200])
    embedding_mask = tf.transpose(embedding_mask,perm=[0,2,1])
    return embedding_mask

class BERT_tensor(tf.keras.layers.Layer):
    def __init__(self,emb_dim,num_heads,ff_dim,CL_num = 3,strat_index = 1,Tensor_dimention = 2):
        super(BERT_tensor, self).__init__()
        self.encoder = [TransformerBlock_Tensor(emb_dim,num_heads,ff_dim,Tensor_dimention=Tensor_dimention) for i in range(8)]
        #self.encoder = TransformerBlock_Tensor(emb_dim,num_heads,ff_dim)
        self.embedding = TokenAndPositionEmbedding_mask(200,3500,256,start_index=strat_index)
        self.classify = layers.Dense(CL_num,activation = 'softmax')
    def call(self, inputs, mask_index=None,pretrain = False,att_mask = None):
        if pretrain:
            mask_index = tf.one_hot(mask_index,200)
            boolean_mask = tf.cast(tf.reduce_sum(mask_index,axis=1),bool)
            inputs = tf.cast(inputs,dtype=tf.int32)
            
        hidden,pad_mask = self.embedding(inputs)
        Att_mask = Attention_mask(pad_mask)
        if att_mask is not None:
            Att_mask = (Att_mask+att_mask)%2
        
        for i in range(8):
            hidden = self.encoder[i](hidden,Att_mask)
    
        if pretrain:
            output = tf.reshape(hidden,[-1,200,256])
            output = self.classify(output)
            output = tf.boolean_mask(output,boolean_mask)
            return output
        else:
            return hidden
        
class BERT_tensor_small(tf.keras.layers.Layer):
    def __init__(self,emb_dim,num_heads,ff_dim,CL_num = 3,strat_index = 1,Tensor_dimention = 2):
        super(BERT_tensor_small, self).__init__()
        self.encoder = [TransformerBlock_Tensor_small(emb_dim,num_heads,ff_dim,Tensor_dimention=Tensor_dimention) for i in range(8)]
        #self.encoder = TransformerBlock_Tensor(emb_dim,num_heads,ff_dim)
        self.embedding = TokenAndPositionEmbedding_mask(200,3500,256,start_index=strat_index)
        self.classify = layers.Dense(CL_num,activation = 'softmax')
    def call(self, inputs, mask_index=None,pretrain = False,att_mask = None):
        if pretrain:
            mask_index = tf.one_hot(mask_index,200)
            boolean_mask = tf.cast(tf.reduce_sum(mask_index,axis=1),bool)
            inputs = tf.cast(inputs,dtype=tf.int32)
            
        hidden,pad_mask = self.embedding(inputs)
        Att_mask = Attention_mask(pad_mask)
        if att_mask is not None:
            Att_mask = (Att_mask+att_mask)%2
        
        for i in range(8):
            hidden = self.encoder[i](hidden,Att_mask)
    
        if pretrain:
            output = tf.reshape(hidden,[-1,200,256])
            output = self.classify(output)
            output = tf.boolean_mask(output,boolean_mask)
            return output
        else:
            return hidden
        
class BERT(tf.keras.layers.Layer):
    def __init__(self,emb_dim,num_heads,ff_dim,CL_num = 141,strat_index = 1):
        super(BERT, self).__init__()
        self.encoder = [TransformerBlock(emb_dim,num_heads,ff_dim) for i in range(8)]
        #self.encoder = TransformerBlock(emb_dim,num_heads,ff_dim)
        self.embedding = TokenAndPositionEmbedding(200,3500,256,start_index=strat_index)
        self.classify = tf.keras.layers.Dense(CL_num,activation = 'softmax')
    def call(self, inputs, mask_index,pretrain = False,att_mask = None):
        if pretrain:
            mask_index = tf.one_hot(mask_index,200)
            boolean_mask = tf.cast(tf.reduce_sum(mask_index,axis=1),bool)
            inputs = tf.cast(inputs,dtype=tf.int32)

        hidden = self.embedding(inputs)
        for i in range(8):
            hidden = self.encoder[i](hidden,attention_mask= att_mask)
    
        if pretrain:
            output = tf.reshape(hidden,[-1,200,256])
            output = self.classify(output)
            output = tf.boolean_mask(output,boolean_mask)
            return output
        else:
            return hidden
        
class BERT_tensor_small_GPU(tf.keras.layers.Layer):
    def __init__(self,emb_dim,num_heads,ff_dim,CL_num = 3,strat_index = 1,Tensor_dimention = 2):
        super(BERT_tensor_small_GPU, self).__init__()
        self.encoder = [TransformerBlock_Tensor_small_GPU(emb_dim,num_heads,ff_dim,Tensor_dimention=Tensor_dimention) for i in range(8)]
        #self.encoder = TransformerBlock_Tensor(emb_dim,num_heads,ff_dim)
        self.embedding = TokenAndPositionEmbedding_mask(200,3500,256,start_index=strat_index)
        self.classify = layers.Dense(CL_num,activation = 'softmax')
    def call(self, inputs, mask_index=None,pretrain = False,att_mask = None):
        if pretrain:
            mask_index = tf.one_hot(mask_index,200)
            boolean_mask = tf.cast(tf.reduce_sum(mask_index,axis=1),bool)
            inputs = tf.cast(inputs,dtype=tf.int32)
            
        hidden,pad_mask = self.embedding(inputs)
        Att_mask = Attention_mask(pad_mask)
        if att_mask is not None:
            Att_mask = (Att_mask+att_mask)%2
        
        for i in range(8):
            hidden = self.encoder[i](hidden,Att_mask)
    
        if pretrain:
            output = tf.reshape(hidden,[-1,200,256])
            output = self.classify(output)
            output = tf.boolean_mask(output,boolean_mask)
            return output
        else:
            return hidden
        
class TransformerBlock_Tensor_small_GPU(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,Tensor_dimention=2):
        super().__init__()
        self.att = MultiheadAttention_tensor_small_GPU(d_model=int(embed_dim*1/num_heads),num_heads=num_heads,Tensor_dimention=Tensor_dimention,dropout=0)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(128),tf.keras.layers.Dense(ff_dim, activation="relu"),tf.keras.layers.Dense(128), tf.keras.layers.Dense(embed_dim),]
            #[tf.keras.layers.Dense(ff_dim, activation="relu"),tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs,mask=None):
        attn_output,_ = self.att(inputs,mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
import tensorflow as tf



import tensorflow as tf


class MultiheadAttention_tensor_small_GPU(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads,classical_rate = 1/4,Tensor_dimention = 2,dropout=0):
        super(MultiheadAttention_tensor_small_GPU, self).__init__()
        #assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model
        self.rate = classical_rate
        assert int(num_heads/classical_rate) == num_heads/classical_rate



        self.wq = tf.keras.layers.Dense(int(d_model)*int(num_heads*classical_rate)*3,use_bias=False)
        self.dropout = dropout
        self.wq_tensor = TNLayer_small_GPU(Tensor_dimention)
        self.bias = tf.Variable(tf.zeros(shape=(3,64)), name="bias", trainable=True)
        self.dense = tf.keras.layers.Dense(d_model*num_heads,kernel_initializer='lecun_normal')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, int(self.num_heads*self.rate), 32))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def split_heads_tensor(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, int(self.num_heads*(1-self.rate)), 32))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, q, mask):
        batch_size = tf.shape(q)[0]
        Q = self.wq(q)
        Q = tf.reshape(Q,(batch_size,tf.shape(Q)[1],3,-1))
        Q = Q + self.bias
        Q_T = self.wq_tensor(q)
        q_ = Q[:,:,0]
        k_ = Q[:,:,1]
        v_ = Q[:,:,2]
        
        q_t = Q_T[:,:,0]
        k_t = Q_T[:,:,1]
        v_t = Q_T[:,:,2]
        
        q_class = self.split_heads(q_, batch_size)

        
        k_class = self.split_heads(k_, batch_size)

        
        v_class = self.split_heads(v_, batch_size)
        
        
        q_tensor = self.split_heads_tensor(q_t,batch_size)

        
        k_tensor = self.split_heads_tensor(k_t,batch_size)

        
        v_tensor = self.split_heads_tensor(v_t,batch_size)

        q,k,v = tf.concat([q_class,q_tensor],axis=1),tf.concat([k_class,k_tensor],axis=1),tf.concat([v_class,v_tensor],axis=1)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.depth*self.num_heads))

        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        scaled_attention_logits = tf.transpose(scaled_attention_logits,perm=[1,0,2,3])
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        scaled_attention_logits = tf.transpose(scaled_attention_logits,perm=[1,0,2,3])
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = tf.keras.layers.Dropout(self.dropout)(attention_weights)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
class TNLayer_small_GPU(tf.keras.layers.Layer):
    def __init__(self,Tensor_dimention = 2):
        super(TNLayer_small_GPU, self).__init__()
        # Create the variables for the layer.


    
        initializer = tf.keras.initializers.RandomNormal(stddev=1/np.sqrt(3+Tensor_dimention-1))
        initializer1 = tf.keras.initializers.RandomNormal(stddev=1/np.sqrt(4+Tensor_dimention+Tensor_dimention-1))
        initializer2 = tf.keras.initializers.RandomNormal(stddev=1/np.sqrt(5+Tensor_dimention-1))
        self.A1 = tf.Variable(initializer(shape=[4,Tensor_dimention,4]),name="a1", trainable=True)
        self.A2 = tf.Variable(initializer1(shape=[4,Tensor_dimention,Tensor_dimention,4]),name="a2", trainable=True)
        self.A3 = tf.Variable(initializer1(shape=[4,Tensor_dimention,Tensor_dimention,4]),name="a3", trainable=True)
        self.A4 = tf.Variable(initializer2(shape=[4,Tensor_dimention,9]),name="a4", trainable=True)
        


        self.bias = tf.Variable(tf.zeros(shape=(3,192)), name="bias", trainable=True)
        
    def call(self, inputs):
        # Define the contraction.
        # We break it out so we can parallelize a batch using
        # tf.vectorized_map (see below).
        Nodes = [tn.Node(self.A1,'a1',backend="tensorflow")]
        Nodes+=[tn.Node(self.A2,f'a{2}',backend="tensorflow")]
        Nodes+=[tn.Node(self.A3,f'a{3}',backend="tensorflow")]
        Nodes+=[tn.Node(self.A4,f'a{4}',backend="tensorflow")]
        input_vec = tf.reshape(inputs, [-1,200,4,4,4,4])
        T_node = tn.Node(input_vec , backend="tensorflow",name = 't') 
        for i in range(len(Nodes)-1):
            if i == 0:
                Nodes[i][1]^Nodes[i+1][1]
            else:
                Nodes[i][2]^Nodes[i+1][1]
        for i in range(len(Nodes)):
            Nodes[i][0]^T_node[i+2]
        bias_var = self.bias
        contraction = T_node@Nodes[0]
        for i in range(1,len(Nodes)):
            contraction = contraction@Nodes[i]
        result = tf.reshape(contraction.tensor,[-1,200,3,192])
        result = result + bias_var
        return result