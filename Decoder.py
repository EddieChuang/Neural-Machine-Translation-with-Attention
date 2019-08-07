import tensorflow as tf
import numpy as np
from Attention import *

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, enc_units, dec_units, att_units, score_type):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dec_units = dec_units
        
        self.embedding = tf.get_variable(
            name='tar_embedding',                  
            shape=(self.vocab_size, self.emb_dim),
            initializer=tf.truncated_normal_initializer(mean=0, stddev=1/np.sqrt(self.vocab_size)))
        self.Wo = tf.get_variable(name='Wo', shape=(self.dec_units, self.vocab_size))
        self.decoder_layer = self.__build_decoder_layer()
        # self.attention_layer = LuongAttention(att_units, enc_units, dec_units, score_type)
        self.attention_layer = BahdanauAttention(att_units, enc_units, dec_units)
    
    def __build_decoder_layer(self):
        layer = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True)
        return layer
    
    def call(self, dec_inp, hidden, enc_out):
        # context_vector shape == (batch_size, 1, enc_units)
        # attention_weights shape == (batch_size, max_src_len, 1)
        context_vector, attention_weights = self.attention_layer(hidden, enc_out, enc_out)
        context_vector = tf.expand_dims(context_vector, axis=1)
        
        dec_inp = tf.nn.embedding_lookup(self.embedding, dec_inp)  # (batch_size, 1, emb_dim)
        dec_inp = tf.concat([context_vector, dec_inp], axis=2)     # (batch_size, 1, emb_dim + enc_units)
        
        # output shape == (batch_size, dec_units)
        # hidden shape == (batch_size, dec_units)
        output, hidden, memory = self.decoder_layer(dec_inp)
        output = tf.reshape(output, shape=(-1, self.dec_units))
        
        output = tf.matmul(output, self.Wo)  # (batch_size, vocab_size)
        return output, hidden, attention_weights
         
        

if __name__ == '__main__':
    pass