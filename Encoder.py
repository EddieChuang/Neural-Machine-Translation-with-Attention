import tensorflow as tf
import numpy as np

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, enc_units):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_units = enc_units
        
        self.embedding = tf.get_variable(
            name='src_embedding', 
            shape=(self.vocab_size, self.emb_dim),
            initializer=tf.truncated_normal_initializer(mean=0, stddev=1/np.sqrt(self.vocab_size)))
        
        self.encoder_layer = self.__build_encoder_layer()
    
    
    def __build_encoder_layer(self):
        layer = tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True)        
        return layer
    
    
    def call(self, x):
        x = tf.nn.embedding_lookup(self.embedding, x)
        output, hidden, memory = self.encoder_layer(x)
        return output, hidden
    

if __name__ == '__main__':
    pass
    