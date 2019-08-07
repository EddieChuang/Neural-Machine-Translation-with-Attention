import tensorflow as tf

class BaseAttention(tf.keras.Model):
    def __init__(self):
        super(BaseAttention, self).__init__()
    
    def call(self, query, keys, values):
        '''
        query shape  == (batch_size, dec_units)
        keys shape   == (batch_size, max_src_len, enc_units)
        values shape == (batch_size, max_src_len, enc_units)
        '''
        
        score = self.score_function(query, keys)          # (batch_size, max_src_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, max_src_len, 1)
        
        context_vector = attention_weights * values            # (batch_size, max_src_len, enc_units)
        context_vector = tf.reduce_sum(context_vector, axis=1) # (batch_size, enc_units)
        
        return context_vector, attention_weights

    
class BahdanauAttention(BaseAttention):
    def __init__(self, att_units, enc_units, dec_units):
        super(BahdanauAttention, self).__init__()

        self.W1 = tf.get_variable(name='W1', shape=(dec_units, att_units))
        self.W2 = tf.get_variable(name='W2', shape=(enc_units, att_units))
        self.V = tf.get_variable(name='V', shape=(att_units, 1))
    
    
    def score_function(self, query, keys):
        '''
        query shape == (batch_size, dec_units)
        keys  shape == (batch_size, max_src_len, enc_units)
        '''
        
        q = tf.matmul(query, self.W1)  # (batch_size, att_units)
        q = tf.expand_dims(q, axis=1)  # (batch_size, 1, att_units)
        k = tf.matmul(keys, self.W2)   # (batch_size, max_src_len, att_units)
        s = tf.matmul(tf.nn.tanh(q + k), self.V)  # (batch_size, max_src_len, 1)
        
        return s  # (batch_size, max_src_len, 1)


class LuongAttention(BaseAttention):
    def __init__(self, att_units, enc_units, dec_units, score_type='general'):
        super(LuongAttention, self).__init__()
        
        self.score_type = score_type
        if self.score_type == 'general':
            self.W = tf.get_variable(name='W', shape=(dec_units, enc_units))
        elif self.score_type == 'concat':
            self.W = tf.get_variable(name='W', shape=(enc_units + dec_units, att_units))
            self.V = tf.get_variable(name='V', shape=(att_units, 1))
    
    def score_function(self, query, keys):
        '''
        query shape == (batch_size, dec_units)
        keys  shape == (batch_size, max_src_len, enc_units)
        '''
        if self.score_type == 'dot':
            # if score_type is 'dot', enc_units and dec_units must be equal
            s = tf.expand_dims(query, axis=1) * keys      # (batch_size, max_src_len, dec_units)
            s = tf.reduce_sum(s, axis=2, keepdims=True)  # (batch_size, max_src_len, 1)
        elif self.score_type == 'general':
            s = tf.expand_dims(tf.matmul(query, self.W), axis=1)  # (batch_size, 1, enc_units)
            s = tf.matmul(keys, s, transpose_b=True)              # (batch_size, max_src_len, 1)
        elif self.score_type =='concat':
            max_src_len = tf.shape(keys)[1]
            q = tf.tile(tf.expand_dims(query, axis=1), [1, max_src_len,1])  # (batch_size, max_src_len, dec_units)
            h = tf.concat([keys, q], axis=2)                                # (batch_size, max_src_len, enc_units + dec_units)
            s = tf.matmul(tf.nn.tanh(tf.matmul(h, self.W)), self.V)         # (batch_size, max_src_len, 1)
        
        
        return s  # (batch_size, max_src_len, 1)



if __name__ == '__main__':
    pass