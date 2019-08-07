from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tensorflow as tf
import numpy as np
from Encoder import Encoder
from Decoder import Decoder
from dataset_utils import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings("ignore")

class Translator:
    def __init__(self, config, inp_lang, tar_lang):
        tf.reset_default_graph()
        
        self.max_length_inp = config['max_length_inp']
        self.max_length_tar = config['max_length_tar']
        self.src_vocab_size = config['src_vocab_size']
        self.tgt_vocab_size = config['tgt_vocab_size']
        self.src_emb_dim = config['src_emb_dim']
        self.tgt_emb_dim = config['tgt_emb_dim']
        self.enc_units = config['enc_units']
        self.dec_units = config['dec_units']
        self.att_units = config['att_units']
        self.score_type = config['score_type']
        
        
        self.inp_lang = inp_lang
        self.tar_lang = tar_lang
        
        self.__build()
    
    
    def __build(self):
        self.encoder = Encoder(self.src_vocab_size, self.src_emb_dim, self.enc_units)
        self.decoder = Decoder(self.tgt_vocab_size, self.tgt_emb_dim, self.enc_units, self.dec_units, self.att_units, self.score_type)
        self.sess = tf.Session()
    
    
    def loss_function(self, y, y_hat):
        
        # (batch_size, tgt_vocab_size)
        y_onehot = tf.reshape(tf.one_hot(y, depth=self.tgt_vocab_size), (-1, self.tgt_vocab_size))          
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y_onehot)  # (batch_size, )
        
        # masking padding loss
        mask = tf.cast(tf.not_equal(y, 0), dtype=tf.float32)  # (batch_size, 1), 0: <pad>
        loss = tf.reduce_mean(ce * mask) 
        
        return loss
    
            
    def train(self, inp_batch, tar_batch, learning_rate, use_teacher_forcing):
        
        batch_size = inp_batch.get_shape().as_list()[0]
        
        dec_inp = tf.expand_dims([self.tar_lang.word_index['<sos>']] * batch_size, axis=1)  # (batch_size, 1)
        outs = tf.expand_dims([self.tar_lang.word_index['<sos>']] * batch_size, axis=1)     # (batch_size, 1)
        
        loss = 0.0
        enc_out, enc_hidden = self.encoder(inp_batch)
        dec_hidden = enc_hidden
        for t in range(1, tar_batch.shape[1]):
            dec_out, dec_hidden, _ = self.decoder(dec_inp, dec_hidden, enc_out)  # (batch_size, tar_vocab_size)
            loss += self.loss_function(tf.slice(tar_batch, [0, t], [batch_size, 1]), dec_out)
            
            if use_teacher_forcing:
                # tf.slice(...) == tar_batch[:, t]
                dec_inp = tf.slice(tar_batch, [0, t], [batch_size, 1])  # (batch_size, 1)
            else:
                dec_inp = tf.expand_dims(tf.argmax(dec_out, axis=1, output_type=tf.int32), axis=1)  # (batch_size, 1)
                                
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

        return loss / int(tar_batch.shape[1]), train_op
    
    def validate(self, inp, tar):
        batch_size = inp.get_shape().as_list()[0]
        
        dec_inp = tf.expand_dims([self.tar_lang.word_index['<sos>']] * batch_size, axis=1)  # (batch_size, 1)
#         outs = tf.expand_dims([self.tar_lang.word_index['<sos>']] * batch_size, axis=1)     # (batch_size, 1)
        
        loss = 0.0
        enc_out, enc_hidden = self.encoder(inp)
        dec_hidden = enc_hidden
        for t in range(1, tar.shape[1]):
            dec_out, dec_hidden, _ = self.decoder(dec_inp, dec_hidden, enc_out)  # (batch_size, tar_vocab_size)
            loss += self.loss_function(tf.slice(tar, [0, t], [batch_size, 1]), dec_out)
            
            out = tf.expand_dims(tf.argmax(dec_out, axis=1, output_type=tf.int32), axis=1)  # (batch_size, 1)
#             outs = tf.concat([outs, out], axis=1)
            
            dec_inp = out
        
        return loss / int(tar.shape[1])
    
    
    def translate(self, inp_sents):
        
        inputs = []
        max_len = max_length(inp_sents)
        for inp_sent in inp_sents:
            inp_sent = ' '.join(inp_sent)
            inp_sent = preprocess_sentence(inp_sent)
            inp = [self.inp_lang.word_index.get(i, 0) for i in inp_sent.split(' ')]
            inputs.append(inp)
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len, padding='post')
        
        
        enc_inp = tf.convert_to_tensor(inputs)
        enc_out, enc_hidden = self.encoder(enc_inp)

        dec_hidden = enc_hidden
        dec_inp = tf.expand_dims([self.tar_lang.word_index['<sos>']] * len(inp_sents), axis=1)  # (batch_size, 1)
        
        outs = tf.expand_dims([self.tar_lang.word_index['<sos>']] * len(inp_sents), axis=1)     # (batch_size, 1)
        for t in range(self.max_length_tar):
            dec_out, dec_hidden, attention_weights = self.decoder(dec_inp, dec_hidden, enc_out)

            out = tf.expand_dims(tf.argmax(dec_out, axis=1, output_type=tf.int32), axis=1)  # (batch_size, 1)
            outs = tf.concat([outs, out], axis=1)

            dec_inp = out
    
        translateds = self.sess.run([outs])
        for t in translateds:
            translateds.append([[self.tar_lang.index_word[idx] for idx in sent if idx == len(self.tar_lang.index_word)] 
                                for sent in t])
        
        return translateds
    
    
    def evaluate(self, inp_sents, tar_sents):
        
        translateds = self.translate(inp_sents)
        
        bleu = 0.0
        for translated, tar_sent in zip(translateds, tar_sents):
            bleu += sentence_bleu([tar_sent], translated, weights=(0.5, 0.5, 0, 0))

        return bleu / len(inp_sents), translateds
        
    