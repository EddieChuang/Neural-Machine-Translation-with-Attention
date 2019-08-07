import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Translator import Translator 
from dataset_utils import download_dataset, load_dataset, max_length

def fit(translator, train_dataset, val_dataset, train_config, use_teacher_forcing=False):

    learning_rate = train_config['learning_rate']
    epoch_size = train_config['epoch_size']
    batch_size = train_config['batch_size']
    learning_rate = train_config['learning_rate']
    train_len = train_config['train_len']
    val_len = train_config['val_len']
    train_num_batch = train_len // batch_size
    val_num_batch = val_len // batch_size

    
    # create training dataset operations
    train_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = train_iterator.make_initializer(train_dataset)
    train_inp, train_tar = train_iterator.get_next()
    train_loss, train_op = translator.train(train_inp, train_tar, learning_rate, use_teacher_forcing)
    

    # create validation dataset operations
    val_iterator = tf.data.Iterator.from_structure(val_dataset.output_types, val_dataset.output_shapes)
    val_init_op = val_iterator.make_initializer(val_dataset)
    val_inp, val_tar = val_iterator.get_next()
    val_loss = translator.validate(val_inp, val_tar)


    translator.sess.run(tf.initializers.global_variables())
    translator.sess.run(val_init_op)
    val_inp_sents, val_tar_sents = translator.sess.run([val_inp, val_tar])

    val_inp_sents = [[translator.inp_lang.index_word[idx] for idx in sent] for sent in val_inp_sents]
    val_tar_sents = [[translator.tar_lang.index_word[idx] for idx in sent] for sent in val_tar_sents]


    for e in range(epoch_size):
        translator.sess.run(train_init_op)
        translator.sess.run(val_init_op)
        total_loss = 0
        tn = tqdm(total=train_len)
        tn.set_description('Epoch: {}/{}'.format(e + 1, epoch_size))
        for b in range(train_num_batch):
            batch_loss, _ = translator.sess.run([train_loss, train_op])
            total_loss += batch_loss

            tn.set_postfix(train_loss=batch_loss)
            tn.update(n=batch_size)
        
        val_loss_ = 0
        for b in range(val_num_batch):
            val_loss_ += translator.sess.run(val_loss)
            
        tn.set_postfix(train_loss=total_loss / train_num_batch, val_loss=val_loss_ / val_num_batch)


if __name__ == '__main__':

    filepath = './spa-eng/spa.txt'

    is_download = False
    if is_download:
        filepath = download_dataset()


    num_examples = 30000
    inp_sent, tar_sent, inp_lang, tar_lang = load_dataset(filepath, num_examples)
    inp_sent_train, inp_sent_val, tar_sent_train, tar_sent_val = train_test_split(inp_sent, tar_sent, test_size=0.2)
    train_len, val_len = len(inp_sent_train), len(inp_sent_val)
    max_length_tar, max_length_inp = max_length(tar_sent), max_length(inp_sent)
    
    model_config = {
        'max_length_inp': max_length_inp,
        'max_length_tar': max_length_tar,
        'src_vocab_size': len(inp_lang.word_index) + 1,
        'tgt_vocab_size': len(tar_lang.word_index) + 1,
        'src_emb_dim': 300,
        'tgt_emb_dim': 300,
        'enc_units': 512,
        'dec_units': 512,
        'att_units': 10,
        'score_type': 'general'
    }

    print(model_config)
    train_config = {
        'epoch_size': 20,
        'batch_size': 64,
        'learning_rate': 0.001,
        'train_len': train_len,
        'val_len': val_len
    }

    translator = Translator(model_config, inp_lang, tar_lang)

    train_dataset = tf.data.Dataset.from_tensor_slices((inp_sent_train, tar_sent_train)).shuffle(train_len)
    train_dataset = train_dataset.batch(train_config['batch_size'], drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((inp_sent_val, tar_sent_val)).shuffle(val_len)
    val_dataset = val_dataset.batch(train_config['batch_size'], drop_remainder=True)

    
    fit(translator, train_dataset, val_dataset, train_config, use_teacher_forcing=False)
    