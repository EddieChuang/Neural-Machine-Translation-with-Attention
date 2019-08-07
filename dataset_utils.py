import tensorflow as tf
import unicodedata
import re
import io
import os

def download_dataset():
    path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)
    filepath = os.path.dirname(path_to_zip) + '/spa-eng/spa.txt'

    print('Dataset is in ' + filepath)
    return filepath

def unicode_to_ascii(s):
    # remove the accents
    # https://www.itread01.com/content/1550449117.html
    # NFD: Normalization Form D, 
    # Mn: Mark, non-spacing
    
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<sos> ' + w + ' <eos>'
    return w


def read_dataset_from_file(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    lang_tokenizer.index_word[0] = '<unk>'
    lang_tokenizer.word_index['<unk>'] = 0

    sent = lang_tokenizer.texts_to_sequences(lang)

    sent = tf.keras.preprocessing.sequence.pad_sequences(sent, padding='post')

    return sent, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    tar_lang, inp_lang = read_dataset_from_file(path, num_examples)

    inp_sent, inp_lang_tokenizer = tokenize(inp_lang)
    tar_sent, tar_lang_tokenizer = tokenize(tar_lang)

    return inp_sent, tar_sent, inp_lang_tokenizer, tar_lang_tokenizer


def max_length(sentences):
    return max(len(sent) for sent in sentences)