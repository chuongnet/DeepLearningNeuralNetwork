from string import punctuation
from nltk.corpus import stopwords
from nltk import sent_tokenize
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from pickle import dump


def load_doc(fn):
    with open(fn, 'r') as file:
        text = file.read()
        file.close()
    return text


def clean_doc(doc):
    #doc.replace('--', ' ')
    re_punc = re.compile('[%s]' % re.escape(punctuation))
    #sentences = sent_tokenize(doc)
    doc = doc.split()
    tokens = [w.lower() for w in doc]
    # tokens = [[w for w in sent if w.isalpha()] for sent in tokens]
    tokens = [re_punc.sub('', w) for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in set(stopwords.words('english'))] 
    tokens = [[w for w in tokens if len(w) > 2]
    return tokens

              
def clean_corpus(corpus):
    corpus = sent_tokenize(corpus)
    corpus = [clean_doc(sent) for sent in corpus]
    return corpus
              

def create_vocab(corpus):
    vocab = Counter()
    for tokens in corpus:
        vocab.update(tokens)
    kw = [k for k, v in vocab.items() if v > 1]
    return kw


def tokens_to_lines(tokens):
    lines = list()
    for sent in tokens:
        line = ' '.join(sent)
        lines.append(line)
    doc = '\n'.join(lines)
    return doc


def lines_to_sequence(lines):
    tokenizer = Tokenizer()
    lines = lines.split('\n')
    tokenizer.fit_on_texts(lines)
    sequence = tokenizer.fit_on_sequences(lines)
    return sequence


def lines_to_metrics(lines):
    tokenizer = Tokenizer()
    lines = lines.split('\n')
    tokenizer.fit_on_texts(lines)
    metrics = tokenizer.texts_to_matrix(lines)
    return metrics


def save_file(fn, doc):
    file = open(fn, 'w')
    file.write(doc)
    file.close()
    return 'success'


def dump_file(fn, data):
    file = open(fn, 'wb')
    dump(data, file)
    file.close()
    return 'success'


# if __name__ == '__main__':
#     fn = 'data/republic_clean.txt'
#     doc = load_doc(fn)
#     tokens = clean_doc(doc)
#     print(tokens[:5])