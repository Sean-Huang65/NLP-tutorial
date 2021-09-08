import unicodedata
import string
import re
from config import *
import torch

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3 # Count SOS and EOS
      
    def index_words(self, sentence):
        if self.name == 'cn':    
            for word in sentence:
                self.index_word(word)
        else:
            for word in sentence.split(' '):
                self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def word2index_method(self, idx):
        if idx in self.word2index:
            return self.word2index[idx]
        return 2 # UNK


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?，。？]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./data/%s-%s_train.txt' % (lang1, lang2)).read().strip().split('\n')
    # lines = open('./data/%s-%s_train.txt' % (lang1, lang2)).readlines()
    
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs

def filter_pair(p):
    return len(p[1].split(' ')) < MAX_LENGTH and len(p[0]) < MAX_LENGTH + 10

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def filter_sentences(sentences):
    return [sen for sen in sentences if len(sen.split(' ') < MAX_LENGTH)]

def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))
    
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs

def get_labels():
    lines = open('%s-%s.txt' % ('cn', 'eng')).read().strip().split('\n')
    
    lines = filter_pairs(lines)
    # Split every line into pairs and normalize
    labels = [normalize_string(l.split('\t')[1]) for l in lines]
    fout = open('./cn-eng_eng.txt', 'w')
    for line in labels:
        fout.write(line+'\n')

def indexes_from_sentence(lang, sentence):
    if lang.name == 'cn':
        return [lang.word2index_method(word) for word in sentence]
    else:
        return [lang.word2index_method(word) for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = torch.LongTensor(indexes).view(-1, 1)
    if USE_CUDA: var = var.cuda()
    return var

# Return a list of indexes, one for each word in the sentence
def variables_from_pair(pair, input_lang, output_lang):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)

def main():
    get_labels()

if __name__ == '__main__':
    main()
