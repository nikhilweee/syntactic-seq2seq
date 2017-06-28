from __future__ import print_function
import time
import nltk
import operator
import pickle
import argparse
import codecs
from spacy.en.language_data import STOP_WORDS

# The POS Tag - operator mapping for every word.
OPERATORS = {
    'VBD': '_ed',
    'NNS': '_s',
    'VBN': '_ed',
    'VBZ': '_s',
    'VBG': '_ing',
    'JJS': '_est',
    'JJR': '_er',
    'RBR': '_er',
    'RBS': '_est',
}

# For every tag returned by nltk.pos_tag, this is the corresponding
# nltk.corpus.wordnet.reader.* tag used by the lemmatizer.
TAGS = {
    'V': nltk.corpus.reader.wordnet.VERB,
    'N': nltk.corpus.reader.wordnet.NOUN,
    'J': nltk.corpus.reader.wordnet.ADJ,
    'R': nltk.corpus.reader.wordnet.ADV,
}

lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize(src_delemmatized, tgt_lemmatized, vocab):
    all_tags = {}
    with codecs.open(src_delemmatized, encoding='utf-8') as src, codecs.open(tgt_lemmatized, 'w', encoding='utf-8') as tgt:
        print('Tagging {} ...'.format(src_delemmatized))
        for line in src:
            tokens = line.split()
            tags = nltk.pos_tag(tokens)
            words = []
            for (word, tag) in tags:
                all_tags[tag] = all_tags[tag] + 1 if tag in all_tags else 1
                if tag in OPERATORS:
                    word_lemma = lemmatizer.lemmatize(word, pos=TAGS[tag[0]])
                    # add this transformation to vocab
                    vocab[word_lemma] = {} if word_lemma not in vocab else vocab[word_lemma]
                    vocab[word_lemma][OPERATORS[tag]] = word
                    word = word_lemma + ' ' + OPERATORS[tag]
                words.append(word)
            tgt.write(' '.join(words) + '\n')
        print('Tagging Complete.')
        all_tags_sorted = sorted(
            all_tags.items(), key=operator.itemgetter(1), reverse=True)
    return vocab, all_tags_sorted

def delemmatize(src_lemmatized, tgt_delemmatized, vocab):
    with codecs.open(src_lemmatized, encoding='utf-8') as src, codecs.open(tgt_delemmatized, 'w', encoding='utf-8') as tgt:
        for src_line in src:
            src_words = src_line.split()
            tgt_words = []
            for idx, word in enumerate(src_words):
                if word in OPERATORS.values():
                    op = word
                    lemma = src_words[idx - 1]
                    if lemma in vocab:
                        if op in vocab[lemma]:
                            word = vocab[lemma][op]
                            tgt_words.pop()
                tgt_words.append(word)
            tgt.write(' '.join(tgt_words) + '\n')

def make_vocab(src_train, tgt_train):
    words = {}
    with open(src_train) as src, open(tgt_train) as tgt:
        for src_line, tgt_line in zip(src, tgt):
            src_words = set(src_line.split())
            tgt_words = set(tgt_line.split())
            # src exclusive
            src_only = src_words - tgt_words
            # tgt exclusive
            tgt_only = tgt_words - src_words
            # combined_words = tgt_words | src_words # combined vocab
            # combined_words = tgt_only | src_only   # ignores common words
            combined_words = tgt_only                # tgt exclusive
            for word in combined_words:
                words[word] = words[word] + 1 if word in words else 1

    sorted_words = sorted(
        words.items(), key=operator.itemgetter(1), reverse=True)
    sorted_words_list = [x[0] for x in sorted_words]

    vocab = list(STOP_WORDS) + sorted_words_list
    return vocab[:opt.vocab_size], sorted_words_list

def encode(src_original, src_encoded, tgt_original, tgt_encoded, vocab):
    hist = {}
    unique_hist = {}

    if not tgt_original:
        tgt_original = src_original
        tgt_encoded = '/dev/null'

    with open(src_original) as src_full, \
        open(tgt_original) as tgt_full, \
        open(src_encoded, 'w') as src_unk, \
        open(tgt_encoded, 'w') as tgt_unk:
        for src_line, tgt_line in zip(src_full, tgt_full):
            src_words = src_line.split()
            tgt_words = tgt_line.split()
            src_unks, tgt_unks = [], []
            src_unks = [
                word for word in src_words if word not in vocab + src_unks]
            tgt_unks = [
                word for word in tgt_words if word not in vocab + src_unks + tgt_unks]
            # unique list of source and target unknowns
            unks = src_unks + tgt_unks
            for idx, unk in enumerate(unks):
                src_words = ['unk{}'.format(
                    idx + 1) if word == unk else word for word in src_words]
                tgt_words = ['unk{}'.format(
                    idx + 1) if word == unk else word for word in tgt_words]
            src_unk.write(' '.join(src_words) + '\n')
            tgt_unk.write(' '.join(tgt_words) + '\n')

def decode(src_original, src_unknown, pred_unknown, pred_decoded):
    with open(src_original) as src, open(src_unknown) as src_unk, \
        open(pred_unknown) as pred_unk, open(pred_decoded, 'w') as pred:

        for src_line, src_unk_line, pred_unk_line in zip(src, src_unk, pred_unk):
            mapping = {}
            for src_word, src_unk_word in zip(src_line.split(), src_unk_line.split()):
                if src_unk_word.lower().startswith('unk'):
                    mapping[src_unk_word.lower()] = src_word
            pred_line = []
            for word in pred_unk_line.split():
                if word.lower().startswith('unk'):
                    word = mapping[word] if word in mapping else word
                pred_line.append(word)
            pred.write(' '.join(pred_line) + '\n')

def main():

    start_time = time.time()

    if opt.lemmatize or opt.delemmatize:
        pickle_file = 'data/vocab.p'
        try:
            vocab = pickle.load(open(pickle_file, 'rb'))
            print('Loaded vocab from {}'.format(pickle_file))
        except IOError:
            vocab = {}
            vocab, _ = lemmatize('data/src-train.txt', '/dev/null', vocab)
            vocab, _ = lemmatize('data/tgt-train.txt', '/dev/null', vocab)
            pickle.dump(vocab, open(pickle_file, 'wb'))
            print('Stored vocab in {}'.format(pickle_file))

    if opt.lemmatize:
        src_delemmatized = 'data/src-dev.txt'
        tgt_lemmatized = 'data/unk-500-lemma/src-dev.lemma.txt'
        _, _ = lemmatize(src_delemmatized, tgt_lemmatized, vocab)

    if opt.delemmatize:
        src_lemmatized = 'data/unk-600-lemma/pred-test.lemma.txt.unique.split'
        tgt_delemmatized = 'data/unk-600-lemma/pred-test.txt.unique.split'
        delemmatize(src_lemmatized, tgt_delemmatized, vocab)

    if opt.encode:
        # Remember to check vocab
        opt.vocab_size = 500
        src_train = 'data/unk-500-lemma/src-train.lemma.txt'
        tgt_train = 'data/unk-500-lemma/tgt-train.lemma.txt'
        vocab, _ = make_vocab(src_train, tgt_train)
        import pudb; pudb.set_trace()

        src_original = 'data/unk-500-lemma/src-dev.lemma.txt'
        src_encoded = 'data/unk-500-lemma/src-dev.unk.lemma.500.txt'
        # Leave empty strings to only encode source files.
        tgt_original = 'data/unk-500-lemma/tgt-dev.lemma.txt'
        tgt_encoded = 'data/unk-500-lemma/tgt-dev.unk.lemma.500.txt'
        encode(src_original, src_encoded, tgt_original, tgt_encoded, vocab)

    if opt.decode:
        src_original = 'data/split-unique/src-test.txt.unique.split'
        src_unknown = 'data/split-unique/src-test.unk.txt.unique.split'
        pred_unknown = 'data/unk-500/pred-test.unk.txt.unique.split'
        pred_decoded = 'data/unk-500/pred-test.txt.unique.split'
        decode(src_original, src_unknown, pred_unknown, pred_decoded)
    print('Done. Took {:f} seconds.'.format(time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='utils.py')

    parser.add_argument('-lemmatize', action='store_true',
                        help='Lemmatize')
    parser.add_argument('-delemmatize', action='store_true',
                        help='Delemmatize')
    parser.add_argument('-encode', action='store_true',
                        help='Replace OOV by UNK tokens')
    parser.add_argument('-decode', action='store_true',
                        help='Replace UNK tokens back with source words')
    parser.add_argument('-vocab_size', type=int, default=500,
                        help="Replace all except these many tokens")
    opt = parser.parse_args()

    main()
