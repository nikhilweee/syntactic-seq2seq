from __future__ import print_function, unicode_literals
import time
import nltk
import operator
import pickle
import argparse
import codecs
import logging

# use spaCy stopwords
# STOP_WORDS = spacy.en.language_data.STOP_WORDS
# use NLTK stopwords
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words('english')

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
        logging.info('Tagging {} ...'.format(src_delemmatized))
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
        logging.info('Tagging Complete.')
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
    logging.info('Making vocab ...')
    words = {}
    with codecs.open(src_train, encoding='utf-8') as src, codecs.open(tgt_train, encoding='utf-8') as tgt:
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

    # import spacy
    # nlp = spacy.load('en')
    # doc = nlp(' '.join(sorted_words_list))
    # ents = [ent.text for ent in doc.ents]
    # vocab = list(STOP_WORDS) + [word.text for word in doc if word.text not in ents]

    vocab = list(STOP_WORDS) + sorted_words_list
    logging.info('Making vocab complete.')
    return vocab[:opt.vocab_size], sorted_words_list

def encode(src_original, src_encoded, tgt_original, tgt_encoded, vocab):
    hist = {}
    unique_hist = {}

    if not tgt_original:
        tgt_original = src_original
        tgt_encoded = '/dev/null'

    with codecs.open(src_original, encoding='utf-8') as src_full, \
        codecs.open(tgt_original, encoding='utf-8') as tgt_full, \
        codecs.open(src_encoded, 'w', encoding='utf-8') as src_unk, \
        codecs.open(tgt_encoded, 'w', encoding='utf-8') as tgt_unk:
        count = 0
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
                # Tag different forms separately
                src_words = ['unk{}'.format(
                    idx + 1) if word == unk else word for word in src_words]
                tgt_words = ['unk{}'.format(
                    idx + 1) if word == unk else word for word in tgt_words]
            src_unk.write(' '.join(src_words) + '\n')
            tgt_unk.write(' '.join(tgt_words) + '\n')
            count += 1
            if count % 1000 == 0:
                logging.info('Wrote {} lines.'.format(count))

def encode_tagged(src_original, src_encoded, tgt_original, tgt_encoded, vocab):
    logging.info('Encoding ...')
    hist = {}
    unique_hist = {}

    if not tgt_original:
        tgt_original = src_original
        tgt_encoded = '/dev/null'

    with codecs.open(src_original, encoding='utf-8') as src_full, \
        codecs.open(tgt_original, encoding='utf-8') as tgt_full, \
        codecs.open(src_encoded, 'w', encoding='utf-8') as src_unk, \
        codecs.open(tgt_encoded, 'w', encoding='utf-8') as tgt_unk:
        for idx, (src_line, tgt_line) in enumerate(zip(src_full, tgt_full)):
            if idx % 1000 == 0:
                logging.info('Processed {} lines'.format(idx))
            src_words = src_line.split()
            tgt_words = tgt_line.split()
            src_unks, tgt_unks = [], []
            src_unks = [
                word for word in src_words if word not in vocab + src_unks]
            tgt_unks = [
                word for word in tgt_words if word not in vocab + src_unks + tgt_unks]
            # unique list of source and target unknowns
            unks = src_unks + tgt_unks
            tags = nltk.pos_tag(unks)
            for idx, (unk, tag) in enumerate(tags):
                # Tag different forms separately.
                # Assuming OPERATORS are already a part of vocab.
                src_words = ['#{}{}'.format(tag.lower(),
                    idx + 1) if word == unk else word for word in src_words]
                tgt_words = ['#{}{}'.format(tag.lower(),
                    idx + 1) if word == unk else word for word in tgt_words]
            src_unk.write(' '.join(src_words) + '\n')
            tgt_unk.write(' '.join(tgt_words) + '\n')
    logging.info('Encoding complete.')

def decode(src_original, src_unknown, pred_unknown, pred_decoded):
    with codecs.open(src_original, encoding='utf-8') as src, codecs.open(src_unknown, encoding='utf-8') as src_unk, \
        codecs.open(pred_unknown, encoding='utf-8') as pred_unk, codecs.open(pred_decoded, 'w', encoding='utf-8') as pred:

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

def decode_tagged(src_original, src_unknown, pred_unknown, pred_decoded):
    with codecs.open(src_original, encoding='utf-8') as src, codecs.open(src_unknown, encoding='utf-8') as src_unk, \
        codecs.open(pred_unknown, encoding='utf-8') as pred_unk, codecs.open(pred_decoded, 'w', encoding='utf-8') as pred:

        for src_line, src_unk_line, pred_unk_line in zip(src, src_unk, pred_unk):
            mapping = {}
            for src_word, src_unk_word in zip(src_line.split(), src_unk_line.split()):
                if src_unk_word.startswith('#'):
                    mapping[src_unk_word] = src_word
            pred_line = []
            for word in pred_unk_line.split():
                if word.startswith('#'):
                    word = mapping[word] if word in mapping else word
                pred_line.append(word)
            pred.write(' '.join(pred_line) + '\n')

def main():

    start_time = time.time()

    if opt.lemmatize or opt.delemmatize:
        pickle_file = 'data/lemmatized/vocab.p'
        try:
            vocab = pickle.load(open(pickle_file, 'rb'))
            logging.info('Loaded vocab from {}'.format(pickle_file))
        except IOError:
            vocab = {}
            vocab, _ = lemmatize('data/raw/src-train.txt', '/dev/null', vocab)
            vocab, _ = lemmatize('data/raw/tgt-train.txt', '/dev/null', vocab)
            pickle.dump(vocab, open(pickle_file, 'wb'))
            logging.info('Stored vocab in {}'.format(pickle_file))

    if opt.lemmatize:
        src_delemmatized = 'data/raw/src-dev.txt'
        tgt_lemmatized = 'data/unk-500-lemma/src-dev.lemma.txt'
        _, _ = lemmatize(src_delemmatized, tgt_lemmatized, vocab)

    if opt.delemmatize:
        src_lemmatized = 'data/unk-600-lemma/pred-test.lemma.txt.unique.split'
        tgt_delemmatized = 'data/unk-600-lemma/pred-test.txt.unique.split'
        delemmatize(src_lemmatized, tgt_delemmatized, vocab)

    if opt.encode:
        # Remember to check vocab
        opt.vocab_size = 5000
        src_train = 'data/lemmatized/src-train.lemma.txt'
        tgt_train = 'data/lemmatized/tgt-train.lemma.txt'
        vocab, _ = make_vocab(src_train, tgt_train)

        src_original = 'data/lemmatized/src-dev.lemma.txt'
        src_encoded = 'data/unk-5000-wordvec/src-dev.unk.lemma.5000.txt'
        # Leave empty strings to only encode source files.
        tgt_original = 'data/lemmatized/tgt-dev.lemma.txt'
        tgt_encoded = 'data/unk-5000-wordvec/tgt-dev.unk.lemma.5000.txt'
        # encode(src_original, src_encoded, tgt_original, tgt_encoded, vocab)
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

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s', '%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(console)

    main()
