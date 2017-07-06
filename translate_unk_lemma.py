from __future__ import division, print_function

import utils
import argparse
import math
import os
import codecs
import onmt
import onmt.Markdown
import torch
import pickle
import logging

parser = argparse.ArgumentParser(description='translate_unk_lemma.py')
onmt.Markdown.add_md_help_argument(parser)

base_dir = 'data/unk-5000-wordvec'

# Used for decoding and delemmatization
src_test = os.path.join('data/raw', 'src-test.txt')
src_test_lemma = os.path.join('data/lemmatized', 'src-test.lemma.txt')
src_test_lemma_unk = os.path.join(base_dir, 'src-test.unk.lemma.5000.txt')
pred_test_lemma_unk = os.path.join(base_dir, 'pred-test.unk.lemma.5000.128.txt')
pred_test_lemma = os.path.join(base_dir, 'pred-test.lemma.128.txt')
pred_test = os.path.join(base_dir, 'pred-test.128.txt')

formatter = logging.Formatter(
    '%(asctime)s %(levelname)-8s %(message)s', '%Y-%m-%d %H:%M:%S')
logfile = logging.FileHandler(
    filename=os.path.join(base_dir, 'translate.log'), mode='a')
logfile.setFormatter(formatter)
console = logging.StreamHandler()
console.setFormatter(formatter)

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.addHandler(logfile)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src', required=False,
                    default=src_test_lemma_unk,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-output', default=pred_test_lemma_unk,
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=3,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true", default=False,
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-gpu', type=int, default=0,
                    help="Device to run on")


def addone(f):
    for line in f:
        yield line
    yield None


def translate():
    logging.info('Translating ...')
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt)
    outF = codecs.open(opt.output, 'w', encoding='utf-8')
    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0
    srcBatch, tgtBatch = [], []
    count = 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()

    for line in addone(codecs.open(opt.src, encoding='utf-8')):
        if line is not None:
            count += 1
            srcTokens = line.split()
            srcBatch += [srcTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        predBatch, predScore, goldScore = translator.translate(srcBatch,
                                                               tgtBatch)
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write(" ".join(predBatch[b][0]) + '\n')
            outF.flush()

        if count % 1024 == 0:
            logging.info('Translated {} sentences'.format(count))
        srcBatch, tgtBatch = [], []

def main():
    translate()
    # decode
    logging.info('Decoding ...')
    utils.decode(src_test_lemma, src_test_lemma_unk, pred_test_lemma_unk, pred_test_lemma)
    # delemmatize
    vocab = pickle.load(open('data/lemmatized/vocab.p', 'rb'))
    logging.info('Delemmatizing ...')
    utils.delemmatize(pred_test_lemma, pred_test, vocab)
    logging.info('Done. Saved predictions to {}'.format(pred_test))


if __name__ == "__main__":
    main()
