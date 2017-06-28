from __future__ import division

import argparse
import math
import os
import spacy
import onmt
import onmt.Markdown
import operator
import torch
import pprint

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

path = 'data'
base_dir = 'data/unk-1000/'
model = os.path.join(base_dir, 'models/')
src = os.path.join(base_dir, 'src-test.unk.excl.txt.unique.split')
output = os.path.join(base_dir, 'pred-test.unk.excl.txt.unique.split')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src', required=False,
                    default=None,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    default=None,
                    help='True target sequence (optional)')
parser.add_argument('-output', default=output,
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=1,
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
parser.add_argument('-dump_beam', action="store_true", default=True,
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=5,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=0,
                    help="Device to run on")


class Curiosity(object):

    def __init__(self):
        self.vocab, _ = self.make_exclusive_vocab()

    def make_exclusive_vocab(self):
        stop_words = spacy.en.language_data.STOP_WORDS
        words = {}
        with open(os.path.join(path, 'src-train.txt')) as src, \
                open(os.path.join(path, 'tgt-train.txt')) as tgt:
            for src_line, tgt_line in zip(src, tgt):
                src_words = set(src_line.split())
                tgt_words = set(tgt_line.split())
                src_only = src_words - tgt_words
                tgt_only = tgt_words - src_words
                combined_words = tgt_only
                for word in combined_words:
                    words[word] = words[word] + 1 if word in words else 1

        sorted_words = sorted(
            words.items(), key=operator.itemgetter(1), reverse=True)
        sorted_words_list = [x[0] for x in sorted_words]
        vocab = list(stop_words) + sorted_words_list
        return vocab[:500], sorted_words_list

    def encode(self, src):
        src_words = src.split()
        src_unks = []
        src_unks = [
            word for word in src_words if word not in self.vocab + src_unks]
        for idx, unk in enumerate(src_unks):
            src_words = ['unk{}'.format(
                idx + 1) if word == unk else word for word in src_words]
        return ' '.join(src_words)

    def decode(self, src, src_unk, pred_unk):
        mapping = {}
        for src_word, src_unk_word in zip(src.split(), src_unk.split()):
            if src_unk_word.lower().startswith('unk'):
                mapping[src_unk_word.lower()] = src_word
        pred_line = []
        # import pudb; pudb.set_trace()
        for word in pred_unk.split():
            if word.lower().startswith('unk'):
                word = mapping[word] if word in mapping else word
            pred_line.append(word)
        return ' '.join(pred_line)

    def ask(self):
        opt = parser.parse_args()
        opt.cuda = opt.gpu > -1
        if opt.cuda:
            torch.cuda.set_device(opt.gpu)

        translator = onmt.Translator(opt)
        predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0
        srcBatch, tgtBatch = [], []

        if opt.dump_beam != "":
            translator.initBeamAccum()

        while True:
            src = input('\n>')
            src_unk = self.encode(src)
            srcTokens = src_unk.split()
            srcBatch += [srcTokens]

            # at the end of file, check last batch
            if not src:
                break
            # import pudb; pudb.set_trace()
            predBatch, predScore, goldScore = translator.translate(srcBatch,
                                                                   tgtBatch)

            if opt.dump_beam:
                # pprint.pprint(translator.beam_accum)
                translator.initBeamAccum()

            pred_unks = [" ".join(predBatch[0][n]) for n in range(opt.n_best)]
            preds = [self.decode(src, src_unk, pred_unks[n]) for n in range(opt.n_best)]
            print('\nUNK: {}'.format(src_unk))
            for n in range(opt.n_best):
                print('BEST {}: \n {} \n {}'.format(n+1, pred_unks[n], preds[n]))
            srcBatch, tgtBatch = [], []


if __name__ == "__main__":
    curiosity = Curiosity()
    curiosity.ask()
