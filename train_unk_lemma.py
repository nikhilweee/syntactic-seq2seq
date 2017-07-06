from __future__ import division

import onmt
import onmt.Markdown
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import subprocess
import os
import sys
import re
import utils
import pickle
import logging
from qgevalcap.eval import eval as evaluate

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options
# Path to *.low.train.pt
base_dir = 'data/unk-5000-wordvec'
data = os.path.join(base_dir, 'preprocessed.unk.lemma.5000.low.train.pt')
save_model = os.path.join(base_dir, 'models/model_emb_128')

test = True
# Used for decoding and delemmatization
src_dev = os.path.join('data/raw', 'src-dev.txt')
src_dev_lemma = os.path.join('data/lemmatized', 'src-dev.lemma.txt')
src_dev_lemma_unk = os.path.join(base_dir, 'src-dev.unk.lemma.5000.txt')
pred_dev_lemma_unk = os.path.join(base_dir, 'pred-dev.unk.lemma.5000.128.txt')
pred_dev_lemma = os.path.join(base_dir, 'pred-dev.lemma.128.txt')
pred_dev = os.path.join(base_dir, 'pred-dev.128.txt')
tgt_dev = os.path.join('data/raw', 'tgt-dev.txt')
glove_6B_100d = 'data/glove/glove.pt'

# test_src = os.path.join(base_dir, 'src-dev.unk.lemma.txt')
# test_tgt = os.path.join(base_dir, 'tgt-dev.unk.lemma.txt')
# test_pred = os.path.join(base_dir, 'pred-dev.unk.lemma.txt')
formatter = logging.Formatter(
    '%(asctime)s %(levelname)-8s %(message)s', '%Y-%m-%d %H:%M:%S')
logfile = logging.FileHandler(
    filename=os.path.join(base_dir, 'train.log'), mode='a')
logfile.setFormatter(formatter)
console = logging.StreamHandler()
console.setFormatter(formatter)

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.addHandler(logfile)

logging.info('This works!')

parser.add_argument('-data', required=False, default=data,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default=save_model,
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=128,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true', default=True,
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

# Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=64,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.8,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

# pretrained word vectors

parser.add_argument('-pre_word_vecs_enc', default=glove_6B_100d,
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec', default=glove_6B_100d,
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

# Evaluate
parser.add_argument('-test', default=test, action="store_true",
                    help="""Evaluate on test set after every epoch.""")
parser.add_argument('-test_src', default=src_dev_lemma_unk,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-test_tgt', default=None,
                    help='True target sequence (optional)')
parser.add_argument('-test_pred', default=pred_dev_lemma_unk,
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-early_stop_epochs',  type=int, default=15,
                    help='Stop after these many epochs if the BLEU score doesn\'t improve.')
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
# parser.add_argument('-batch_size', type=int, default=30,
#                     help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true", default=False,
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

opt = parser.parse_args()

logging.info(opt)
# Save Config
with open(os.path.join(base_dir, 'opt.txt'), 'w') as f:
    f.write(str(opt))

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = generator(out_t)
        loss_t = crit(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data) \
                                   .masked_select(
                                       targ_t.ne(onmt.Constants.PAD).data) \
                                   .sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        # exclude original indices
        batch = data[i][:-1]
        outputs = model(batch)
        # exclude <s> from targets
        targets = batch[1][1:]
        loss, _, num_correct = memoryEfficientLoss(
            outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words, total_num_correct / total_words


def trainModel(model, trainData, validData, dataset, optim):
    logging.info(model)
    model.train()

    # Define criterion of each GPU.
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()

    bleu4_scores = []

    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words, report_num_correct = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            # Exclude original indices.
            batch = trainData[batchIdx][:-1]

            model.zero_grad()
            outputs = model(batch)
            # Exclude <s> from targets.
            targets = batch[1][1:]
            loss, gradOutput, num_correct = memoryEfficientLoss(
                outputs, targets, model.generator, criterion)

            outputs.backward(gradOutput)

            # Update the parameters.
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][1].data.sum()
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                logging.info(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
                       "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                      (epoch, i + 1, len(trainData),
                       report_num_correct / report_tgt_words * 100,
                       math.exp(report_loss / report_tgt_words),
                       report_src_words / (time.time() - start),
                       report_tgt_words / (time.time() - start),
                       time.time() - start_time))

                report_loss, report_tgt_words = 0, 0
                report_src_words, report_num_correct = 0, 0
                start = time.time()

        return total_loss / total_words, total_num_correct / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        logging.info('')
        logging.info('Learning rate: %g' % optim.lr)

        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        logging.info('Train perplexity: %g' % train_ppl)
        logging.info('Train accuracy: %g' % (train_acc * 100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        logging.info('Validation perplexity: %g' % valid_ppl)
        logging.info('Validation accuracy: %g' % (valid_acc * 100))

        #  (3) update the learning rate
        optim.updateLearningRate(valid_ppl, epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = (model.generator.module.state_dict()
                                if len(opt.gpus) > 1
                                else model.generator.state_dict())
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }

        model_file_name = '{}_temp.pt'.format(opt.save_model)
        torch.save(checkpoint, model_file_name)

        if opt.test:
            test(model_file_name)
            # decode
            utils.decode(src_dev_lemma, src_dev_lemma_unk,
                         pred_dev_lemma_unk, pred_dev_lemma)
            # delemmatize
            vocab = pickle.load(open('data/lemmatized/vocab.p', 'rb'))
            utils.delemmatize(pred_dev_lemma, pred_dev, vocab)

            # cmd = 'perl multi-bleu.perl {} < {}'.format('data/split/dev/tgt-dev.unq.txt.split', pred_dev)
            # bleu = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
            # bleu4 = re.findall('\d{1,2}\.\d{2}', bleu)
            # bleu4 = float(bleu4[0])
            scores = evaluate(pred_dev, src_dev, tgt_dev)
            logging.info(scores)
            bleu4 = scores[3]
            bleu4_scores.append(bleu4)

            # Save best model
            if bleu4 == max(bleu4_scores):
                model_file_name = '%s_e%d_bleu4_%.5f_ppl_%.2f.pt' % \
                                  (opt.save_model, epoch, bleu4, valid_ppl)
                logging.info('Saving model to {}'.format(model_file_name))
                torch.save(checkpoint, model_file_name)

            # Early stopping
            if len(bleu4_scores) - bleu4_scores.index(max(bleu4_scores)) > opt.early_stop_epochs:
                logging.info("BLEU score didn't improve since {} epochs".format(
                    opt.early_stop_epochs))
                logging.info("Best BLEU is {} on epoch {}".format(
                    max(bleu4_scores), bleu4_scores.index(max(bleu4_scores)) + opt.start_epoch))
                logging.info("Stopping ...")
                sys.exit()


def test(model_file_name):
    opt.cuda = True
    opt.model = model_file_name
    translator = onmt.Translator(opt)
    outF = open(opt.test_pred, 'w')
    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0
    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.test_tgt) if opt.test_tgt else None

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()

    def addone(f):
        for line in f:
            yield line
        yield None

    for line in addone(open(opt.test_src)):
        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split()
                tgtBatch += [tgtTokens]

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
        if tgtF is not None:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write(" ".join(predBatch[b][0]) + '\n')
            outF.flush()

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    print("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (predScore[b][n],
                                             " ".join(predBatch[b][n])))

                print('')

        srcBatch, tgtBatch = [], []

    def reportScore(name, scoreTotal, wordsTotal):
        logging.info("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, scoreTotal / wordsTotal,
            name, math.exp(-scoreTotal / wordsTotal)))

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum, open(opt.dump_beam, 'w'))


def main():
    logging.info("Loading data from '%s'" % opt.data)
    dataset = torch.load(opt.data)

    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        logging.info('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']
    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True)

    dicts = dataset['dicts']
    logging.info(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    logging.info(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    logging.info(' * maximum batch size. %d' % opt.batch_size)

    logging.info('Building model...')

    encoder = onmt.Models.Encoder(opt, dicts['src'])
    decoder = onmt.Models.Decoder(opt, dicts['tgt'])

    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, dicts['tgt'].size()),
        nn.LogSoftmax())

    model = onmt.Models.NMTModel(encoder, decoder)

    if opt.train_from:
        logging.info('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                            if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        logging.info('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        logging.info('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        logging.info(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    logging.info('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
