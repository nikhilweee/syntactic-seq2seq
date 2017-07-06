from __future__ import division

import onmt
import onmt.Markdown
import onmt.Models
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
<<<<<<< HEAD
from torch.autograd import Variable
import math
import time
import subprocess
import os
import sys
import re
import utils
import pickle
=======
>>>>>>> ecbce3330acbe97c2319e3cb13ce38a5a399b876

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options
# Path to *.low.train.pt
base_dir = 'data/unk-600-lemma'
data = os.path.join(base_dir, 'preprocessed.unk.lemma.low.train.pt')
save_model = os.path.join(base_dir, 'models/model')

test = True
# Used for decoding and delemmatization
src_dev = 'data/unk-600-lemma/src-dev.unq.txt'
src_dev_lemma = 'data/unk-600-lemma/src-dev.lemma.unq.txt'
src_dev_lemma_unk = 'data/unk-600-lemma/src-dev.unk.lemma.unq.txt'
pred_dev_lemma_unk = 'data/unk-600-lemma/pred-dev.unk.lemma.unq.txt'
pred_dev_lemma = 'data/unk-600-lemma/pred-dev.lemma.unq.txt'
pred_dev = 'data/unk-600-lemma/pred-dev.unq.txt'

# test_src = os.path.join(base_dir, 'src-dev.unk.lemma.txt')
# test_tgt = os.path.join(base_dir, 'tgt-dev.unk.lemma.txt')
# test_pred = os.path.join(base_dir, 'pred-dev.unk.lemma.txt')

parser.add_argument('-data', required=False,
                    default=data,
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
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-feature_vec_size', type=int, default=100,
                    help='Feature vec sizes')

parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-rnn_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help="""The gate type to use in the RNNs""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true', default=True,
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-copy_attn', action="store_true",
                    help='Train copy attention layer.')
parser.add_argument('-coverage_attn', action="store_true",
                    help='Train a coverage attention layer.')
parser.add_argument('-lambda_coverage', type=float, default=1,
                    help='Lambda value for coverage.')

parser.add_argument('-encoder_layer', type=str, default='rnn',
                    help="""Type of encoder layer to use.
                    Options: [rnn|mean|transformer]""")
parser.add_argument('-decoder_layer', type=str, default='rnn',
                    help='Type of decoder layer to use. [rnn|transformer]')
parser.add_argument('-context_gate', type=str, default=None,
                    choices=['source', 'target', 'both'],
                    help="""Type of context gate to use [source|target|both].
                    Do not select for no context gate.""")
parser.add_argument('-attention_type', type=str, default='dotprod',
                    choices=['dotprod', 'mlp'],
                    help="""The attention type to use:
                    dotprot (Luong) or MLP (Bahdanau)""")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init).
                    Use 0 to not use initialization""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-position_encoding', action='store_true',
                    help='Use a sinusoid to mark relative words positions.')
parser.add_argument('-share_decoder_embeddings', action='store_true',
                    help='Share the word and softmax embeddings for decoder.')

parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-truncated_decoder', type=int, default=0,
                    help="""Truncated bptt.""")

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
parser.add_argument('-start_checkpoint_at', type=int, default=0,
                    help="""Start checkpointing every epoch after and including this
                    epoch""")
parser.add_argument('-decay_method', type=str, default="",
                    help="""Use a custom learning rate decay [|noam] """)
parser.add_argument('-warmup_steps', type=int, default=4000,
                    help="""Number of warmup steps for custom decay.""")


# pretrained word vectors

parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")
parser.add_argument('-log_server', type=str, default="",
                    help="Send logs to this crayon server.")
parser.add_argument('-experiment_name', type=str, default="",
                    help="Name of the experiment for logging.")

parser.add_argument('-seed', type=int, default=-1,
                    help="""Random seed used for the experiments
                    reproducibility.""")

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
parser.add_argument('-early_stop_epochs',  type=int, default=5,
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

print(opt)
# Save Config
with open(os.path.join(base_dir, 'opt.txt'), 'w') as f:
    f.write(str(opt))

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


# Set up the Crayon logging server.
if opt.log_server != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.log_server)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.experiment_name in experiments:
        cc.remove_experiment(opt.experiment_name)
    experiment = cc.create_experiment(opt.experiment_name)


def eval(model, criterion, data):
    stats = onmt.Loss.Statistics()
    model.eval()
    loss = onmt.Loss.MemoryEfficientLoss(opt, model.generator, criterion,
                                         eval=True, copy_loss=opt.copy_attn)
    for i in range(len(data)):
        batch = data[i]
        outputs, attn, dec_hidden = model(batch.src, batch.tgt, batch.lengths)
        batch_stats, _, _ = loss.loss(batch, outputs, attn)
        stats.update(batch_stats)
    model.train()
    return stats


def trainModel(model, trainData, validData, dataset, optim):
    model.train()

    # Define criterion of each GPU.
    if not opt.copy_attn:
        criterion = onmt.Loss.NMTCriterion(dataset['dicts']['tgt'].size(), opt)
    else:
        criterion = onmt.modules.CopyCriterion

    bleu4_scores = []

    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        mem_loss = onmt.Loss.MemoryEfficientLoss(opt, model.generator,
                                                 criterion,
                                                 copy_loss=opt.copy_attn)

        # Shuffle mini batch order.
        batchOrder = torch.randperm(len(trainData))

        total_stats = onmt.Loss.Statistics()
        report_stats = onmt.Loss.Statistics()

        for i in range(len(trainData)):
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
<<<<<<< HEAD
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
                print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
                       "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                      (epoch, i+1, len(trainData),
                       report_num_correct / report_tgt_words * 100,
                       math.exp(report_loss / report_tgt_words),
                       report_src_words/(time.time()-start),
                       report_tgt_words/(time.time()-start),
                       time.time()-start_time))
=======
            batch = trainData[batchIdx]
            target_size = batch.tgt.size(0)

            dec_state = None
            trunc_size = opt.truncated_decoder if opt.truncated_decoder \
                else target_size

            for j in range(0, target_size-1, trunc_size):
                trunc_batch = batch.truncate(j, j + trunc_size)

                # Main training loop
                model.zero_grad()
                outputs, attn, dec_state = model(trunc_batch.src,
                                                 trunc_batch.tgt,
                                                 trunc_batch.lengths,
                                                 dec_state)
                batch_stats, inputs, grads \
                    = mem_loss.loss(trunc_batch, outputs, attn)
>>>>>>> ecbce3330acbe97c2319e3cb13ce38a5a399b876

                torch.autograd.backward(inputs, grads)

                # Update the parameters.
                optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                if dec_state is not None:
                    dec_state.detach()

            report_stats.n_src_words += batch.lengths.data.sum()

            if i % opt.log_interval == -1 % opt.log_interval:
                report_stats.output(epoch, i+1, len(trainData),
                                    total_stats.start_time)
                if opt.log_server:
                    report_stats.log("progress", experiment, optim)
                report_stats = onmt.Loss.Statistics()

        return total_stats

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')
        print('Learning rate: %g' % optim.lr)

        #  (1) train for one epoch on the training set
        train_stats = trainEpoch(epoch)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        #  (2) evaluate on the validation set
        valid_stats = eval(model, criterion, validData)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # Log to remote server.
        if opt.log_server:
            train_stats.log("train", optim, experiment)
            valid_stats.log("valid", optim, experiment)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_stats.ppl(), epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = (model.generator.module.state_dict()
                                if len(opt.gpus) > 1
                                else model.generator.state_dict())
        #  (4) drop a checkpoint
<<<<<<< HEAD
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
            utils.decode(src_dev_lemma, src_dev_lemma_unk, pred_dev_lemma_unk, pred_dev_lemma)
            # delemmatize
            vocab = pickle.load(open('data/split/dev/vocab.p', 'rb'))
            utils.delemmatize(pred_dev_lemma, pred_dev, vocab)

            cmd = 'perl multi-bleu.perl {} < {}'.format('data/split/dev/tgt-dev.unq.txt.split', pred_dev)
            bleu = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
            print(bleu)
            bleu4 = re.findall('\d\.\d{2}', bleu)[0]
            bleu4_scores.append(bleu4)

            # Save best model
            if bleu4 > max(bleu4_scores):
                model_file_name = '%s_e%d_acc_%.2f_ppl_%.2f.pt' % \
                                  (opt.save_model, epoch, 100*valid_acc, valid_ppl)
                print('Saving model to {}'.format(model_file_name))
                torch.save(checkpoint, model_file_name)

            # Early stopping
            if len(bleu4_scores) - bleu4_scores.index(max(bleu4_scores)) > opt.early_stop_epochs:
                print("BLEU score didn't improve since {} epochs".format(opt.early_stop_epochs))
                print("Best BLEU is {} on epoch {}".format(max(bleu4_scores), bleu4_scores.index(max(bleu4_scores)) + 1))
                print(" Stopping ...")
                sys.exit()


def test(model_file_name):
    opt.cuda = True
    opt.model = model_file_name
    translator = onmt.Translator(opt)
    outF = open(opt.test_pred, 'w', encoding='utf-8')
    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0
    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.test_tgt, encoding='utf-8') if opt.test_tgt else None

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()

    def addone(f):
        for line in f:
            yield line
        yield None

    for line in addone(open(opt.test_src, encoding='utf-8')):
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
        print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, scoreTotal / wordsTotal,
            name, math.exp(-scoreTotal/wordsTotal)))

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum, open(opt.dump_beam, 'w', encoding='utf-8'))
=======
        if epoch >= opt.start_checkpoint_at:
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'optim': optim
            }
            torch.save(checkpoint,
                       '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                       % (opt.save_model, valid_stats.accuracy(),
                          valid_stats.ppl(), epoch))
>>>>>>> ecbce3330acbe97c2319e3cb13ce38a5a399b876


def main():
    print("Loading data from '%s'" % opt.data)
    dataset = torch.load(opt.data)
    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']
    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"),
                             srcFeatures=dataset['train'].get('src_features'),
                             tgtFeatures=dataset['train'].get('tgt_features'),
                             alignment=dataset['train'].get('alignments'))
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"),
                             srcFeatures=dataset['valid'].get('src_features'),
                             tgtFeatures=dataset['valid'].get('tgt_features'),
                             alignment=dataset['valid'].get('alignments'))

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    if 'src_features' in dicts:
        for j in range(len(dicts['src_features'])):
            print(' * src feature %d size = %d' %
                  (j, dicts['src_features'][j].size()))

    dicts = dataset['dicts']
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.encoder_type == "text":
        encoder = onmt.Models.Encoder(opt, dicts['src'],
                                      dicts.get('src_features', None))
    elif opt.encoder_type == "img":
        encoder = onmt.modules.ImageEncoder(opt)
        assert("type" not in dataset or dataset["type"] == "img")
    else:
        print("Unsupported encoder type %s" % (opt.encoder_type))

    decoder = onmt.Models.Decoder(opt, dicts['tgt'])

    if opt.copy_attn:
        generator = onmt.modules.CopyGenerator(opt, dicts['src'], dicts['tgt'])
    else:
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()),
            nn.LogSoftmax())
        if opt.share_decoder_embeddings:
            generator[0].weight = decoder.word_lut.weight

    model = onmt.Models.NMTModel(encoder, decoder, len(opt.gpus) > 1)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                            if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s'
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
        print('Multi gpu training ', opt.gpus)
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        if opt.param_init != 0.0:
            print('Intializing params')
            for p in model.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_enc)
        decoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_dec)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
