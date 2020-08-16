import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import uuid
from torch.autograd import Variable
from model import NetworkCIFAR as Network
from  genotypes import *
from dataload_h5 import *


parser = argparse.ArgumentParser("FER2013")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=[12,12,12,12,20,20], help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='SGAS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')


args = parser.parse_args()

args.save = './ensemble_log'

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'ensemble_log{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 7

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)

  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)

  model_names = ["12best_test_90weights_relabel.pt","12best_test_85weights_relabel1.pt","12best_test_85weights_relabel_0.2_0.03.pt",
                 "12best_test_85weights_relabel2.pt","best_test_85weights_relabel.pt","best_test_90weights_relabel.pt"]
  logging.info('model_names {}'.format(model_names))
  models = []
  for i in range(len(model_names)):
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers[i], args.auxiliary, genotype)
    model = model.cuda()
    model.load_state_dict(torch.load('./ensemble_models/{}'.format(model_names[i])))
    model.drop_path_prob = args.drop_path_prob
    models.append(model)
  # torch.nn.DataParallel(model,args.gpu)
  # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  _, valid_queue, test_queue = GetFER2013_for_retrain(args)

  with torch.no_grad():
    valid_acc, valid_obj = infer(valid_queue, models, criterion)
    logging.info('valid_acc %f', valid_acc)

  with torch.no_grad():
    test_acc, test_obj = test(test_queue, models, criterion)
    logging.info('test_acc %f', test_acc)


def infer(valid_queue, models, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  for model in models:
    model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)
    init = 0
    for model in models:
      logits, _ = model(input)
      if init == 0:
        ensemble_logits = logits/len(models)
      else:
        ensemble_logits += logits/len(models)
        init = 1
    loss = criterion(ensemble_logits, target)

    prec1, prec5 = utils.accuracy(ensemble_logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def test(test_queue, models, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  for model in models:
    model.eval()

  for step, (input, target) in enumerate(test_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)
    init = 0
    for model in models:
      logits, _ = model(input)
      if init == 0:
        ensemble_logits = logits / len(models)
        init = 1
      else:
        ensemble_logits += logits / len(models)
    loss = criterion(ensemble_logits, target)

    prec1, prec5 = utils.accuracy(ensemble_logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()