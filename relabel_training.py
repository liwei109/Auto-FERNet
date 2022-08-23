import os
import sys
import time
import glob
import torch
from tools import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import uuid
from torch.autograd import Variable
from NAS.model import NetworkCIFAR as Network
from NAS.genotypes import *
from dataloader.dataload_h5 import *


parser = argparse.ArgumentParser("Relable")
# parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.02272721, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=12, help='total number of layers')
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

parser.add_argument('--checkpoint', type=str, default='./85_weights_12.pt', help='which checkpoint to use')
parser.add_argument('--relabel_threshold', type=float, default=0.2, help='relabel threshold')
parser.add_argument('--fes', default=True, help='use similarity matrix')
parser.add_argument('--fes_threshold', type=float, default=0.03, help='fes threshold')


args = parser.parse_args()

args.save = './relabel_log/12_85eval-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), str(uuid.uuid4()))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
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
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  # torch.nn.DataParallel(model,args.gpu)
  model = model.cuda()
  model.load_state_dict(torch.load(args.checkpoint)) # change the site

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_queue, valid_queue, test_queue = GetFER2013_for_retrain(args)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_val_acc, best_test_acc = 0., 0.
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * (epoch+115) / (600) # args.epochs
    # model.drop_path_prob = args.drop_path_prob * (epoch + 109) / (600)  # args.epochs

    relabel = False
    if epoch > 0:
      relabel = True

    train_acc, train_obj = train(train_queue, model, criterion, optimizer, relabel)
    logging.info('train_acc %f', train_acc)

    with torch.no_grad():
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        utils.save(model, os.path.join(args.save, 'best_val_weights_relabel.pt'))
      logging.info('valid_acc %f\tbest_val_acc %f', valid_acc, best_val_acc)

    with torch.no_grad():
      test_acc, test_obj = test(test_queue, model, criterion)
      if test_acc > best_test_acc:
        best_test_acc = test_acc
        utils.save(model, os.path.join(args.save, 'best_test_weights_relabel.pt'))
      logging.info('test_acc %f\tbest_test_acc %f', test_acc, best_test_acc)

    utils.save(model, os.path.join(args.save, 'weights_relabel.pt'))

def train(train_queue, model, criterion, optimizer, relabel):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.train()


  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    soft_logits = torch.nn.functional.softmax(logits, dim=1)

    if relabel:
      for i in range(target.size(0)):
        if args.fes:
          if torch.max(soft_logits[i]) - soft_logits[i][target[i].item()] < args.relabel_threshold \
                  and torch.max(soft_logits[i]) - soft_logits[i][target[i].item()] > 0\
                  and FES[target[i].item()][soft_logits[i].topk(1, 0, True)[1].item()] > args.fes_threshold:
            target[i] = soft_logits[i].topk(1, 0, True)[1]

        else:
          if torch.max(soft_logits[i]) - soft_logits[i][target[i].item()] < args.relabel_threshold:  # changed ********* FES
            target[i] = soft_logits[i].topk(1, 0, True)[1]

    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def test(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
