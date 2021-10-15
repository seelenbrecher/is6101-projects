import random, os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import BertModel, BertTokenizer

from models import FakeNewsClassifier
from torch.nn import NLLLoss

from dataLoader import DataLoader

import logging

logger = logging.getLogger(__name__)


def reproducible():
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    

def correct_prediction(output, labels, threshold):
    preds = output > threshold
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def train_model(model, args, trainset_reader, validset_reader):
  def _eval_model(model, validset_reader, threshold):
    model.eval()
    correct_pred = 0.0
    for index, data in enumerate(validset_reader):
        inputs, labels = data
        probs = model(inputs)
        correct_pred += correct_predictions(probs, labels, threshold)
    # TODO: How to define the total samples in the dataset
    accuracy = correct_pred / validset_reader.total_num
    return accuracy
  
  # save the best model inside outdir
  save_path = args.outdir + '/model'
  
  best_accuracy = 0.0
  running_loss = 0.0
  
  # setup the optimizers. some may not be stated in the paper, but might be useful
  t_total = int(trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.epoch)
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = BertAdam(optimizer_grouped_parameters,
                       lr=args.learning_rate) # follows the paper, use ADAM
  scheduler = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=1000, num_training_steps=t_total
  )
  
  global_step = 0
  for epoch in range(int(args.epoch)):
      model.train()
      optimizer.zero_grad()
      for index, data in enumerate(trainset_reader):
          inputs, labels = data
          probs = model(inputs)
          loss = F.binary_cross_entropy(probs.squeeze(-1), labels)
          running_loss += loss.item()
          if args.gradient_accumulation_steps > 1:
              loss = loss / args.gradient_accumulation_steps
          loss.backward()
          global_step += 1
          if global_step % args.gradient_accumulation_steps == 0:
              optimizer.step()
              optimizer.zero_grad()
              logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
          if global_step % args.eval_step == 0:
              logger.info('Start eval!')
              with torch.no_grad():
                  dev_accuracy = _eval_model(model, validset_reader, threshold)
                  logger.info('Dev total acc: {0}'.format(dev_accuracy))
                  if dev_accuracy > best_accuracy:
                      best_accuracy = dev_accuracy

                      torch.save({'epoch': epoch,
                                  'model': model.state_dict(),
                                  'best_accuracy': best_accuracy}, save_path + ".best.pt")
                      logger.info('prec = {}, rec={}, f1={}'.format(prec, rec, f1))
                      logger.info("Saved best epoch {0}, best accuracy {1}".format(epoch, best_accuracy))


def main(args):
    logger.info('initializing bert tokenizer')
    tokenizer = BertTokenizer.from_pretrained(args.bert_type)
    
    logger.info("loading training set")
    # TODO: call dataloader here
    trainset_reader = DataLoader(args.train_path, tokenizer, args)
    validset_reader = DataLoader(args.valid_path, tokenizer, args)

    logger.info('initializing bert model')
    bert_model = BertModel.from_pretrained(args.bert_type)
    
    model = FakeNewsClassifier(bert_model, args)
    model = nn.DataParallel(model)
    model = model.cuda()
    train_model(model, args, trainset_reader, validset_reader)
  
if __name__ == "__main__":
    reproducible()
    parser = argparse.ArgumentParser()
    
    # BERT arguments
    parser.add_argument('--bert_type', default='bert-base-multilingual-cased')
    parser.add_argument('--tokenizer_max_length', type=int, default=128,  help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden dim BERT")
    
    # train and val arguments
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    
    # optimizer arguments
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate")
    parser.add_argument("--epoch", default=3, type=int)
    
    # arguments that is not stated in the paper, but might be useful
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_step", default=500, type=int,
                        help="number of steps required to do the evaluation")
    
    # prediction arguments
    parser.add_argument("--threshold", default=0.5, type=float, help="Threshold to decide the label given the logits")

    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    
    main(args)
