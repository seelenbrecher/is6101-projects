import random, os
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import BertModel, BertTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaModel

from models import FakeNewsClassifier
from torch.nn import NLLLoss

from dataLoader import DataLoaderTest

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


def eval_model(model, validset_reader, outdir, name):
    outpath = outdir + name
    print('eval model')

    with open(outpath, "w") as f:
        for index, data in enumerate(validset_reader):
            inputs, labels, ids = data
            probs = model(inputs)
            preds = (probs > args.threshold).squeeze(-1).tolist()
            assert len(preds) == len(ids)
            for step in range(len(preds)):
                instance = {"id": ids[step], "predicted_label": preds[step]}
                f.write(json.dumps(instance) + "\n")


def main(args):
    logger.info('Start testing!')
    
    logger.info('initializing bert tokenizer')
    if args.roberta:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_type)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_type)
    
    # TODO: Load test dataloader
    logger.info("loading validation set")
    validset_reader = DataLoaderTest(args.test_path, tokenizer, args)
    
    logger.info('initializing estimator model')
    if args.roberta:
        bert_model = XLMRobertaModel.from_pretrained(args.bert_type)
    else:
        bert_model = BertModel.from_pretrained(args.bert_type)
    
    model = FakeNewsClassifier(bert_model, args)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = model.cuda()
    model.eval()
    
    eval_model(model, validset_reader, args.outdir, args.name)
    model.eval()
  

if __name__ == "__main__":
    reproducible()
    parser = argparse.ArgumentParser()
    
    # test path arguments
    parser.add_argument('--test_path', help='test path')
    parser.add_argument("--batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', help='where to store the output')
    
    
    # BERT argument
    parser.add_argument('--bert_type', default='bert-base-multilingual-cased')
    parser.add_argument('--tokenizer_max_length', type=int, default=128,  help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden dim BERT")
    parser.add_argument("--use_token", default=False, action='store_true', help="use token level or not")
    parser.add_argument("--compl_classifier", default=False, action='store_true', help="use 2 layers classifier")
    parser.add_argument("--roberta", default=False, action='store_true', help="use roberta")
    
    # prediction argument
    parser.add_argument("--threshold", default=0.5, type=float, help="Threshold to decide the label given the logits")
    
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S')
    logger.info(args)
    
    main(args)
