import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable


def _truncate_seq_pair(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break

        else:
            tokens.pop()

def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    # sent_a, title, sent_b = sentence
    tokens = tokenizer.tokenize(sentence)

    # tokens_b = None
    # tokens_t = None
    # if sent_b and title:
    #     tokens_t = tokenizer.tokenize(title)
    #     tokens_b = tokenizer.tokenize(sent_b)
    #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4 - len(tokens_t))
    # else:
    #     # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 1:
        tokens = tokens[:(max_seq_length - 1)]

    tokens =  ["[CLS]"] + tokens  #+ ["[SEP]"]
    segment_ids = [0] * len(tokens)
    # if tokens_b and tokens_t:
    #     tokens = tokens + tokens_t + ["[SEP]"] + tokens_b + ["[SEP]"]
    #     segment_ids += [1] * (len(tokens_b) + len(tokens_t) + 2)
    #print (tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids





def tok2int_list(src_list, tokenizer, max_seq_length, max_seq_size=-1):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    for step, sent in enumerate(src_list):
        input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        seg_padding.append(input_seg)
    if max_seq_size != -1:
        inp_padding = inp_padding[:max_seq_size]
        msk_padding = msk_padding[:max_seq_size]
        seg_padding = seg_padding[:max_seq_size]
        inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
        msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
        seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding


class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, test=False, cuda=True, batch_size=8):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.tokenizer_max_length
        self.tweet_num = 5    #no.of tweets per user
        # self.label_map = label_map
        self.threshold = args.threshold
        self.data_path = data_path
        users = self.read_file(data_path)
        self.users = users
        inputs, labels, ids = list(zip(* users))
        self.inputs = inputs
        self.labels = labels
        self.ids = ids
        self.test = test

        self.total_num = len(users)
        if self.test:
            self.total_step = self.total_num / batch_size #np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = self.total_num / batch_size
            self.shuffle()
        self.step = 0



    def read_file(self, data_path):
        users = []
        with open(data_path, 'r', encoding='utf-8') as fin:
            for step, line in enumerate(fin):
                if step > 100000:
                    break
                entry = json.loads(line.encode('utf-8'))
                if entry['label'] is None:
                    continue
                tweets = entry['tweets']
                tweet_text = list()
                for tweet in tweets:
                    tweet_text.append(tweet['tweet_text'])   #TODO process tweets and append
                    # evi_list.append([self.process_sent(claim), self.process_wiki_title(evidence[0]),
                    #                  self.process_sent(evidence[2])])
                label = entry['label']
                ids = entry['user_id']
                # evi_list = evi_list[:self.evi_num]
                users.append([tweet_text, label, ids])
        return users


    def shuffle(self):
        np.random.shuffle(self.users)

    def process_sent(self, sentence):
        sentence = re.sub(" LSB.*?RSB", "", sentence)
        sentence = re.sub("LRB RRB ", "", sentence)
        sentence = re.sub("LRB", " ( ", sentence)
        sentence = re.sub("RRB", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub("LRB", " ( ", title)
        title = re.sub("RRB", " )", title)
        title = re.sub("COLON", ":", title)
        return title


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        if self.step < self.total_step:
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            labels = self.labels[self.step * self.batch_size : (self.step+1)*self.batch_size]
            ids = self.ids[self.step * self.batch_size : (self.step+1)*self.batch_size]

            inp_padding_inputs, msk_padding_inputs, seg_padding_inputs = [], [], []
            for step in range(len(inputs)):
                inp, msk, seg = tok2int_list(inputs[step], self.tokenizer, self.max_len, self.tweet_num)
                inp_padding_inputs += inp
                msk_padding_inputs += msk
                seg_padding_inputs += seg

            inp_tensor_input = Variable(
                torch.LongTensor(inp_padding_inputs)).view(-1, self.tweet_num, self.max_len)
            msk_tensor_input = Variable(
                torch.LongTensor(msk_padding_inputs)).view(-1, self.tweet_num, self.max_len)
            seg_tensor_input = Variable(
                torch.LongTensor(seg_padding_inputs)).view(-1, self.tweet_num, self.max_len)
            lab_tensor = Variable(
                torch.FloatTensor(labels))
            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
                lab_tensor = lab_tensor.cuda()
            self.step += 1
            return (inp_tensor_input, msk_tensor_input, seg_tensor_input), lab_tensor, ids
        else:
            self.step = 0
            if not self.test:
                self.shuffle()
                inputs, labels, ids = list(zip(*self.users))
                self.inputs = inputs
                self.labels = labels
                self.ids = ids
            raise StopIteration()

class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, cuda=True, batch_size=8):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.tokenizer_max_length
        self.tweet_num = 5   #no.of tweets per user
        # self.label_map = label_map
        self.threshold = args.threshold
        self.data_path = data_path
        users = self.read_file(data_path)
        self.users = users
        inputs, labels, ids = list(zip(* users))
        self.inputs = inputs
        self.ids = ids
        self.labels = labels

        self.total_num = len(users)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        self.step = 0

    def process_sent(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title


    def read_file(self, data_path):
        users = []
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                if step > 100000:
                    break
                entry = json.loads(line.encode('utf-8'))
                tweets = entry['tweets']
                tweet_text = list()
                for tweet in tweets:
                    tweet_text.append(tweet['tweet_text'])   #TODO process tweets and append
                    # evi_list.append([self.process_sent(claim), self.process_wiki_title(evidence[0]),
                    #                  self.process_sent(evidence[2])])
                userid = entry['user_id']
                label = entry['label']
                # evi_list = evi_list[:self.evi_num]
                users.append([tweet_text, label, userid])
        return users
        #         id = instance['id']
        #         evi_list = evi_list[:self.evi_num]
        #         users.append([evi_list, id])
        # return users


    def shuffle(self):
        np.random.shuffle(self.users)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        if self.step < self.total_step:
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            labels = self.labels[self.step * self.batch_size : (self.step+1)*self.batch_size]
            # user_ids = self.ids[self.step * self.batch_size : (self.step+1)*self.batch_size]

            ids = self.ids[self.step * self.batch_size : (self.step+1)*self.batch_size]
            inp_padding_inputs, msk_padding_inputs, seg_padding_inputs = [], [], []
            for step in range(len(inputs)):
                inp, msk, seg = tok2int_list(inputs[step], self.tokenizer, self.max_len, self.tweet_num)
                inp_padding_inputs += inp
                msk_padding_inputs += msk
                seg_padding_inputs += seg

            inp_tensor_input = Variable(
                torch.LongTensor(inp_padding_inputs)).view(-1, self.tweet_num, self.max_len)
            msk_tensor_input = Variable(
                torch.LongTensor(msk_padding_inputs)).view(-1, self.tweet_num, self.max_len)
            seg_tensor_input = Variable(
                torch.LongTensor(seg_padding_inputs)).view(-1, self.tweet_num, self.max_len)

            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()

            self.step += 1
            return (inp_tensor_input, msk_tensor_input, seg_tensor_input), labels, ids
        else:
            self.step = 0
            raise StopIteration()
