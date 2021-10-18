import argparse
import json

from sklearn.metrics import precision_recall_fscore_support


def main(args):
    data = {}
    with open(args.input, 'r') as f:
        for step, x in enumerate(f):
            x = json.loads(x)
            if x['id'] not in data:
                data[x['id']] = []
            data[x['id']].append(x['predicted_label'])
        
    ground_truth = {}
    with open(args.ground_truth, 'r') as f:
        for step, x in enumerate(f):
            if step > 100000:
                break
            x = json.loads(x.encode('utf-8'))
            if x['label'] is None:
                continue
            ground_truth[x['user_id']] = x['label']
    
    golds = []
    preds = []
    
    for userid in data:
        if userid not in ground_truth:
            continue
        true_cnt = data[userid].count(True)
        false_cnt = data[userid].count(False)
        
        if true_cnt > false_cnt:
            preds.append(1)
        else:
            preds.append(0)
        
        golds.append(ground_truth[userid])

    prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average='binary')
    print('prec = {}, rec={}, f1={}'.format(prec, rec, f1))
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # BERT arguments
    parser.add_argument('--input', default='output/fake-news-classifier-v0-dev.json')
    parser.add_argument('--ground_truth', default='data/coaid/FNSC_test.json')
    args = parser.parse_args()
    main(args)
