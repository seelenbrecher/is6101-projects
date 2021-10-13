import pandas as pd
import time 
import argparse
import json

start = time.time()
coaid = pd.read_csv('coaid_ext.csv')
print('time to read', time.time()-start)
userMap = {}
unmappedTweets = []

def getLabel(tweetId):
    # st = time.time()
    # print(coaid.dtypes)
    tweetId = int(tweetId)
    try:
        ind = coaid.index[coaid['tweet_id']==tweetId].tolist()
        # print('time to search', time.time()-st)
        # print('index',ind)
        label = int(coaid.loc[ind[0]]['label'])
    # print(label)
    except:
        print('Cannot get label for:', tweetId)
        unmappedTweets.append(tweetId)
        label = None

    return label

def createDataset(args):
    with open(args.in_path, 'r') as f:
        for x in f:
            tweetentry = json.loads(x)
            if tweetentry['user_id'] in userMap:
                tweet = { 'tweet_id': tweetentry['id_str'], 'tweet_text': tweetentry['full_text']}
                userMap[tweetentry['user_id']]['tweets'].append(tweet)
                # print(userMap[tweetentry['user_id']])
                if len(userMap[tweetentry['user_id']]['tweets']) == 5:
                    with open(args.out_path, 'a') as f:
                        f.write('{}\n'.format(json.dumps(userMap[tweetentry['user_id']])))
                    userMap[tweetentry['user_id']]['tweets'] = []
                    print('appended to output file')
            else:
                tweet = { 'tweet_id': tweetentry['id_str'], 'tweet_text': tweetentry['full_text']}
                # print(tweetentry['id_str'])
                label = getLabel(tweetentry['id_str'])
                # if label != none:
                userMap[tweetentry['user_id']] = {'user_id':tweetentry['user_id'], 'tweets': [tweet], 'label': label}
                # print(userMap[tweetentry['user_id']])
        #   data.append(x['id_str'])
        # return data

    

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path')
    parser.add_argument('--out_path')
    # parser.add_argument('--by_tweet_ids', action='store_true')
    args = parser.parse_args()

    # if args.by_tweet_ids:
    #     print('true')
    createDataset(args)

main()