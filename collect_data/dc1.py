import argparse
import json
import requests
import tqdm


def retrieve_queried_tweet_ids(out_path):
    """
    Read the outfile to obtain the last tweet_ids queried
    """
    try:
      print('trying outfile query retrieval')
      data = []
      with open(out_path, 'r') as f:
        for x in f:
          x = json.loads(x)
          data.append(x['id_str'])
        return data
    except:
      print('return empty array')
      return []

def retrieve_tweet_ids(in_path):
    """
    Read the in_file all tweet ids we wish to query
    """
    print('retrieving input tweets')
    tweet_ids = []
    with open(in_path, 'r') as f:
        for x in f:
            tweet_ids.append(x.strip())
    return tweet_ids

def get_tweet_by_id(id):
    url = 'https://cdn.syndication.twimg.com/tweet?features=tfw_experiments_cookie_expiration%3A1209600%3Btfw_horizon_tweet_embed_9555%3Ahte%3Btfw_space_card%3Aoff&id={}&lang=en'.format(id)

    try:
        resp = requests.get(url.format(id)).json()
    except:
        print("can't get tweet id = {}".format(id))
        return {}

    data = {
      'id_str': resp['id_str'],
      'full_text': resp['text'],
      'lang': resp['lang'],
      'user_name': resp['user']['screen_name'],
      'user_id': resp['user']['id_str'],
      'created_at': resp['created_at'],
    }

    if 'in_reply_to_screen_name' in resp:
        data['in_reply_to_screen_name'] = resp['in_reply_to_screen_name']
    if 'in_reply_to_status_id_str' in resp:
        data['in_reply_to_status_id_str'] = resp['in_reply_to_status_id_str']
    if 'in_reply_to_user_id_str' in resp:
        data['in_reply_to_user_id_str'] = resp['in_reply_to_user_id_str']

    data['user_mentions_user_ids'] = []
    data['user_mentions_screen_names'] = []
    for mention in resp['entities']['user_mentions']:
        data['user_mentions_user_ids'].append(mention['id_str'])
        data['user_mentions_screen_names'].append(mention['screen_name'])

    return data


def process(args):
    # tweet_ids = stores all unqueried tweet ids that we want to query
    print('inside process')
    tweet_ids = retrieve_tweet_ids(args.in_path)
    queried_tweet_ids = retrieve_queried_tweet_ids(args.out_path)

    for id in queried_tweet_ids:
      try:
          tweet_ids.remove(str(id))
      except:
          # id has been removed before
          pass
    
    with open(args.out_path, 'a') as f:
      for step, id in tqdm.tqdm(enumerate(tweet_ids)):
        data = get_tweet_by_id(id)
        if data != {}:
          f.write('{}\n'.format(json.dumps(data)))
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path')
    parser.add_argument('--out_path')
    parser.add_argument('--by_tweet_ids', action='store_true')
    args = parser.parse_args()

    if args.by_tweet_ids:
        print('true')
        process(args)

main()
