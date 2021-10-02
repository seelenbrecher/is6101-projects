import argparse
import pandas as pd
import stweet as st
from shutil import copyfile

def collect_tweet_by_tweet_ids(tweet_ids, out_path):
    """
    Scrape tweets by tweet id.
    Output in out_path, and return all the collected tweets.
    Not sure if is there any rate limit.
    Contains user_id, full_text, in_reply_to_user_id, in_reply_to_status_id

    TODO: If we there is a rate limit, we need to batch the tweet ids
    """
    tweets_by_ids_task = st.TweetsByIdsTask(tweet_ids)
    tweets_collector = st.CollectorTweetOutput()

    st.TweetsByIdsRunner(
        tweets_by_ids_task=tweets_by_ids_task,
        tweet_outputs=[tweets_collector, st.CsvTweetOutput(out_path)]
    ).run()

    tweets = tweets_collector.get_scrapped_tweets()

    return tweets


def retrieve_queried_tweet_ids(out_path):
    """
    Read the outfile to obtain the last tweet_ids queried
    """
    try:
        print('trying outfile query retrieval')
        output_pd = pd.read_csv(out_path)
        return output_pd['id_str'].tolist()
    except:
        print('returning nempty array')
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


def process(args):
    # tweet_ids = stores all unqueried tweet ids that we want to query
    print('inside process')
    tweet_ids = retrieve_tweet_ids(args.in_path)
    
    counter = 0
    prev_tweet_ids_len = len(tweet_ids)

    while True:
        # get queried tweets
        queried_tweet_ids = retrieve_queried_tweet_ids(args.out_path)
        
        # all tweet_ids have been queried
        if len(tweet_ids) == 0:
            break
        
        # get unqueried tweet ids
        for id in queried_tweet_ids:
            try:
                print('queried tweet id removed')
                tweet_ids.remove(str(id))
            except:
                # id has been removed before
                pass
        
        print('queried_tweet_ids', queried_tweet_ids)
        print('unqueried_tweet_ids', tweet_ids)
        
        
        # if the number of unqueried tweet ids still the same until 3 runs, something wrong with one of the data. remove it 1by1
        if len(tweet_ids) == prev_tweet_ids_len:
            counter += 1
        if counter == 3:
            tweet_ids = tweet_ids[1:]
            counter = 0
        prev_tweet_ids_len = len(tweet_ids)
        
        # create backup file
        try:
            copyfile(args.out_path, '{}.bk'.format(args.out_path))
        except:
            # out_path has not created yet
            pass
        
        try:
            collect_tweet_by_tweet_ids(tweet_ids, args.out_path)
        except:
            pass
        


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