import argparse
import stweet as st

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path')
    parser.add_argument('--out_path')
    parser.add_argument('--by_tweet_ids', action='store_true')
    args = parser.parse_args()

    if args.by_tweet_ids:
        tweet_ids = []
        with open(args.in_path, 'r') as f:
            for x in f:
                tweet_ids.append(x.strip())
        collect_tweet_by_tweet_ids(tweet_ids, args.out_path)

main()
