# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:07:58 2025
@author: gowtham.balachan
"""
import praw
import pandas as pd
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import time
import logging
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Get Reddit API credentials from environment variables
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
user_agent = os.getenv('REDDIT_USER_AGENT')

# Validate credentials
if not all([client_id, client_secret, user_agent]):
    raise ValueError("Missing Reddit API credentials in environment variables")

def initialize_reddit():
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

def fetch_subreddit_posts(reddit, subreddit_name, start_time, end_time, num_posts, max_retries=300):
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            posts = []
            logging.info(f"Attempt {retry_count + 1}: Processing subreddit: {subreddit_name}")
            
            for post in reddit.subreddit(subreddit_name).new(limit=num_posts):
                if start_of_today <= post.created_utc <= end_of_today:
                    logging.info(f"Found post for: {subreddit_name}")
                    posts.append({
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'author': post.author.name if post.author else 'Deleted',
                        'upvotes': post.score,
                        'comments': post.num_comments,
                        'created_utc': post.created_utc,
                        'url': post.url,
                        'content': post.selftext
                    })
            
            # If we found no posts but had no errors, that's okay - return empty list
            return posts
            
        except (RequestException, ConnectionResetError) as e:
            retry_count += 1
            logging.warning(f"Attempt {retry_count}/{max_retries} failed for {subreddit_name}: {str(e)}")
            
            if retry_count >= max_retries:
                logging.error(f"Max retries reached for {subreddit_name}")
                return []
                
            logging.info("Waiting 5 seconds before retrying...")
            time.sleep(5)
            
            # Reinitialize Reddit client on each retry
            try:
                reddit = initialize_reddit()
            except Exception as e:
                logging.warning(f"Failed to reinitialize Reddit client: {str(e)}")
                continue
                
        except Exception as e:
            logging.error(f"Unexpected error processing {subreddit_name}: {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                return []
            time.sleep(5)
            continue

def main():
    # Specify the subreddit and number of posts to fetch
    subreddits = [
        'MachineLearning', 'singularity',
        'ArtificialInteligence', 'compsci'
    ]
    num_posts = 50  # Number of posts to fetch per subreddit

    # Get today's start and end timestamps
    global start_of_today, end_of_today  # Make these global for the fetch_subreddit_posts function
    today = datetime.now(timezone.utc).date()
    start_of_today = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc).timestamp()
    end_of_today = datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc).timestamp()

    all_posts = []
    processed_subreddits = []
    
    for subreddit in subreddits:
        # Initialize a new Reddit instance for each subreddit
        try:
            reddit = initialize_reddit()
        except Exception as e:
            logging.error(f"Failed to initialize Reddit client: {str(e)}")
            continue
            
        # Try to fetch posts for this subreddit
        subreddit_posts = fetch_subreddit_posts(
            reddit, subreddit, start_of_today, end_of_today, num_posts
        )
        
        # If we got any posts, add them to our results
        if subreddit_posts:
            all_posts.extend(subreddit_posts)
            processed_subreddits.append(subreddit)
            
        # Add delay between subreddits
        time.sleep(2)

    # Create a DataFrame
    df = pd.DataFrame(all_posts)

    # Convert created_utc to datetime and save to CSV
    if not df.empty:
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        output_file = 'reddit_data.csv'
        df.to_csv(output_file, index=False)
        logging.info(f"DataFrame created with {len(df)} posts and saved to '{output_file}'")
    else:
        logging.warning("No posts found for today.")

    logging.info(f"Processed Subreddits: {processed_subreddits}")
    logging.info("\nFirst few rows of the DataFrame:")
    logging.info(df.head())

if __name__ == "__main__":
    main()