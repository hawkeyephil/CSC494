import pandas as pd


# Language
language = "en"

#scraper = sntwitter.TwitterSearchScraper(query + " lang:" + language) 


tweets = []
tweet_count = 10  


import snscrape.modules.twitter as sntwitter
import pandas

# Creating list to append tweet data to
tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('COVID Vaccine since:2022-01-01 until:2022-05-31').get_items()):
    if i>5:
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username']) 

#Filepath and filenames 
test_File = '~/Downloads/test_File.csv'

tweets_df2.to_csv(test_File, index=False)

