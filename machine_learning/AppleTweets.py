from pathlib import Path 
import pandas as pd 
import LexiconInference as sa 
import NBoWInference as nbow 
import CNNInference as cnn 
from sklearn import metrics 
import matplotlib.pyplot as plt 
import plotly.express as px 

#Relative filepath to .csv file with URLs to Apple Stock Tweets/Dates  
path = Path(__file__).parent.parent / "AAPLTweets.csv" 
#Pandas Dataframe that contains all of the Tweets and Dates 
df_Tweets = pd.read_csv(path, sep = ',') 

#Dataframe to store sentiment scores 
sentiment = {'Date': [], 'Tweet': [], 'Predicted Lexicon Sentiment': [], 'Predicted NBoW Sentiment': [], 'Predicted CNN Sentiment': []}
df_Scores = pd.DataFrame(sentiment)

#Extracts sentiment from every Tweet with each model  
for index, row in df_Tweets.iterrows(): 
    date = row[0]
    tweet = row[1] 

    #Extract sentiment using the lexicon approach 
    lexicon_Score = sa.sentiment_analyzer(tweet, 'LM') 
    lexicon_Polarity = lexicon_Score['polarity']
    if float(lexicon_Polarity) > 0.25:
        lexicon_Class = 1 
    elif float(lexicon_Polarity) < -0.25:
        lexicon_Class = -1
    else:
        lexicon_Class = 0

    #Extract the sentiment using the NBoW model 
    nbow_Scores = nbow.predict_Sentiment(tweet, 'Finance') 
    if nbow_Scores[0] == 2: 
        nbow_Class = 1 
    elif nbow_Scores[0] == 0: 
        nbow_Class = -1
    else: 
        nbow_Class = 0

    #Extract sentiment using the cnn model 
    cnn_Scores = cnn.predict_Sentiment(tweet, 'Finance')
    if cnn_Scores[0] == 2: 
        cnn_Class = 1 
    elif cnn_Scores[0] == 0: 
        cnn_Class = -1 
    else: 
        cnn_Class = 0 

    #Appends the sentiment scores for that Date and Tweet to the scores dataframe 
    sentiment_Scores = {'Date': date, 'Tweet': tweet, 'Predicted Lexicon Sentiment': lexicon_Class, 'Predicted NBoW Sentiment': nbow_Class, 'Predicted CNN Sentiment': cnn_Class} 
    #print(sentiment_Scores) 
    df_Scores = df_Scores.append(sentiment_Scores, ignore_index = True) 

#Saves sentiment scores to a csv file 
filename = '~/Downloads/TweetSentiments.csv' 
df_Scores.to_csv(filename, index = False, header = True) 