#Import Statements
import datasets 
import pandas as pd
import LexiconInference as sa 
import NBoWInference as nbow 
import CNNInference as cnn 
import plotly.express as px 

#Stock Tweets Sentiment dataset from Hugging Face 
#train_Data, test_Data = datasets.load_dataset("emad12/stock_tweets_sentiment", split = ["train", "test"]) 
#IMDb dataset --> split into trainning and test sets (25000 reviews each) 
train_Data, test_Data = datasets.load_dataset("imdb", split = ["train", "test"])

#Creates a subset of the training data for validation 
train_Valid_Data = train_Data.train_test_split(test_size = 0.25) 
train_Data = train_Valid_Data["train"]
valid_Data = train_Valid_Data["test"]
print(len(train_Data), len(valid_Data), len(test_Data))
print(valid_Data)
#Stores and sorts tweets with corresponding sentiment scores in ascending order 
#validation_Data = pd.DataFrame({'tweets': valid_Data["tweet"], 'sentiment': valid_Data["sentiment"]}) 
validation_Data = pd.DataFrame({'text': valid_Data["text"], 'sentiment': valid_Data["label"]})
sorted_Validation_Data = validation_Data.sort_values(by = ['sentiment'])

#Dataframe to store sentiment scores 
sentiment = {'True Sentiment': [], 'Predicted Lexicon Sentiment': [], 'Predicted NBoW Sentiment': [], 'Predicted CNN Sentiment': []}
df_Scores = pd.DataFrame(sentiment)

#Iterates through every tweet and makes a sentiment inference for each model (this should be a function) 
for index, row in sorted_Validation_Data.iterrows(): 
    text = row['text'] 
    sentiment = row['sentiment'] 

    #lexicon_Score = sa.sentiment_analyzer(tweet, 'LM').get('polarity') 
    lexicon_Score = sa.sentiment_analyzer(text, 'Harvard-IV').get('polarity')
    #if(lexicon_Score > 0.25): 
        #lexicon_Class = 1 
    #elif(lexicon_Score < -0.25): 
        #lexicon_Class = -1 
    #else: 
        #lexicon_Class = 0
    if float(lexicon_Score) > 0.00:
        lexicon_Class = 1 
    else:
        lexicon_Class = 0

    #nbow_Score = nbow.predict_Sentiment(tweet, 'Finance')[0] 
    nbow_Score = nbow.predict_Sentiment(text, 'General')[0]
    #match nbow_Score:
        #case 0:
            #nbow_Class = -1
        #case 2:
            #nbow_Class = 1 
        #case _: 
            #nbow_Clas = 0 
    if nbow_Score == 1: 
        nbow_Class = 1 
    elif nbow_Score == 0: 
        nbow_Class = 0 
    else: 
        nbow_Class = 'N/A'

    #cnn_Score = cnn.predict_Sentiment(tweet, 'Finance')[0]
    cnn_Score = cnn.predict_Sentiment(text, 'General')[0]
    #match cnn_Score:
        #case 0:
            #cnn_Class = -1
        #case 2:
            #cnn_Class = 1
        #case _: 
            #cnn_Class = 0 
    if cnn_Score == 1: 
        cnn_Class = 1 
    elif cnn_Score == 0: 
        cnn_Class = 0 
    else: 
        cnn_Class = 'N/A' 

    sentiment_Scores = {'True Sentiment': sentiment, 'Predicted Lexicon Sentiment': lexicon_Class, 'Predicted NBoW Sentiment': nbow_Class, 'Predicted CNN Sentiment': cnn_Class} 
    df_Scores = df_Scores.append(sentiment_Scores, ignore_index = True)

#plotly confusion matrix visualization 
colors = [(0, 'red'), (0.5, 'white'), (1, 'green')]
fig = px.imshow(df_Scores, labels={"x": "Model", "y": "Tweet"}, color_continuous_scale = colors, title = "Movie Review Sentiment Classifer Confusion Matrix") 
fig.show() 
