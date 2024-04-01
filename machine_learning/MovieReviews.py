from pathlib import Path 
import pandas as pd 
import LexiconInference as sa 
import NBoWInference as nbow 
import CNNInference as cnn

#Relative filepath to .csv file with URLs to Movie Reviews/Ratings  
path = Path(__file__).parent.parent / "MovieReviews.csv" 
#Pandas Dataframe that contains all of the reviews and ratings 
df_Reviews = pd.read_csv(path, sep = ',') 

#Dataframe to store sentiment scores 
sentiment = {'Review': [], 'Sentiment': [], 'Lexicon_Sentiment': [], 'NBoW_Sentiment': [], 'CNN_Sentiment': []}
df_Scores = pd.DataFrame(sentiment)

#Extracts sentiment from every review with each model  
for index, row in df_Reviews.iterrows(): 
    review = row[0]
    rating = row[1] 

    #Extract sentiment using the lexicon approach 
    lexicon_Score = sa.sentiment_analyzer(review, 'Harvard-IV') 
    lexicon_Polarity = lexicon_Score['polarity']
    if float(lexicon_Polarity) > 0.00:
        lexicon_Class = 1 
    else:
        lexicon_Class = 0

    #Extract the sentiment using the NBoW model 
    nbow_Scores = nbow.predict_Sentiment(review, 'General') 
    if nbow_Scores[0] == 1: 
        nbow_Class = 1 
    elif nbow_Scores[0] == 0: 
        nbow_Class = 0 
    else: 
        nbow_Class = 'N/A'

    #Extract sentiment using the cnn model 
    cnn_Scores = cnn.predict_Sentiment(review, 'General')
    if cnn_Scores[0] == 1: 
        cnn_Class = 1 
    elif cnn_Scores[0] == 0: 
        cnn_Class = 0 
    else: 
        cnn_Class = 'N/A'

    #Appends the sentiment scores for that review to the scores dataframe 
    sentiment_Scores = {'Review': review, 'Sentiment': rating, 'Lexicon_Sentiment': lexicon_Class, 'NBoW_Sentiment': nbow_Class, 'CNN_Sentiment': cnn_Class} 
    #print(sentiment_Scores) 
    df_Scores.append(sentiment_Scores, ignore_index = True)
    