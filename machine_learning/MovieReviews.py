from pathlib import Path 
import pandas as pd 
import LexiconInference as sa 
import NBoWInference as nbow 
import CNNInference as cnn 
from sklearn import metrics 
import matplotlib.pyplot as plt 
import plotly.express as px 
import seaborn as sns 

#Relative filepath to .csv file with URLs to Movie Reviews/Ratings  
path = Path(__file__).parent.parent / "MovieReviews.csv" 
#Pandas Dataframe that contains all of the reviews and ratings 
df_Reviews = pd.read_csv(path, sep = ',') 

#Dataframe to store sentiment scores 
sentiment = {'True Sentiment': [], 'Predicted Lexicon Sentiment': [], 'Predicted NBoW Sentiment': [], 'Predicted CNN Sentiment': []}
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
    sentiment_Scores = {'True Sentiment': rating, 'Predicted Lexicon Sentiment': lexicon_Class, 'Predicted NBoW Sentiment': nbow_Class, 'Predicted CNN Sentiment': cnn_Class} 
    #print(sentiment_Scores) 
    df_Scores = df_Scores.append(sentiment_Scores, ignore_index = True) 

#Lexicon confusion matrix 
lexicon_Confusion_Matrix = metrics.confusion_matrix(df_Scores['True Sentiment'], df_Scores['Predicted Lexicon Sentiment']) 
print(lexicon_Confusion_Matrix) 

#NBoW confusion matrix 
nbow_Confusion_Matrix = metrics.confusion_matrix(df_Scores['True Sentiment'], df_Scores['Predicted NBoW Sentiment']) 
print(nbow_Confusion_Matrix) 

#CNN confusion matrix
cnn_Confusion_Matrix = metrics.confusion_matrix(df_Scores['True Sentiment'], df_Scores['Predicted CNN Sentiment']) 
print(cnn_Confusion_Matrix) 

#matplotlib visualization
labels = [["Positive", "Negative"], ["Positive", "Negative"]]
plt.imshow(lexicon_Confusion_Matrix, cmap='Reds', interpolation='nearest') 
plt.xticks([0, 1], labels[0])
plt.yticks([0, 1], labels[1])
plt.colorbar()
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.title('Lexicon Confusion Matrix')
#plt.show() 

#plotly visualization 
colors = [(0, 'red'), (0.5, 'white'), (1, 'green')]
fig = px.imshow(df_Scores, labels={"x": "Model", "y": "Review"}, color_continuous_scale=colors, title="Star Wars Reviews Sentiment Analysis")
fig.show() 