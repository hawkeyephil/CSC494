#Import Statements
import pathlib as pl
import os
import LexiconInference as sa 
import NBoWInference as nbow 
import CNNInference as cnn
import matplotlib.pyplot as plt 
import pandas as pd 

#Folder path to FEDStatements 
folder_path = pl.Path(__file__).parent.parent / 'FEDStatements'

#Changes the current working directory to the above path 
os.chdir(folder_path) 

#Lists to contain sentiment scores and meeting dates 
lexicon_Scores = list() 
nbow_Scores = list() 
cnn_Scores = list()
meeting_Dates = list() 

#Reads individual file and returns the sentiment score that the analyzer gave it
def read_txt_file(file_path): 
    try:
        with open(file_path, 'r') as f:
            file_content = f.read() 
            #Debugging
            #print(file_content)   
            lexicon_Score = sa.sentiment_analyzer(file_content, 'LM').get('polarity')
            if(lexicon_Score > 0.25): 
                lexicon_Score = 1 
            elif(lexicon_Score < -0.25): 
                lexicon_Score = -1 
            else: 
                lexicon_Score = 0

            nbow_Score = nbow.predict_Sentiment(file_content, 'Finance')[0] 
            match nbow_Score:
                case 0:
                    nbow_Score = -1
                case 2:
                    nbow_Score = 1
                case _: 
                    nbow_Score = 0 

            cnn_Score = cnn.predict_Sentiment(file_content, 'Finance')[0] 
            match cnn_Score:
                case 0:
                    cnn_Score = -1
                case 2:
                    cnn_Score = 1
                case _: 
                    cnn_Score = 0
            sentiment_Scores = {'lexicon': lexicon_Score, 'nbow': nbow_Score, 'cnn': cnn_Score}
            return(sentiment_Scores)
    except FileNotFoundError:
        return "File not found. Please provide a valid file path." 

#Iterate through the list of files in the folder 
for file in os.listdir():
    if file.endswith(".txt"):
        file_path = os.path.join(folder_path, file) 
        file = file[:10] 
        meeting_Dates.append(file) 
        sentiment_Scores = read_txt_file(file_path)
        lexicon_Scores.append(sentiment_Scores.get('lexicon')) 
        nbow_Scores.append(sentiment_Scores.get('nbow'))
        cnn_Scores.append(sentiment_Scores.get('cnn')) 

#Creates a dataframe with the sentiment scores and meeting dates 
dataframe = pd.DataFrame({'Meeting Dates': meeting_Dates, 'Lexicon Score': lexicon_Scores, 'NBoW Score': nbow_Scores, 'CNN Score': cnn_Scores})
#Saves the dataframe to a CSV file in the downloads folder 
filename = '~/Downloads/FedStatementSentimentScores.csv' 
dataframe.to_csv(filename, index = False, header = True) 


#Spaces meeting dates on the x-axis 
#plt.xticks(range(0, len(meeting_dates), 20), meeting_dates[::20], rotation=45)
#Generates the visualization 
#plt.plot(meeting_dates, sentiment_scores) 
#Plot formatting 
#plt.xlabel('Meeting Date') 
#plt.ylabel('Sentiment Polarity Score') 
#plt.title('Historical FED Statement Sentiment') 
#plt.grid()
#Displays the plot 
#plt.show() 

