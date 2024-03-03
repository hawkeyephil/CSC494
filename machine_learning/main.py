#Import Statements
import pathlib as pl
import os
import machine_learning.SentimentAnalyzer as sa 
import matplotlib.pyplot as plt 
import pandas as pd

#Folder path to FEDStatements 
folder_path = pl.Path(__file__).parent / 'FEDStatements'

#Changes the current working directory to the above path 
os.chdir(folder_path) 

#Lists to contain sentiment scores and meeting dates 
sentiment_scores = list() 
meeting_dates = list()

#Reads individual file and returns the sentiment score that the analyzer gave it
def read_txt_file(file_path): 
    try:
        with open(file_path, 'r') as f:
            file_content = f.read() 
            #Debugging
            #print(file_content)   
            sentiment_score = sa.sentiment_analyzer(file_content) 
            return(sentiment_score)
    except FileNotFoundError:
        return "File not found. Please provide a valid file path." 

#Iterate through the list of files in the folder 
for file in os.listdir():
    if file.endswith(".txt"):
        file_path = os.path.join(folder_path, file) 
        file = file[:10] 
        meeting_dates.append(file) 
        sentiment_scores.append(read_txt_file(file_path)) 

#Creates a dataframe with the sentiment scores and meeting dates 
#dataframe = pd.DataFrame({'Column1': sentiment_scores, 'Column2': meeting_dates})
#Saves the dataframe to a CSV file
#dataframe.to_csv('filepath', index=False, header=False)

#Spaces meeting dates on the x-axis 
plt.xticks(range(0, len(meeting_dates), 20), meeting_dates[::20], rotation=45)
#Generates the visualization 
plt.plot(meeting_dates, sentiment_scores) 
#Plot formatting 
plt.xlabel('Meeting Date') 
plt.ylabel('Sentiment Polarity Score') 
plt.title('Historical FED Statement Sentiment') 
plt.grid()
#Displays the plot 
plt.show() 

