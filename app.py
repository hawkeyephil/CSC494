#Import Statements
from flask import Flask, render_template, request, redirect, url_for, session 
from dotenv import load_dotenv 
import os
import machine_learning.SentimentAnalyzer as sa 

#Gives each file a unique name 
app = Flask(__name__) 
#Sets secret key
app.secret_key = os.getenv('SECRET_KEY') 

#Decorator which specifies url endpoint 
@app.route('/') 
#Function that defines that the index page is displayed 
def index():
    return render_template('index.html') 

#Route to the sandbox page 
@app.route('/sandbox') 
def sandbox(): 
    return render_template('sandbox.html')

#Sandbox sentiment route 
@app.route('/analyze_sentiment', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        # Get the user input from the textarea
        user_input = request.form['user_text'] 
        print(user_input)

        #Analyze the sentiment using the dictionary approach 
        sentiment_Scores = sa.sentiment_analyzer(user_input) 

        #Extract the polarity and subjectivity scores 
        polarity = sentiment_Scores['polarity']
        subjectivity = sentiment_Scores['subjectivity'] 
    
    #Display the results to the page 
    return render_template('sandbox.html', polarity = polarity, subjectivity = subjectivity)

#Starts server and runs the app locally at port 5005
app.run(host = 'localhost', port = 5005, debug = True) 




