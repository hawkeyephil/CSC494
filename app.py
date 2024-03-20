#Import Statements
from flask import Flask, jsonify, render_template, request, redirect, url_for, session 
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

polarityglobe = 0.0

#Sandbox sentiment route 
@app.route('/analyze_sentiment', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        # Get the user input from the textarea
        user_input = request.form['user_text'] 
        print(user_input)

        #Analyze the sentiment using the dictionary approach 
        sentiment_Scores = sa.sentiment_analyzer(user_input) 

        #Extract the polarity and subjectivity scores 
        polarity = sentiment_Scores['polarity']
        subjectivity = sentiment_Scores['subjectivity'] 

        sentiment_bar(polarity)
    #Display the results to the page 
    return render_template('sandbox.html', polarity=polarity, subjectivity=subjectivity)

def sentiment_bar(score): 
    global polarityglobe 
    polarityglobe = score 
    print('test1')

@app.route('/calculate_value')
def calculate_value():
    # Your computation logic here
    result = 42  # Replace with your actual calculation 
    print('test2')
    return jsonify({'polarity': polarityglobe})

@app.route('/analyze_sentiment1', methods=['POST'])
def analyze1_sentiment():
    if request.method == 'POST':
        # Get the user input from the textarea (you can adapt this part)
        #user_input = request.form['user_text']

        # Analyze the sentiment using your existing logic
        # Replace this with your actual sentiment analysis code
        polarity = polarityglobe  # Replace with your calculated polarity
        print('testttt')
        # Return the polarity as JSON
        return jsonify({'polarity': polarity})

#Starts server and runs the app locally at port 5005
app.run(host = 'localhost', port = 5005, debug = True) 
polarityglobe = 0.0




