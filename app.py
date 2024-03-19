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

@app.route('/sandbox') 
def sandbox(): 
    return render_template('sandbox.html')

@app.route('/process_text', methods=['POST']) 
def process_text(): 
    #User text entered into form 
    user_text = request.form.get('user_text') 
    #Process text to be sent back to user 
    sentiment_score = sa.sentiment_analyzer(user_text) 
    #Stores sentiment_score in session 
    session['processed_result'] = sentiment_score
    #Redirect to results page 
    return redirect(url_for('results')) 

@app.route('/results') 
def results(): 
    #Get result from session variable 
    processed_result = session.get('processed_result') 
    #Renders results page with processed result 
    return render_template('results.html', result = processed_result)

#Starts server and runs the app locally at port 5005
app.run(host = 'localhost', port = 5005, debug = True) 




