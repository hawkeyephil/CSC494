#Import Statements
from flask import Flask, jsonify, render_template, request, redirect, url_for, session 
from dotenv import load_dotenv 
import os
import machine_learning.LexiconInference as dict 
import machine_learning.NBoWInference as nbow 
import machine_learning.CNNInference as cnn

#Gives each file a unique name 
app = Flask(__name__) 
#Sets secret key
app.secret_key = os.getenv('SECRET_KEY') 

#Decorator which specifies url endpoint 
@app.route('/') 
#Function that defines that the index page is displayed 
def index(): 
    session.setdefault('polarity', 0.0)
    return render_template('index.html') 

#Route to the sandbox page 
@app.route('/sandbox') 
def sandbox(): 
    #Establishes sentiment score variable for the session 
    session['polarity'] = 0.0
    return render_template('sandbox.html')

#Sandbox sentiment route (this can be phased out now with the js script in place)
@app.route('/analyze_sentiment', methods=['GET', 'POST'])
def analyze_sentiment(): 
    application = request.form.get('application_Select')
    #model = request.form.get('model_Select') 
    if request.method == 'POST':
        #Fetches the user input from the textarea
        user_input = request.form['user_text'] 
        #print(user_input)

        if application == 'Finance':      
            #Analyze the sentiment using the dictionary approach 
            sentiment_Scores = dict.sentiment_analyzer(user_input, 'LM') 
            #Extract the polarity and subjectivity scores 
            polarity = format(sentiment_Scores['polarity'], '.2f')
            subjectivity = format(sentiment_Scores['subjectivity'], '.2f') 
        elif application == 'General': 
            #Analyze the sentiment using the dictionary approach 
            sentiment_Scores = dict.sentiment_analyzer(user_input, 'Harvard-IV') 
            #Extract the polarity and subjectivity scores 
            polarity = format(sentiment_Scores['polarity'], '.2f')
            subjectivity = format(sentiment_Scores['subjectivity'], '.2f') 
            
            #Extract the sentiment using the NBoW model 
            sentiment_Scores = nbow.predict_Sentiment(user_input) 
            print('NBoW Model Selected')
            polarity = sentiment_Scores[0]
            subjectivity = sentiment_Scores[1]
            
            #Extract the sentiment using the CNN model 
            sentiment_Scores = cnn.predict_Sentiment(user_input)
            print('CNN Model Selected')
            polarity = sentiment_Scores[0]
            subjectivity = 0.00
        else:
            print('Invalid Application Selected') 
            polarity = 0.00
            subjectivity = 0.00 
    #Store the polarity score in the session 
    session['polarity'] = polarity 
    #Display the results to the page 
    return render_template('sandbox.html', polarity=polarity, subjectivity=subjectivity)

#Returns the polarity score to the frontend 
@app.route('/polarity_score', methods=['POST'])
def polarity_score():
    if request.method == 'POST': 
        #print(session["polarity"])
        polarity = session["polarity"]
        #Return the polarity as JSON to frontend 
        return jsonify({'polarity': polarity})

#Starts server and runs the app locally at port 5005
app.run(host = 'localhost', port = 5005, debug = True) 




