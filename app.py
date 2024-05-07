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
    session.setdefault('lexicon_Polarity', 0.0)    
    session.setdefault('nbow_Class', 'Neutral') 
    session.setdefault('nbow_Probability', 0.0)
    session.setdefault('cnn_Class', 'Neutral') 
    session.setdefault('cnn_Probability', 0.0)
    return render_template('index.html') 

#Route to the sandbox page 
@app.route('/sandbox') 
def sandbox(): 
    #Establishes sentiment score variable for the session 
    #session['lexicon_Polarity'] = 0.0
    return render_template('sandbox.html')

#Sandbox sentiment route (this can be phased out now with the js script in place)
@app.route('/analyze_sentiment', methods=['GET', 'POST'])    
def analyze_sentiment(): 
    application = request.form.get('application_Select')
    #model = request.form.get('model_Select') 
    lexicon_Class = 'Neutral'
    lexicon_Polarity = 0.00 
    nbow_Class = 'Neutral' 
    nbow_Probability = 0.00 
    cnn_Class = 'Neutral' 
    cnn_Probability = 0.00
    if request.method == 'POST':
        #Fetches the user input from the textarea
        user_input = request.form['user_text'] 
        #print(user_input) 

        if not user_input: 
            print("No User Input") 
        else: 
            if application == 'Finance':      
                #Analyze the sentiment using the dictionary approach 
                lexicon_Scores = dict.sentiment_analyzer(user_input, 'LM') 
                #Extract the polarity and subjectivity scores 
                lexicon_Polarity = format(lexicon_Scores['polarity'], '.2f') 
                #subjectivity = format(lexicon_Scores['subjectivity'], '.2f')
                if float(lexicon_Polarity) > 0.00:
                    lexicon_Class = 'Positive' 
                elif float(lexicon_Polarity) < 0.00: 
                    lexicon_Class = 'Negative' 
                else: 
                    lexicon_Class = 'Neutral'

                #Extract the sentiment using the NBoW model 
                nbow_Scores = nbow.predict_Sentiment(user_input, application) 
                print('NBoW Model Selected')
                print(nbow_Scores)
                if nbow_Scores[0] == 2: 
                    nbow_Class = 'Positive' 
                elif nbow_Scores[0] == 0: 
                    nbow_Class = 'Negative' 
                else: 
                    nbow_Class = 'Neutral'
                nbow_Probability = nbow_Scores[1]
                            
                #Extract the sentiment using the CNN model 
                cnn_Scores = cnn.predict_Sentiment(user_input, application)
                print('CNN Model Selected') 
                print(cnn_Scores)
                if cnn_Scores[0] == 2: 
                    cnn_Class = 'Positive' 
                elif cnn_Scores[0] == 0: 
                    cnn_Class = 'Negative' 
                else: 
                    cnn_Class = 'Neutral'
                cnn_Probability = cnn_Scores[1]
            elif application == 'General':
                #Analyze the sentiment using the dictionary approach 
                lexicon_Scores = dict.sentiment_analyzer(user_input, 'Harvard-IV') 
                #Extract the polarity and subjectivity scores 
                lexicon_Polarity = format(lexicon_Scores['polarity'], '.2f') 
                print(lexicon_Polarity)
                #subjectivity = format(lexicon_Scores['subjectivity'], '.2f')
                if float(lexicon_Polarity) > 0.00:
                    lexicon_Class = 'Positive' 
                elif float(lexicon_Polarity) < 0.00: 
                    lexicon_Class = 'Negative' 
                else: 
                    lexicon_Class = 'Neutral'

                #Extract the sentiment using the NBoW model 
                nbow_Scores = nbow.predict_Sentiment(user_input, application) 
                print('NBoW Model Selected') 
                print(nbow_Scores)
                if nbow_Scores[0] == 1: 
                    nbow_Class = 'Positive' 
                elif nbow_Scores[0] == 0: 
                    nbow_Class = 'Negative' 
                else: 
                    nbow_Class = 'Neutral' 
                print(nbow_Class)
                nbow_Probability = nbow_Scores[1]
                print(nbow_Probability) 

                #Extract the sentiment using the CNN model 
                cnn_Scores = cnn.predict_Sentiment(user_input, application)
                print('CNN Model Selected')
                print(cnn_Scores)
                if cnn_Scores[0] == 1: 
                    cnn_Class = 'Positive' 
                elif cnn_Scores[0] == 0: 
                    cnn_Class = 'Negative' 
                else: 
                    cnn_Class = 'Neutral' 
                print(cnn_Class)
                cnn_Probability = cnn_Scores[1]
                print(cnn_Probability)
            else:
                print('Invalid Application Selected')  
                 
        #Store the lexicon polarity, nbow probability, and the cnn probability score in the session 
        session['lexicon_Polarity'] = lexicon_Polarity 
        session['nbow_Class'] = nbow_Class
        session['nbow_Probability'] = nbow_Probability 
        session['cnn_Class'] = cnn_Class
        session['cnn_Probability'] = cnn_Probability 
    #Display the results to the page 
    return render_template('sandbox.html', lexicon_Class=lexicon_Class, nbow_Class=nbow_Class, cnn_Class=cnn_Class)

#Returns the polarity score to the frontend 
@app.route('/polarity_score', methods=['POST'])
def polarity_score():
    if request.method == 'POST': 
        #print(session["polarity"])
        lexicon_Polarity = session["lexicon_Polarity"] 
        nbow_Class = session["nbow_Class"] 
        nbow_Probability = session["nbow_Probability"] 
        cnn_Class = session["cnn_Class"] 
        cnn_Probability = session["cnn_Probability"] 
        #Return the polarity as JSON to frontend 
        return jsonify({'lexicon_Polarity': lexicon_Polarity, 'nbow_Class': nbow_Class, 'nbow_Probability': nbow_Probability, 'cnn_Class': cnn_Class, 'cnn_Probability': cnn_Probability})

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        comments = request.form.get('comments')

        # Process the data (e.g., save to a database, send an email, etc.)
        print(email)
        # Return an appropriate response (e.g., a thank-you message)
        return render_template('contact.html')

#Starts server and runs the app locally at port 5005
#app.run(host = 'localhost', port = 5005, debug = True) 
#Starts server and runs the app locally at port 5005 for LAN access  
app.run(host = '0.0.0.0', port = 5005, debug = True) 





