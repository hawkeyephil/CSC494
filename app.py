from flask import Flask, render_template, request, redirect, url_for

#Gives each file a unique name 
app = Flask(__name__) 

#Decorator which specifies url endpoint 
@app.route('/') 
#Function defines what is shown 
def hello(): 
    return render_template('index.html') 

@app.route('/process_text', methods=['POST']) 
def process_text(): 
    #User text entered into form 
    user_text = request.form.get('user_text') 
    #Process text to be sent back to user 
    processed_result = 'Result' 
    #Redirect to results page 
    return redirect(url_for('results', result = processed_result)) 

@app.route('/results') 
def results(): 
    processed_result = request.args.get('result') 
    return render_template('results.html', result = processed_result)

#Starts server and runs the app locally at port 5005
app.run(host = 'localhost', port = 5005, debug = True) 




