from flask import Flask 

#Gives each file a unique name 
app = Flask(__name__) 

#Decorator which specifies url endpoint 
@app.route('/hello') 
#Function defines what is shown 
def hello(): 
    return 'Hello World!' 

#Starts server and runs the app locally at port 5005
app.run(host='localhost', port=5005, debug=True) 




