#Import Statements 
import pysentiment2 as ps 
import machine_learning.LexiconPreProcessor as pp

#Sentiment Dictionaries 
#Harvard IV-4 (general purpose)
Harvard_IV_4 = ps.HIV4() 
#Loughran and McDonald (finance specific) 
lm = ps.LM() 

#Debugging 
text = ''

def sentiment_analyzer(text, dictionary): 
    #pysentiment2 tokenizer which is not needed with the PreProcessor class 
    #tokens = Harvard_IV_4.tokenize(text) 
    tokens = pp.pre_processor(text) 
    #Debugging 
    #print(tokens) 

    polarity = 0
    subjectivity = 0 

    #LM Dictionary (finance)
    if(dictionary == 'LM'): 
        #Returns dictionary with # Positive words/# Negative words/Polarity score/Subjectivity score 
        lm_score = lm.get_score(tokens) 
        #Extracts polarity score
        lm_polarity = str(lm_score.get('Polarity')) 
        polarity = float(lm_polarity) 
        #Extracts subjectivity score
        lm_subjectivity = str(lm_score.get('Subjectivity'))
        subjectivity = float(lm_subjectivity) 
        #print('Finance')
    #Harvard-IV Dictionary (general)
    elif(dictionary == 'Harvard-IV'): 
        #Returns dictionary with # Positive words/# Negative words/Polarity score/Subjectivity score 
        h_score = Harvard_IV_4.get_score(tokens)
        #Extracts polarity score
        h_polarity = str(h_score.get('Polarity')) 
        polarity = float(h_polarity) 
        #Extracts subjectivity score 
        h_subjectivity = str(h_score.get('Subjectivity'))
        subjectivity = float(h_subjectivity) 
        #print('General')
    else: 
        print("No Dictionary Selected")

    sentiment_Scores = {'polarity': polarity, 'subjectivity': subjectivity}
    #Returns polarity calculated using the Loughran and McDonald Dictionary 
    return sentiment_Scores 
 