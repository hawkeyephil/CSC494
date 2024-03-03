#Import Statements 
import pysentiment2 as ps 
import PreProcessor as pp

#Sentiment Dictionaries 
#Harvard IV-4 (general purpose)
Harvard_IV_4 = ps.HIV4() 
#Loughran and McDonald (finance specific) 
lm = ps.LM() 

#Debugging 
text = ''

def sentiment_analyzer(text): 
    #pysentiment2 tokenizer which is not needed with the PreProcessor class 
    #tokens = Harvard_IV_4.tokenize(text) 
    tokens = pp.pre_processor(text) 
    #Debugging 
    #print(tokens) 

    #Returns dictionary with # Positive words/# Negative words/Polarity score/Subjectivity score  
    h_score = Harvard_IV_4.get_score(tokens) 
    lm_score = lm.get_score(tokens) 
    print(lm_score)


    #Extracts polarity scores 
    h_polarity = str(h_score.get('Polarity'))
    lm_polarity = str(lm_score.get('Polarity')) 

    #print('Harvard IV-4 Score: ' + h_polarity) 
    #print('Loughran and McDonald Score: ' + lm_polarity) 

    #Returns polarity calculated using the Loughran and McDonald Dictionary 
    return float(lm_polarity) 