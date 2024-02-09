#import nltk 
#nltk.download('all') 

#Import Statements 
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 

#Sample Text 
text = "Natural language processing (NLP) is a field in CS. It can be quite challenging at times. Test 12" 

#Pre-Processing Pipeline 
#Changes the case of all characters to lowr 
text = text.lower()

#Tokenization/Special character removal 
tokenizer = RegexpTokenizer(r'\w+') 
tokens = tokenizer.tokenize(text) 
print(tokens) 

#Number remover 
result = [word for word in tokens if not word.isnumeric()]              
print(result)

#Stopword remover 
english_stopwords = set(stopwords.words('English')) 
result = [word for word in result if word not in english_stopwords] 
print(result) 

#Porter Stemming Algorithm 
porter_Stemmer = PorterStemmer() 
result = [porter_Stemmer.stem(word) for word in result] 
print(result) 

#print(sent_tokenize(text)) 
#print(word_tokenize(text)) 