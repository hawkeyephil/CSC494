#import nltk 
#nltk.download('all') 

#Import Statements 
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 

#Sample Text 
text = "Recent indicators suggest that economic activity expanded at a strong pace in the third quarter. Job gains have moderated since earlier in the year but remain strong, and the unemployment rate has remained low. Inflation remains elevated. The U.S. banking system is sound and resilient. Tighter financial and credit conditions for households and businesses are likely to weigh on economic activity, hiring, and inflation. The extent of these effects remains uncertain. The Committee remains highly attentive to inflation risks. The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. In support of these goals, the Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. The Committee will continue to assess additional information and its implications for monetary policy. In determining the extent of additional policy firming that may be appropriate to return inflation to 2 percent over time, the Committee will take into account the cumulative tightening of monetary policy, the lags with which monetary policy affects economic activity and inflation, and economic and financial developments. In addition, the Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities, as described in its previously announced plans. The Committee is strongly committed to returning inflation to its 2 percent objective. In assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook. The Committee would be prepared to adjust the stance of monetary policy as appropriate if risks emerge that could impede the attainment of the Committee's goals. The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments."
test = "Natural language processing (NLP) is a field in CS. It can be quite challenging at times. Test 12"

def pre_processor(text): 
    #Pre-Processing Pipeline 
    #Changes the case of all characters to lowr 
    text = text.lower()

    #Tokenization/Special character removal 
    tokenizer = RegexpTokenizer(r'\w+') 
    tokens = tokenizer.tokenize(text) 
    #print(tokens) 

    #Number remover 
    result = [word for word in tokens if not word.isnumeric()]              
    #print(result)

    #Stopword remover 
    english_stopwords = set(stopwords.words('English')) 
    result = [word for word in result if word not in english_stopwords] 
    #print(result) 

    #Porter Stemming Algorithm 
    porter_Stemmer = PorterStemmer() 
    result = [porter_Stemmer.stem(word) for word in result] 
    #print(result) 

    return result
    #print(sent_tokenize(text)) 
    #print(word_tokenize(text)) 