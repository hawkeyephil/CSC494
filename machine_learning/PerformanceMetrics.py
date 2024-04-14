#Import Statements
import datasets 
import pandas as pd
import LexiconInference as sa 
import NBoWInference as nbow 
import CNNInference as cnn 

#Stock Tweets Sentiment dataset from Hugging Face 
#train_Data, test_Data = datasets.load_dataset("emad12/stock_tweets_sentiment", split = ["train", "test"]) 
#IMDb dataset --> split into trainning and test sets (25000 reviews each) 
train_Data, test_Data = datasets.load_dataset("imdb", split = ["train", "test"])

#Creates a subset of the training data for validation 
train_Valid_Data = train_Data.train_test_split(test_size = 0.25) 
train_Data = train_Valid_Data["train"]
valid_Data = train_Valid_Data["test"]
print(len(train_Data), len(valid_Data), len(test_Data))
print(valid_Data)
#Stores and sorts tweets with corresponding sentiment scores in ascending order 
#validation_Data = pd.DataFrame({'tweets': valid_Data["tweet"], 'sentiment': valid_Data["sentiment"]}) 
validation_Data = pd.DataFrame({'text': valid_Data["text"], 'sentiment': valid_Data["label"]})
sorted_Validation_Data = validation_Data.sort_values(by = ['sentiment'])  

#Dataframe to store sentiment scores 
metrics = {'Model': [], 'True Positives': [], 'False Positives': [], 'True Negatives': [], 'False Negatives': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
df_Metrics = pd.DataFrame(metrics)

#Accuracy 
#Number Corrrect Predictions/Total Number of Predictions 
def accuracy_Calculator(true_Positives, true_Negatives, total_Predictions): 
    correct_Predictions = true_Positives + true_Negatives 
    accuracy = correct_Predictions/total_Predictions 
    print('Accuracy: ' + str(accuracy)) 
    return accuracy

#Precision 
#Number of the positive predictions made which are correct (true positives/(true positives + false positives)) 
def precision_Calculator(true_Positives, false_Positives): 
    precision = true_Positives/(true_Positives + false_Positives)
    print('Precision: ' + str(precision)) 
    return precision 

#Recall/Sensitivity 
#Number of the positive cases the  correctly predicted over all the positive cases in the data (true positives/(true positives + false negatives)) 
def recall_Calculator(true_Positives, false_Negatives): 
    recall = true_Positives/(true_Positives + false_Negatives) 
    print('Recall/Sensitivity: ' + str(recall)) 
    return recall 

#F1-Score 
#Combination of both precision and recall (harmonic mean of the two) 
#Harmonic mean is an average of  more suitable for ratios (precision and recall) than the traditional arithmetic mean 
#2 * ((precision * recall)/(precision + recall)) 
def f1_Score_Calculator(precision, recall): 
    f1_Score = 2*((precision * recall)/(precision + recall)) 
    print('F1-Score: ' + str(f1_Score)) 
    return recall

#Iterates through every tweet and makes a sentiment inference for each model (this should be a function) 
def sentiment_Predictor(model, application, data): 
    total_Predictions = data.__len__() 
    true_Positives = 0 
    false_Positives = 0 
    true_Negatives = 0 
    false_Negatives = 0 

    for index, row in data.iterrows(): 
        text = row['text'] 
        sentiment = row['sentiment'] 
        if(application == 'LM' or application == 'Harvard-IV'): 
            prediction = sa.sentiment_analyzer(text, application).get('polarity') 
            if float(prediction) > 0.00:
                lexicon_Class = 1 
                if(lexicon_Class == sentiment): 
                    true_Positives = true_Positives + 1
                else: 
                    false_Positives = false_Positives + 1
            else:
                lexicon_Class = 0 
                if(lexicon_Class == sentiment): 
                    true_Negatives = true_Negatives + 1
                else: 
                    false_Negatives = false_Negatives + 1
        elif(application == 'Finance' or application == 'General'): 
            if(model == 'NBoW'):
                prediction = nbow.predict_Sentiment(text, application)[0] 
                if prediction == 1: 
                    nbow_Class = 1 
                    if(nbow_Class == sentiment): 
                        true_Positives = true_Positives + 1
                    else: 
                        false_Positives = false_Positives + 1
                else: 
                    nbow_Class = 0 
                    if(nbow_Class == sentiment): 
                        true_Negatives = true_Negatives + 1
                    else: 
                        false_Negatives = false_Negatives + 1
            elif(model == 'CNN'):
                prediction = cnn.predict_Sentiment(text, application)[0] 
                if prediction == 1: 
                    cnn_Class = 1 
                    if(cnn_Class == sentiment): 
                        true_Positives = true_Positives + 1
                    else: 
                        false_Positives = false_Positives + 1
                else: 
                    cnn_Class = 0 
                    if(cnn_Class == sentiment): 
                        true_Negatives = true_Negatives + 1
                    else: 
                        false_Negatives = false_Negatives + 1
            else: 
                print('Invalid Model')
        else: 
            print('Invalid Application') 

    #Calls performance metrics calculator functions 
    accuracy = accuracy_Calculator(true_Positives, true_Negatives, total_Predictions) 
    precision = precision_Calculator(true_Positives, false_Positives)
    recall = recall_Calculator(true_Positives, false_Negatives)
    f1_Score = f1_Score_Calculator(precision, recall) 

    #Debugging 
    #print('True Positives: ' + str(true_Positives)) 
    #print('False Positives: ' + str(false_Positives))
    #print('True Negatives: ' + str(true_Negatives))
    #print('False Negatives: ' + str(false_Negatives)) 

    model_Metrics = {'Model': model, 'True Positives': true_Positives, 'False Positives': false_Positives, 'True Negatives': true_Negatives, 'False Negatives': false_Negatives, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1_Score} 
    #print(model_Metrics)
    return model_Metrics 

#Generates performance metrics for each model 
df_Metrics = df_Metrics.append(sentiment_Predictor('Lexicon', 'Harvard-IV', sorted_Validation_Data), ignore_index = 'True') 
df_Metrics = df_Metrics.append(sentiment_Predictor('NBoW', 'General', sorted_Validation_Data), ignore_index = 'True') 
df_Metrics = df_Metrics.append(sentiment_Predictor('CNN', 'General', sorted_Validation_Data), ignore_index = 'True') 

#Saves model performance metrics to a csv file 
filename = '~/Downloads/ModelMetrics.csv' 
df_Metrics.to_csv(filename, index = False, header = True) 