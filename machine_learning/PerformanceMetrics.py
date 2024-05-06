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
#metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 
        #'AA': [], 'AB': [], 'AC': [], 
        #'BA': [], 'BB': [], 'BC': [], 
        #'CA': [], 'CB': [], 'CC': []
        #}
df_Metrics = pd.DataFrame(metrics)

#Accuracy 
#Number Corrrect Predictions/Total Number of Predictions 
def binary_Accuracy_Calculator(true_Positives, true_Negatives, total_Predictions): 
    correct_Predictions = true_Positives + true_Negatives 
    accuracy = correct_Predictions/total_Predictions 
    print('Accuracy: ' + str(accuracy)) 
    return accuracy

def nonbinary_Accuracy_Calculator(class_AA_True_Positives, class_BB_True_Positives, class_CC_True_Positives, total_Predictions): 
    correct_Predictions = class_AA_True_Positives + class_BB_True_Positives + class_CC_True_Positives 
    accuracy = correct_Predictions/total_Predictions 
    print('Accuracy: ' + str(accuracy)) 
    return accuracy 

#Precision 
#Number of the positive predictions made which are correct (true positives/(true positives + false positives)) 
def binary_Precision_Calculator(true_Positives, false_Positives): 
    precision = true_Positives/(true_Positives + false_Positives)
    print('Precision: ' + str(precision)) 
    return precision 

def nonbinary_Precision_Calculator(class_AA_True_Positives, class_AB_False_Negatives, class_AC_False_Negatives, 
        class_BA_False_Negatives, class_BB_True_Positives, class_BC_False_Negatives, 
        class_CA_False_Negatives, class_CB_False_Negatives, class_CC_True_Positives,
        ): 
    precision_A = class_AA_True_Positives/(class_AA_True_Positives + class_AB_False_Negatives + class_AC_False_Negatives) 
    precision_B = class_BB_True_Positives/(class_BA_False_Negatives + class_BB_True_Positives + class_BC_False_Negatives)
    precision_C = class_CC_True_Positives/(class_CA_False_Negatives + class_CB_False_Negatives + class_CC_True_Positives)
    precision = (precision_A + precision_B + precision_C)/3
    print('Precision: ' + str(precision)) 
    return precision 

#Recall/Sensitivity 
#Number of the positive cases the  correctly predicted over all the positive cases in the data (true positives/(true positives + false negatives)) 
def binary_Recall_Calculator(true_Positives, false_Negatives): 
    recall = true_Positives/(true_Positives + false_Negatives) 
    print('Recall/Sensitivity: ' + str(recall)) 
    return recall 

def nonbinary_Recall_Calculator(class_AA_True_Positives, class_AB_False_Negatives, class_AC_False_Negatives, 
        class_BA_False_Negatives, class_BB_True_Positives, class_BC_False_Negatives, 
        class_CA_False_Negatives, class_CB_False_Negatives, class_CC_True_Positives,
        ): 
    recall_A = class_AA_True_Positives/(class_AA_True_Positives + class_BA_False_Negatives + class_CA_False_Negatives) 
    recall_B = class_BB_True_Positives/(class_AB_False_Negatives + class_BB_True_Positives + class_CB_False_Negatives)
    recall_C = class_CC_True_Positives/(class_AC_False_Negatives + class_BC_False_Negatives + class_CC_True_Positives)
    recall = (recall_A + recall_B + recall_C)/3
    print('Recall: ' + str(recall)) 
    return recall 

#F1-Score 
#Combination of both precision and recall (harmonic mean of the two) 
#Harmonic mean is an average of  more suitable for ratios (precision and recall) than the traditional arithmetic mean 
#2 * ((precision * recall)/(precision + recall)) 
def f1_Score_Calculator(precision, recall): 
    f1_Score = 2*((precision * recall)/(precision + recall)) 
    print('F1-Score: ' + str(f1_Score)) 
    return f1_Score

#Iterates through every tweet and makes a sentiment inference for each model (this should be a function) 
def binary_Sentiment_Predictor(model, application, data): 
    total_Predictions = data.__len__() 
    true_Positives = 0 
    false_Positives = 0 
    true_Negatives = 0 
    false_Negatives = 0 

    for index, row in data.iterrows(): 
        text = row['text'] 
        sentiment = row['sentiment'] 
        if(application == 'Harvard-IV'): 
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
            
        elif(application == 'General'): 
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
    accuracy = binary_Accuracy_Calculator(true_Positives, true_Negatives, total_Predictions) 
    precision = binary_Precision_Calculator(true_Positives, false_Positives)
    recall = binary_Recall_Calculator(true_Positives, false_Negatives)
    f1_Score = f1_Score_Calculator(precision, recall) 

    #Debugging 
    #print('True Positives: ' + str(true_Positives)) 
    #print('False Positives: ' + str(false_Positives))
    #print('True Negatives: ' + str(true_Negatives))
    #print('False Negatives: ' + str(false_Negatives)) 

    model_Metrics = {'Model': model, 'Accuracy': accuracy, 'Precision': precision, 
        'Recall': recall, 'F1-Score': f1_Score, 'True Positives': true_Positives, 'False Positives': false_Positives, 
        'True Negatives': true_Negatives, 'False Negatives': false_Negatives
        } 
    #print(model_Metrics)
    return model_Metrics 

#Iterates through every tweet and makes a sentiment inference for each model  
def nonbinary_Sentiment_Predictor(model, application, data): 
    total_Predictions = data.__len__() 
    class_AA_True_Positives = 0 
    class_AB_False_Negatives = 0 
    class_AC_False_Negatives = 0 

    class_BA_False_Negatives = 0 
    class_BB_True_Positives = 0 
    class_BC_False_Negatives = 0

    class_CA_False_Negatives = 0 
    class_CB_False_Negatives = 0 
    class_CC_True_Positives = 0 

    for index, row in data.iterrows(): 
        text = row['tweets'] 
        sentiment = row['sentiment'] 
        #Lexicon Approach 
        if(application == 'LM'): 
            prediction = sa.sentiment_analyzer(text, application).get('polarity') 
            if float(prediction) > 0.25:
                lexicon_Class = 1 
                if(sentiment == lexicon_Class): 
                    class_AA_True_Positives = class_AA_True_Positives + 1 
                elif(sentiment == 0): 
                    class_AB_False_Negatives = class_AB_False_Negatives + 1
                else: 
                    class_AC_False_Negatives = class_AC_False_Negatives + 1
            elif float(prediction) < -0.25:
                lexicon_Class = -1 
                if(sentiment == lexicon_Class): 
                    class_CC_True_Positives = class_CC_True_Positives + 1
                elif(sentiment == 0): 
                    class_CB_False_Negatives = class_CB_False_Negatives + 1 
                else: 
                    class_CA_False_Negatives = class_CA_False_Negatives + 1
            else: 
                lexicon_Class = 0
                if(sentiment == lexicon_Class): 
                    class_BB_True_Positives = class_BB_True_Positives + 1
                elif(sentiment == 1): 
                    class_BA_False_Negatives = class_BA_False_Negatives + 1 
                else: 
                    class_BC_False_Negatives = class_BC_False_Negatives + 1 
        #NBoW Approach 
        elif(application == 'Finance'): 
            if(model == 'NBoW'):
                prediction = nbow.predict_Sentiment(text, application)[0] 
                if prediction == 0: 
                    nbow_Class = -1 
                    if(sentiment == nbow_Class): 
                        class_AA_True_Positives = class_AA_True_Positives + 1 
                    elif(sentiment == 0): 
                        class_AB_False_Negatives = class_AB_False_Negatives + 1
                    else: 
                        class_AC_False_Negatives = class_AC_False_Negatives + 1
                elif prediction == 2:
                    nbow_Class = 1
                    if(sentiment == nbow_Class): 
                        class_CC_True_Positives = class_CC_True_Positives + 1
                    elif(sentiment == 0): 
                        class_CB_False_Negatives = class_CB_False_Negatives + 1 
                    else: 
                        class_CA_False_Negatives = class_CA_False_Negatives + 1
                else: 
                    nbow_Class = 0 
                    if(sentiment == nbow_Class): 
                        class_BB_True_Positives = class_BB_True_Positives + 1
                    elif(sentiment == 1): 
                        class_BA_False_Negatives = class_BA_False_Negatives + 1 
                    else: 
                        class_BC_False_Negatives = class_BC_False_Negatives + 1
            #CNN Approach 
            elif(model == 'CNN'):
                prediction = cnn.predict_Sentiment(text, application)[0] 
                if prediction == 2: 
                    cnn_Class = 1 
                    if(sentiment == cnn_Class): 
                        class_AA_True_Positives = class_AA_True_Positives + 1 
                    elif(sentiment == 0): 
                        class_AB_False_Negatives = class_AB_False_Negatives + 1
                    else: 
                        class_AC_False_Negatives = class_AC_False_Negatives + 1
                elif prediction == 0: 
                    cnn_Class = -1 
                    if(sentiment == cnn_Class): 
                        class_CC_True_Positives = class_CC_True_Positives + 1
                    elif(sentiment == 0): 
                        class_CB_False_Negatives = class_CB_False_Negatives + 1 
                    else: 
                        class_CA_False_Negatives = class_CA_False_Negatives + 1
                else: 
                    cnn_Class = 0 
                    if(sentiment == cnn_Class): 
                        class_BB_True_Positives = class_BB_True_Positives + 1
                    elif(sentiment == 1): 
                        class_BA_False_Negatives = class_BA_False_Negatives + 1 
                    else: 
                        class_BC_False_Negatives = class_BC_False_Negatives + 1
            else: 
                print('Invalid Model')

        else: 
            print('Invalid Application') 

    #Calls performance metrics calculator functions 
    accuracy = nonbinary_Accuracy_Calculator(class_AA_True_Positives, class_BB_True_Positives, class_CC_True_Positives, total_Predictions) 
    precision = nonbinary_Precision_Calculator(class_AA_True_Positives, class_AB_False_Negatives, class_AC_False_Negatives, 
        class_BA_False_Negatives, class_BB_True_Positives, class_BC_False_Negatives, 
        class_CA_False_Negatives, class_CB_False_Negatives, class_CC_True_Positives,
        )
    recall = nonbinary_Recall_Calculator(class_AA_True_Positives, class_AB_False_Negatives, class_AC_False_Negatives, 
        class_BA_False_Negatives, class_BB_True_Positives, class_BC_False_Negatives, 
        class_CA_False_Negatives, class_CB_False_Negatives, class_CC_True_Positives,
        )
    f1_Score = f1_Score_Calculator(precision, recall) 

    model_Metrics = {'Model': model, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1_Score, 
        'AA': class_AA_True_Positives, 'AB': class_AB_False_Negatives, 'AC': class_AC_False_Negatives, 
        'BA': class_BA_False_Negatives, 'BB': class_BB_True_Positives, 'BC': class_BC_False_Negatives, 
        'CA': class_CA_False_Negatives, 'CB': class_AB_False_Negatives, 'CC': class_CC_True_Positives
        } 
    print(model_Metrics)

    return model_Metrics

#Generates performance metrics for each model 
df_Metrics = df_Metrics.append(binary_Sentiment_Predictor('Lexicon', 'Harvard-IV', sorted_Validation_Data), ignore_index = 'True') 
df_Metrics = df_Metrics.append(binary_Sentiment_Predictor('NBoW', 'General', sorted_Validation_Data), ignore_index = 'True') 
df_Metrics = df_Metrics.append(binary_Sentiment_Predictor('CNN', 'General', sorted_Validation_Data), ignore_index = 'True') 

#df_Metrics = df_Metrics.append(nonbinary_Sentiment_Predictor('Lexicon', 'LM', sorted_Validation_Data), ignore_index = 'True') 
#df_Metrics = df_Metrics.append(nonbinary_Sentiment_Predictor('NBoW', 'Finance', sorted_Validation_Data), ignore_index = 'True') 
#df_Metrics = df_Metrics.append(nonbinary_Sentiment_Predictor('CNN', 'Finance', sorted_Validation_Data), ignore_index = 'True') 

#Saves model performance metrics to a csv file 
filename = '~/Downloads/ModelMetrics.csv' 
df_Metrics.to_csv(filename, index = False, header = True) 