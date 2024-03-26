#Hugging Face library
import datasets 
#Text processing 
import torchtext 

#Financial Phrasebook dataset from Hugging Face 
#finance_Data = datasets.load_dataset("financial_phrasebank", "sentences_50agree", split=["train"]) 
#finance_Data = finance_Data[0]

#Stock Tweets Sentiment dataset from Hugging Face 
train_Data, test_Data = datasets.load_dataset("emad12/stock_tweets_sentiment", split=["train", "test"]) 

#Tokenizes string and trims to specified length 
def tokenize_Sentiment(example):
    #tokens = tokenizer(example["sentence"])[:max_length]
    sentiment = example["sentiment"] 
    if sentiment == -1: 
        return {"label": 0}
    elif sentiment == 0:
        return {"label": 1}
    else: 
        return {"label": 2}
#Tokenize each example and add it to the dictionary 
train_Data = train_Data.map(tokenize_Sentiment)
test_Data = test_Data.map(tokenize_Sentiment)

#Debugging 
#print(finance_Data) 
#print(finance_Data.features) 
#print(finance_Data[0]) 

#Splits the whole dataset into training and testing data 
#train_Test_Split = finance_Data.train_test_split(test_size = 0.50)
#train_Data = train_Test_Split["train"]
#test_Data = train_Test_Split["test"] 

print(len(train_Data), len(test_Data))

tokenizer = torchtext.data.utils.get_tokenizer("basic_english") 

#Tokenizes string and trims to specified length 
def tokenize_Example(example, tokenizer, max_length):
    #tokens = tokenizer(example["sentence"])[:max_length]
    tokens = tokenizer(example["tweet"])[:max_length]
    return {"tokens": tokens} 

#Longest string is 315 tokens 
max_Length = 512 

#Tokenize each example and add it to the dictionary 
train_Data = train_Data.map(tokenize_Example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_Length})
test_Data = test_Data.map(tokenize_Example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_Length}) 

#Splits the train dataset into a train and validation dataset 
train_Valid_Data = train_Data.train_test_split(test_size = 0.25)
train_Data = train_Valid_Data["train"]
valid_Data = train_Valid_Data["test"]
print(len(train_Data), len(valid_Data), len(test_Data))

#Any tokens appearing less than 5 times are replaced with <unk> 
min_Frequency = 5 
#Pads sentences with <pad> to make them all the same length 
special_Tokens = ["<unk>", "<pad>"] 
#Creates vocabulary from training set where every unique token has a corresponding index 
#Each index is used to create a one hot vector (where all elements are 0 except one) and the dimensionality equals number of unique tokens 
vocab = torchtext.vocab.build_vocab_from_iterator(train_Data["tokens"], min_freq = min_Frequency, specials = special_Tokens) 
print(len(vocab)) 

unk_Index = vocab["<unk>"]
pad_Index = vocab["<pad>"]
#Unknown tokens pass the <unk> token instead of an error 
vocab.set_default_index(unk_Index) 

#Function that sets the ids field to the index of each token in an example 
def numericalize_Example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids} 

#Normalizes all data 
train_Data = train_Data.map(numericalize_Example, fn_kwargs={"vocab": vocab})
valid_Data = valid_Data.map(numericalize_Example, fn_kwargs={"vocab": vocab})
test_Data = test_Data.map(numericalize_Example, fn_kwargs={"vocab": vocab}) 

#Converts ids and labels fields from integers to tensors 
train_Data = train_Data.with_format(type="torch", columns=["ids", "label"])
valid_Data = valid_Data.with_format(type="torch", columns=["ids", "label"])
test_Data = test_Data.with_format(type="torch", columns=["ids", "label"])

output_Dim = len(train_Data.unique("label")) 

#Get functions 
def get_Train_Data():
    return train_Data 

def get_Valid_Data():
    return valid_Data 

def get_Test_Data(): 
    return test_Data

def get_Output_Dim(): 
    return output_Dim 

def get_Pad_Index(): 
    return pad_Index 

def get_Vocab():
    return vocab