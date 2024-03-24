#Data from huggingface 
import datasets 
#Text processing 
import torchtext

#IMDb dataset --> split into trainning and test sets (25000 reviews each) 
train_Data, test_Data = datasets.load_dataset("imdb", split=["train", "test"]) 
#Debugging: Shows splits/features  
print(train_Data)
print(test_Data)
print(train_Data.features)  

#Tokenizer --> ML needs numbers so strings must be converted to tokens 
tokenizer = torchtext.data.utils.get_tokenizer("basic_english") 
def tokenize_Example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    return {"tokens": tokens} 

#Limits review length  
max_Length = 256

#Adds token as a new key in each dictionary 
train_Data = train_Data.map(tokenize_Example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_Length})
test_Data = test_Data.map(tokenize_Example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_Length}) 

#Debugging  
print(train_Data) 
print(train_Data[0]["tokens"][:25]) 

#Validation set creation 
test_Size = 0.25
train_Valid_Data = train_Data.train_test_split(test_size = test_Size)
train_Data = train_Valid_Data["train"]
valid_Data = train_Valid_Data["test"] 

#Shows sizes of each dataset 
print(len(train_Data), len(valid_Data), len(test_Data)) 

#Creates vocabulary 
min_freq = 5
special_Tokens = ["<unk>", "<pad>"]
vocab = torchtext.vocab.build_vocab_from_iterator(train_Data["tokens"], min_freq=min_freq, specials=special_Tokens)  
print(len(vocab)) 

#Debugging 
print(vocab.get_itos()[:10])
print(vocab["and"]) 

#Stores unknown and padding tokens as 0 and 1, respectively 
unk_Index = vocab["<unk>"]
pad_Index = vocab["<pad>"]  

#Specifies the unknown token to be returned for tokens not in the vocabulary 
vocab.set_default_index(unk_Index)
#print(vocab["not_in_vocab"])
 

#Function that gets the index of each token and stores it in an id field 
def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids} 

#Applies normalization function to sets  
train_Data = train_Data.map(numericalize_example, fn_kwargs={"vocab": vocab})
valid_Data = valid_Data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_Data = test_Data.map(numericalize_example, fn_kwargs={"vocab": vocab}) 

#Debugging: Normalization  
#print(train_Data[0]["tokens"][:5])
#print(vocab.lookup_indices(train_Data[0]["tokens"][:5])) 
#print(train_Data[0]["ids"][:5]) 

#Integers to tensors for PyTorch  
train_Data = train_Data.with_format(type="torch", columns=["ids", "label"])
valid_Data = valid_Data.with_format(type="torch", columns=["ids", "label"])
test_Data = test_Data.with_format(type="torch", columns=["ids", "label"])
print(train_Data[0]["label"]) 
print(train_Data[0]["ids"][:10]) 
#with_format removes all columns not specified 
print(train_Data[0].keys()) 

#Removing tokens means need to convert back to Python list to get human readable tokens 
print(vocab.lookup_tokens(train_Data[0]["ids"][:10].tolist())) 

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