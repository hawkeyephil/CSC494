import torch
import torch.nn as nn
import torchtext  

#Model declaration 
class NBoW(nn.Module):
    def __init__(self, vocab_Size, embedding_Dim, output_Dim, pad_Index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_Size, embedding_Dim, padding_idx = pad_Index)
        self.fc = nn.Linear(embedding_Dim, output_Dim)

    def forward(self, ids):
        embedded = self.embedding(ids)
        pooled = embedded.mean(dim=1)
        prediction = self.fc(pooled)
        return prediction

def load_NBoW(vocab_Size, embedding_Dim, output_Dim, pad_Index, application): 
    #Initializes instance of the model 
    model = NBoW(vocab_Size, embedding_Dim, output_Dim, pad_Index) 
    #Loads the state dictionary from memory 
    model.load_state_dict(torch.load(application)) 
    #Sets the model to evaluation mode 
    model.eval()  
    return model

#Loads the nbow model parameters  
general_Vocab = torch.load("NBoW_cache/vocab.pt") 
general_Vocab_Size = len(general_Vocab) 

general_Embedding_Dim = torch.load("NBoW_cache/embedding_Dim.pt")
general_Output_Dim = torch.load("NBoW_cache/output_Dim.pt")
general_Pad_Index = torch.load("NBoW_cache/pad_Index.pt")

#Creates an instance of the nbow model 
general = 'NBoW_cache/nbow.pt'
nbow_General = load_NBoW(general_Vocab_Size, general_Embedding_Dim, general_Output_Dim, general_Pad_Index, general)


#Loads the finance nbow model parameters  
finance_Vocab = torch.load("finance_NBoW_cache/vocab.pt") 
finance_Vocab_Size = len(finance_Vocab) 

finance_Embedding_Dim = torch.load("finance_NBoW_cache/embedding_Dim.pt")
finance_Output_Dim = torch.load("finance_NBoW_cache/output_Dim.pt")
finance_Pad_Index = torch.load("finance_NBoW_cache/pad_Index.pt")

#Creates an instance of the finance nbow model 
finance = 'finance_NBoW_cache/nbow.pt' 
nbow_Finance = load_NBoW(finance_Vocab_Size, finance_Embedding_Dim, finance_Output_Dim, finance_Pad_Index, finance)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Creates tokenizer instance 
tokenizer = torchtext.data.utils.get_tokenizer("basic_english") 

#Function that returns sentiment prediction   
def predict_Sentiment(text, application):
    tokens = tokenizer(text)
    if application == 'Finance': 
        ids = finance_Vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
        prediction = nbow_Finance(tensor).squeeze(dim=0) 
    else: 
        ids = general_Vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
        prediction = nbow_General(tensor).squeeze(dim=0) 
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item() 
    return predicted_class, predicted_probability 

#Debugging 
#text = 'This film is great!' 
#print(predict_Sentiment(text, 'General')) 

