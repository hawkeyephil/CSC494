import torch
import torch.nn as nn
import torchtext  

#Loads model parameters  
vocab = torch.load("vocab.pt") 
vocab_Size = len(vocab) 

embedding_Dim = torch.load("embedding_Dim.pt")
output_Dim = torch.load("output_Dim.pt")
pad_Index = torch.load("pad_Index.pt")

#Creates tokenizer instance 
tokenizer = torchtext.data.utils.get_tokenizer("basic_english") 

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


def load_NBoW(): 
    #Initializes instance of the model 
    model = NBoW(vocab_Size, embedding_Dim, output_Dim, pad_Index) 
    #Loads the state dictionary from memory 
    model.load_state_dict(torch.load('nbow.pt')) 
    #Sets the model to evaluation mode 
    model.eval()  
    return model

#Creates an instance of the model 
nbow = load_NBoW()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Function that returns sentiment prediction   
def predict_sentiment(text):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = nbow(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item() 
    return predicted_class, predicted_probability 

#Debugging 
#text = 'That moviee was fantastic!'
#print(predict_sentiment(text))

