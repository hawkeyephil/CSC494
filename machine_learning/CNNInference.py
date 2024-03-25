import torch
import torch.nn as nn
import torchtext

#Loads model parameters  
vocab = torch.load("CNN_cache/vocab.pt") 
vocab_Size = len(vocab) 

embedding_Dim = torch.load("CNN_cache/embedding_Dim.pt")
output_Dim = torch.load("CNN_cache/output_Dim.pt")
pad_Index = torch.load("CNN_cache/pad_Index.pt") 

n_Filters = torch.load("CNN_cache/n_Filters.pt") 
filter_Sizes = torch.load("CNN_cache/filter_Sizes.pt")
min_Length = max(filter_Sizes) 
dropout_Rate = torch.load("CNN_cache/dropout_Rate.pt")

#Creates tokenizer instance 
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

class CNN(nn.Module):
    def __init__(self, vocab_Size, embedding_Dim, n_Filters, filter_Sizes, output_Dim, dropout_Rate, pad_Index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_Size, embedding_Dim, padding_idx=pad_Index)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embedding_Dim, n_Filters, filter_size)
                for filter_size in filter_Sizes
            ]
        )
        self.fc = nn.Linear(len(filter_Sizes) * n_Filters, output_Dim)
        self.dropout = nn.Dropout(dropout_Rate)

    def forward(self, ids):
        #ids = [batch size, seq len]
        embedded = self.dropout(self.embedding(ids))
        #embedded = [batch size, seq len, embedding dim]
        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, embedding dim, seq len]
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n filters, seq len - filter_sizes[n] + 1]
        pooled = [conv.max(dim=-1).values for conv in conved]
        #pooled_n = [batch size, n filters]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        #cat = [batch size, n filters * len(filter_sizes)]
        prediction = self.fc(cat)
        #prediction = [batch size, output dim]
        return prediction

def load_CNN(): 
    #Initializes instance of the model 
    model = CNN(vocab_Size, embedding_Dim, n_Filters, filter_Sizes, output_Dim, dropout_Rate, pad_Index) 
    #Loads the state dictionary from memory 
    model.load_state_dict(torch.load('CNN_cache/cnn.pt')) 
    #Sets the model to evaluation mode 
    model.eval()  
    return model

#Creates an instance of the model 
cnn = load_CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Function that predicts sentiment 
def predict_Sentiment(text, model, tokenizer, vocab, device, min_Length, pad_Index):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    if len(ids) < min_Length:
        ids += [pad_Index] * (min_Length - len(ids))
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_Class = prediction.argmax(dim=-1).item()
    predicted_Probability = probability[predicted_Class].item()
    return predicted_Class, predicted_Probability 

#Debugging 
text = "This film is fantastic!" 
print(predict_Sentiment(text, cnn, tokenizer, vocab, device, min_Length, pad_Index))
