import torch
import torch.nn as nn
import torchtext 

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

def load_CNN(vocab_Size, embedding_Dim, n_Filters, filter_Sizes, output_Dim, dropout_Rate, pad_Index, application): 
    #Initializes instance of the model 
    model = CNN(vocab_Size, embedding_Dim, n_Filters, filter_Sizes, output_Dim, dropout_Rate, pad_Index) 
    #Loads the state dictionary from memory 
    model.load_state_dict(torch.load(application)) 
    #Sets the model to evaluation mode 
    model.eval()  
    return model

#Loads the cnn model parameters 
general_Vocab = torch.load("CNN_cache/vocab.pt") 
general_Vocab_Size = len(general_Vocab) 

general_Embedding_Dim = torch.load("CNN_cache/embedding_Dim.pt")
general_Output_Dim = torch.load("CNN_cache/output_Dim.pt")
general_Pad_Index = torch.load("CNN_cache/pad_Index.pt") 

general_n_Filters = torch.load("CNN_cache/n_Filters.pt") 
general_Filter_Sizes = torch.load("CNN_cache/filter_Sizes.pt")
general_Min_Length = max(general_Filter_Sizes) 
general_Dropout_Rate = torch.load("CNN_cache/dropout_Rate.pt")

#Creates an instance of the cnn model 
general = 'CNN_cache/cnn.pt'
cnn_General = load_CNN(general_Vocab_Size, general_Embedding_Dim, general_n_Filters, general_Filter_Sizes, general_Output_Dim, general_Dropout_Rate, general_Pad_Index, general)

#Loads the finance cnn model parameters 
finance_Vocab = torch.load("finance_CNN_cache/vocab.pt") 
finance_Vocab_Size = len(finance_Vocab) 

finance_Embedding_Dim = torch.load("finance_CNN_cache/embedding_Dim.pt")
finance_Output_Dim = torch.load("finance_CNN_cache/output_Dim.pt")
finance_Pad_Index = torch.load("finance_CNN_cache/pad_Index.pt") 

finance_n_Filters = torch.load("finance_CNN_cache/n_Filters.pt") 
finance_Filter_Sizes = torch.load("finance_CNN_cache/filter_Sizes.pt")
finance_Min_Length = max(finance_Filter_Sizes) 
finance_Dropout_Rate = torch.load("finance_CNN_cache/dropout_Rate.pt")

#Creates an instance of the finance cnn model 
finance = 'finance_CNN_cache/cnn.pt'
cnn_Finance = load_CNN(finance_Vocab_Size, finance_Embedding_Dim, finance_n_Filters, finance_Filter_Sizes, finance_Output_Dim, finance_Dropout_Rate, finance_Pad_Index, finance)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Creates tokenizer instance 
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

#Function that predicts sentiment 
def predict_Sentiment(text, application):
    tokens = tokenizer(text) 
    if application == 'Finance': 
        ids = finance_Vocab.lookup_indices(tokens)
        if len(ids) < finance_Min_Length:
            ids += [finance_Pad_Index] * (finance_Min_Length - len(ids))
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
        prediction = cnn_Finance(tensor).squeeze(dim=0) 
    else: 
        ids = general_Vocab.lookup_indices(tokens)
        if len(ids) < general_Min_Length:
            ids += [general_Pad_Index] * (general_Min_Length - len(ids))
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
        prediction = cnn_General(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_Class = prediction.argmax(dim=-1).item()
    predicted_Probability = probability[predicted_Class].item()
    return predicted_Class, predicted_Probability 

#Debugging 
#text = 'This film is great!'
#print(predict_Sentiment(text, 'General'))
