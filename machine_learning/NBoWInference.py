import torch
import torch.nn as nn 
import torchtext 
import datasets 

#Building the model 
class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        pooled = embedded.mean(dim=1)
        # pooled = [batch size, embedding dim]
        prediction = self.fc(pooled)
        # prediction = [batch size, output dim]
        return prediction

def load_model():
    model = NBoW(21635, 300, 2, 1)  # Initialize your model
    model.load_state_dict(torch.load('nbow.pt'))  # Load the state dictionary
    model.eval()  # Set to evaluation mode
    return model

# Load the model
loaded_model = load_model()


train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"]) 

#Max length of review 
max_length = 256
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    return {"tokens": tokens}

#Adds token as a new key in each dictionary 
train_data = train_data.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})

#Validation data creation 
test_size = 0.25
train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]
min_freq = 5
special_tokens = ["<unk>", "<pad>"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = torchtext.vocab.build_vocab_from_iterator(train_data["tokens"], min_freq=min_freq, specials=special_tokens) 


text = "This film is not great, it's terrible"
#Predict sentiment 
def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability 

print(predict_sentiment(text, loaded_model, tokenizer, vocab, device))

