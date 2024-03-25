import collections
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm 
import machine_learning.MLPreProcessor as pp 

#Seed 
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True 

#Datasets 
train_Data = pp.get_Train_Data() 
valid_Data = pp.get_Valid_Data() 
test_Data = pp.get_Test_Data()  

embedding_Dim = 300 
output_Dim = pp.get_Output_Dim()
pad_Index = pp.get_Pad_Index() 

#Vocabulary 
vocab = pp.get_Vocab() 
vocab_Size = len(vocab) 

n_Filters = 100
filter_Sizes = [3, 5, 7] 
min_Length = max(filter_Sizes)
dropout_Rate = 0.25

#Function that creates a batch 
def get_Collate(pad_Index):
    def collate(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value = pad_Index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch
    return collate

#Function that loads data  
def get_Data_Loader(dataset, batch_Size, pad_Index, shuffle = False):
    collate = get_Collate(pad_Index)
    data_Loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_Size,
        collate_fn = collate,
        shuffle = shuffle,
    )
    return data_Loader

batch_Size = 512

#Creates dataloader for each set  
train_Data_Loader = get_Data_Loader(train_Data, batch_Size, pad_Index, shuffle=True)
valid_Data_Loader = get_Data_Loader(valid_Data, batch_Size, pad_Index)
test_Data_Loader = get_Data_Loader(test_Data, batch_Size, pad_Index)

#Model declaration 
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

#Model instance 
cnn = CNN(vocab_Size, embedding_Dim, n_Filters, filter_Sizes, output_Dim, dropout_Rate, pad_Index)

#Function that returns the number of trainable parameters 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {count_parameters(cnn):,} trainable parameters")

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

cnn.apply(initialize_weights)

vectors = torchtext.vocab.GloVe()
pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
cnn.embedding.weight.data = pretrained_embedding
optimizer = optim.Adam(cnn.parameters())
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = cnn.to(device)
criterion = criterion.to(device)

#Training 
def train(data_Loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_Loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

#Evaluation and accuracy functions 
def evaluate(data_Loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_Loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def get_accuracy(prediction, label):
    batch_Size, _ = prediction.shape
    predicted_Classes = prediction.argmax(dim=-1)
    correct_Predictions = predicted_Classes.eq(label).sum()
    accuracy = correct_Predictions / batch_Size
    return accuracy

#Training 
n_epochs = 10
best_Valid_Loss = float("inf")

metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_Loss, train_Acc = train(train_Data_Loader, model, criterion, optimizer, device)
    valid_Loss, valid_Acc = evaluate(valid_Data_Loader, model, criterion, device)
    metrics["train_losses"].append(train_Loss)
    metrics["train_accs"].append(train_Acc)
    metrics["valid_losses"].append(valid_Loss)
    metrics["valid_accs"].append(valid_Acc)
    if valid_Loss < best_Valid_Loss:
        best_Valid_Loss = valid_Loss
        torch.save(model.state_dict(), "CNN_cache/cnn.pt")
        torch.save(vocab, "CNN_cache/vocab.pt")
        torch.save(embedding_Dim, "CNN_cache/embedding_Dim.pt")
        torch.save(output_Dim, "CNN_cache/output_Dim.pt")
        torch.save(pad_Index, "CNN_cache/pad_Index.pt")
        torch.save(n_Filters, "CNN_cache/n_Filters.pt") 
        torch.save(filter_Sizes, "CNN_cache/filter_Sizes.pt")
        torch.save(dropout_Rate, "CNN_cache/dropout_Rate.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_Loss:.3f}, train_acc: {train_Acc:.3f}")
    print(f"valid_loss: {valid_Loss:.3f}, valid_acc: {valid_Acc:.3f}") 

#Visualizations 
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_losses"], label="train loss")
ax.plot(metrics["valid_losses"], label="valid loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid() 

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="train accuracy")
ax.plot(metrics["valid_accs"], label="valid accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid() 

#Loading saved model 
model.load_state_dict(torch.load("CNN_cache/cnn.pt"))

#Calls evaluate function 
test_loss, test_acc = evaluate(test_Data_Loader, model, criterion, device) 
print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}") 
