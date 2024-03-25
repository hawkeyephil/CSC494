#Import statements 
import collections 
#Visualize results 
import matplotlib.pyplot as plt 
#Numerical processing 
import numpy as np 
#Tensor computations 
import torch 
#Neural networks 
import torch.nn as nn 
#Optimizers 
import torch.optim as optim 
#Text processing 
import torchtext 
#Progress measuring 
import tqdm 
#PreProcessor 
import NBoWPreProcessor as pp

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

#Batch size --> Larger the better for parallel computation, less compute time, and faster training/evaluation 
batch_Size = 512
#Creates dataloader for each set  
train_Data_Loader = get_Data_Loader(train_Data, batch_Size, pad_Index, shuffle=True)
valid_Data_Loader = get_Data_Loader(valid_Data, batch_Size, pad_Index)
test_Data_Loader = get_Data_Loader(test_Data, batch_Size, pad_Index) 

#Model declaration 
class NBoW(nn.Module):
    def __init__(self, vocab_Size, embedding_Dim, output_Dim, pad_Index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_Size, embedding_Dim, padding_idx = pad_Index)
        self.fc = nn.Linear(embedding_Dim, output_Dim)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        pooled = embedded.mean(dim=1)
        # pooled = [batch size, embedding dim]
        prediction = self.fc(pooled)
        # prediction = [batch size, output dim]
        return prediction 

#Model instance 
nbow = NBoW(vocab_Size, embedding_Dim, output_Dim, pad_Index) 

#Function that returns the number of trainable parameters 
def count_Parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_Parameters(nbow):,} trainable parameters") 

vectors = torchtext.vocab.GloVe() 
hello_vector = vectors.get_vecs_by_tokens("hello") 

print(hello_vector.shape) 
print(hello_vector[:32] )

pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos()) 
print(pretrained_embedding.shape) 
print(nbow.embedding.weight)
print(pretrained_embedding) 

nbow.embedding.weight.data = pretrained_embedding
print(nbow.embedding.weight)

optimizer = optim.Adam(nbow.parameters())
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = nbow.to(device)
criterion = criterion.to(device) 

#Training 
def train(data_Loader, model, criterion, optimizer, device):
    model.train()
    epoch_Losses = []
    epoch_Accs = []
    for batch in tqdm.tqdm(data_Loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_Accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_Losses.append(loss.item())
        epoch_Accs.append(accuracy.item())
    return np.mean(epoch_Losses), np.mean(epoch_Accs) 

#Evaluation and accuracy functions 
def evaluate(data_Loader, model, criterion, device):
    model.eval()
    epoch_Losses = []
    epoch_Accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_Loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_Accuracy(prediction, label)
            epoch_Losses.append(loss.item())
            epoch_Accs.append(accuracy.item())
    return np.mean(epoch_Losses), np.mean(epoch_Accs) 

def get_Accuracy(prediction, label):
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
    train_Loss, train_Acc = train(
        train_Data_Loader, model, criterion, optimizer, device
    )
    valid_Loss, valid_Acc = evaluate(valid_Data_Loader, model, criterion, device)
    metrics["train_losses"].append(train_Loss)
    metrics["train_accs"].append(train_Acc)
    metrics["valid_losses"].append(valid_Loss)
    metrics["valid_accs"].append(valid_Acc)
    if valid_Loss < best_Valid_Loss:
        best_valid_loss = valid_Loss
        torch.save(model.state_dict(), "NBoW_cache/nbow.pt") 
        torch.save(vocab, "NBoW_cache/vocab.pt") 
        torch.save(embedding_Dim, "NBoW_cache/embedding_Dim.pt") 
        torch.save(output_Dim, "NBoW_cache/output_Dim.pt") 
        torch.save(pad_Index, "NBoW_cache/pad_Index.pt")
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
plt.show() 

#Loading saved model 
model.load_state_dict(torch.load("NBoW_cache/nbow.pt"))

#Calls evaluate function 
test_loss, test_acc = evaluate(test_Data_Loader, model, criterion, device) 
print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}") 