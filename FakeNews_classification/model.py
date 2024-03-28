import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter



#https://datasets.activeloop.ai/docs/ml/datasets/liar-dataset/
#ds = deeplake.load('hub://activeloop/liar-train')
#The LIAR dataset training set is composed of 10,269 statements.


writer = SummaryWriter('runs/experiment_1')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device') 



class Dataset(Dataset):
    def __init__(self,data_path, tokenizer, max_len):
        label_encoder = LabelEncoder()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_path, sep ='\t', header=None, usecols=[0,1,2])
        self.data.columns = ['ID','Label','Statement']
        self.data['Encoded_Label'] = label_encoder.fit_transform(self.data['Label'])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        statement_index = self.data.columns.get_loc('Statement')
        label_index = self.data.columns.get_loc('Encoded_Label')
        statement = self.data.iloc[idx,statement_index]
        label = self.data.iloc[idx,label_index]
        
        encoding = self.tokenizer.encode_plus(
            statement,
            add_special_tokens= True,
            max_length = self.max_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TextClassifier,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self,text):
        embedded = self.embedding(text)
        averaged = torch.mean(embedded, dim = 1)
        output = self.fc(averaged)
        return output


batch_size = 16
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = Dataset(data_path='liar_dataset/train.tsv',tokenizer=tokenizer,max_len=128)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

test_dataset = Dataset(data_path='liar_dataset/test.tsv',tokenizer=tokenizer,max_len=128)
test_dataloader = DataLoader((test_dataset),batch_size=batch_size,shuffle=True)


model = TextClassifier(vocab_size=30000, embedding_dim= 100, num_classes=6)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_test_loss = 0
    for batch_idx, batch in enumerate(train_dataloader): 
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) 

        optimizer.zero_grad()
        outputs = model.forward(input_ids)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train loss', loss.item(), epoch * len(train_dataloader) + batch_idx)
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    model.eval()
    
    with torch.no_grad():
        for batch in test_dataloader:
            test_loss = criterion(outputs,labels)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) 
            outputs = model.forward(input_ids)
            total_test_loss+=test_loss.item()
        test_loss /= len(test_dataloader)
    
    writer.add_scalar('Loss/Test', total_test_loss, epoch)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.flush()
writer.close()
#test_tsv = 'liar_dataset/test.tsv'
#data = pd.read_csv(test_tsv, sep ='\t', header=None, usecols=[0,1,2])
#data.columns = ['ID', 'Label', 'Statement']
#label_encoder = LabelEncoder()
#data['Encoded_Label'] = label_encoder.fit_transform(data['Label'])
#print(data.head())
#    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')