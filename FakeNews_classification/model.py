import pandas as pd
import deeplake
#https://github.com/activeloopai/deeplake
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

#https://datasets.activeloop.ai/docs/ml/datasets/liar-dataset/
#ds = deeplake.load('hub://activeloop/liar-train')
#The LIAR dataset training set is composed of 10,269 statements.



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device') 



class Liar_dataset(Dataset):
    def __init__(self,data_path, tokenizer, max_len):
        label_encoder = LabelEncoder()
        test_tsv = data_path+'/'+'test.tsv'
        train_tsv = data_path+'/'+'train.tsv'
        valid_tsv = data_path+'/'+'valid.tsv'
        
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.data_train = pd.read_csv(train_tsv, sep ='\t', header=None, usecols=[0,1,2])
        self.data_train.columns = ['ID','Label','Statement']
        self.data_train['Encoded_Label'] = label_encoder.fit_transform(self.data_train['Label'])
        self.data_test = pd.read_csv(test_tsv, sep ='\t', header=None, usecols=[0,1,2])
        self.data_valid = pd.read_csv(valid_tsv, sep ='\t', header=None, usecols=[0,1,2])

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, idx):
        
        statement_index = self.data_train.columns.get_loc('Statement')
        label_index = self.data_train.columns.get_loc('Encoded_Label')
        statement = self.data_train.iloc[idx,statement_index]
        label = self.data_train.iloc[idx,label_index]
        
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
dataset = Liar_dataset(data_path='liar_dataset',tokenizer=tokenizer,max_len=128)
dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)


model = TextClassifier(vocab_size=30000, embedding_dim= 100, num_classes=6)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    i = 0
    for batch in dataloader: 
        i += 1
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) 

        optimizer.zero_grad()


        outputs = model.forward(input_ids)

        loss = criterion(outputs,labels)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        if i%50 == 0:
            print('50 batchs traités !')
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')


#test_tsv = 'liar_dataset/test.tsv'
#data = pd.read_csv(test_tsv, sep ='\t', header=None, usecols=[0,1,2])
#data.columns = ['ID', 'Label', 'Statement']
#label_encoder = LabelEncoder()
#data['Encoded_Label'] = label_encoder.fit_transform(data['Label'])
#print(data.head())
#    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')