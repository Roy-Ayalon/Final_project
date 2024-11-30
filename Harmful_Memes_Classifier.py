import os
import json
import pandas as pd
import numpy as np

from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from transformers import BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the JSON data into a DataFrame
data = []
with open('/Users/royayalon/Final_project/data/train.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
print(df.head())

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize according to the pre-trained model's requirements
    transforms.Normalize([0.485, 0.456, 0.406],   # Mean
                         [0.229, 0.224, 0.225])   # Std
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # Maximum length for BERT tokenizer

class MemeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transforms, img_dir):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get text and label
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['label']
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # Load and preprocess image
        img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx]['img'].split('/')[-1])
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

train_dataset = MemeDataset(train_df, tokenizer, image_transforms, img_dir='img')
val_dataset = MemeDataset(val_df, tokenizer, image_transforms, img_dir='img')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Remove the classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output [CLS] token representation
        return outputs.pooler_output
    
class MultimodalClassifier(nn.Module):
    def __init__(self, hidden_size=512):
        super(MultimodalClassifier, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        # Combine image and text features
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Concatenate features
        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.classifier(combined_features)
        return output
    
model = MultimodalClassifier()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0
    
    for data in tqdm(data_loader):
        images = data['image'].to(device)
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device).float().unsqueeze(1)
        
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        preds = (outputs >= 0.5).float()
        correct_predictions += torch.sum(preds == labels)
        
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for data in tqdm(data_loader):
            images = data['image'].to(device)
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device).float().unsqueeze(1)
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            losses.append(loss.item())
            preds = (outputs >= 0.5).float()
            correct_predictions += torch.sum(preds == labels)
            
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.BCELoss(weight=class_weights[df['label'].values])

epochs = 5
best_accuracy = 0

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(model, val_loader, criterion, device)
    print(f'Validation loss {val_loss} accuracy {val_acc}')
    
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc