import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
import glob
from pathlib import Path
torch.manual_seed(1)
np.random.seed(1)

# $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb=8192"



training_path = 'C:\\Users\\jrura\\Desktop\\biai_resources\\datasets\\huge\\asl\\ASL_Alphabet_Dataset\\asl_alphabet_train'
testing_path = 'C:\\Users\\jrura\\Desktop\\biai_resources\\datasets\\huge\\asl\\ASL_Alphabet_Dataset\\asl_alphabet_test'
classes = os.listdir(training_path)
train_images = []
train_labels = []
test_labels = []
test_images = []

for cl in classes:
    files = os.listdir(training_path + '/' + cl)
    for file in files:
        file_path = os.path.join(training_path, cl, file)
        train_images.append(file_path) # add path to image
        # maybe check if file is image
        train_labels.append(cl) # add label

test_file_list = os.listdir(testing_path)
for file in test_file_list:
    file_path = os.path.join(testing_path, file)
    test_images.append(file_path)
    test_labels.append(file.split('_')[0]) # add label to test image


train_images = pd.Series(train_images, name= 'image_paths')
train_labels = pd.Series(train_labels, name='labels')

test_images = pd.Series(test_images, name= 'image_paths')
test_labels = pd.Series(test_labels, name='labels')

train_dataFrame = pd.DataFrame(pd.concat([train_images, train_labels], axis=1))
test_dataFrame = pd.DataFrame(pd.concat([test_images, test_labels], axis=1))


train_dataFrame, validation_df = train_test_split(train_dataFrame, train_size=0.8, random_state=0)


labelEnc = LabelEncoder()
train_dataFrame['encoded_labels'] = labelEnc.fit_transform(train_dataFrame['labels'])
validation_df['encoded_labels'] = labelEnc.transform(validation_df['labels'])
test_dataFrame['encoded_labels'] = labelEnc.transform(test_dataFrame['labels'])


class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, dataFrame, transform=transforms.Compose([transforms.ToTensor()])):
        self.dataFrame = dataFrame
        self.transform = transform
    
    def __len__(self):
        length = len(self.dataFrame)
        return length
    
    def __getitem__(self, index):
        image_path = self.dataFrame.iloc[index, 0]
        label = self.dataFrame.iloc[index, 2]
        label = torch.tensor(label)
        image = Image.open(image_path).convert('RGB')
        img = np.array(image)
        image = self.transform(image=img)['image']
        return image, label
    

train_transforms = A.Compose([
    A.GaussNoise(p=0.5),
    A.Blur(p=0.5),
    A.Resize(200,200),
     # resize image to 100x100
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])  

test_transforms = A.Compose([
    A.Resize(200,200),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])


train_dataset = ASLDataset(train_dataFrame, transform=train_transforms)
validation_dataset = ASLDataset(validation_df, transform=test_transforms)
test_dataset = ASLDataset(test_dataFrame, transform=test_transforms)

batch_size = 128 # 32 images per batch

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size) # shuffle data
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 100 * 100, 81),
            nn.Dropout(0.2),
            nn.BatchNorm1d(81),
            nn.LeakyReLU(81, 29))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 32 * 100 * 100)
        x = self.fc(x)
        return x
    

class EarlyStop:
    
    # stops when valid loss doesn't improve for patience number of epochs
    def __init__(self, patience=5, verbose=True, delta=0):
        # patience is number of epochs to wait before stopping when loss doesn't improve
        # verbose is whether to print out epoch logs
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf # set initial "min" to infinity

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# retrain model

model = SimpleCNN()
# model.fc = nn.Linear(32 * 100 * 100, len(classes))
model.load_state_dict(torch.load('asl_model_giga_v2.pt'))

for param in model.parameters():
    param.requires_grad = False

model.to(device)

model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


epochs = 6

total_train_loss = []
total_valid_loss = []
best_valid_loss = np.Inf
early_stop = EarlyStop(patience=5, verbose=True)

for epoch in range(epochs):
    print('Epoch:', epoch + 1)
    train_loss = []
    valid_loss = []
    train_correct = 0
    train_total = 0
    valid_correct = 0
    valid_total = 0
    for image, target in train_loader:
        model.train()
        image, target = image.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target.to(torch.long))  # Convert target labels to torch.long
        train_loss.append(loss.item())
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
        
        loss.backward()
        optimizer.step()
        
    for image, target in validation_loader:
        with torch.no_grad():
            model.eval()
            image, target = image.to(device), target.to(device)
            
            output = model(image)
            loss = criterion(output, target.to(torch.long))  # Convert target labels to torch.long
            valid_loss.append(loss.item())
            _, predicted = torch.max(output.data, 1)
            valid_total += target.size(0)
            valid_correct += (predicted == target).sum().item()
            
    epoch_train_loss = np.mean(train_loss)
    epoch_valid_loss = np.mean(valid_loss)
    print(f'Epoch {epoch + 1}, training loss: {epoch_train_loss:.4f}, validation loss: {epoch_valid_loss:.4f}, training accuracy: {(100 * train_correct / train_total):.4f}%, validation accuracy: {(100 * valid_correct / valid_total):.4f}%')
    
    if epoch_valid_loss < best_valid_loss:
        torch.save(model.state_dict(), 'asl_model_giga_v3.pt')
        print('Model improved. Saving model.')
        best_valid_loss = epoch_valid_loss
    
    early_stop(epoch_valid_loss, model)
        
    if early_stop.early_stop:
        print("Early stopping")
        break
        
    lr_scheduler.step(epoch_valid_loss)
    total_train_loss.append(epoch_train_loss)
    total_valid_loss.append(epoch_valid_loss)




