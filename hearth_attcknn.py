import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

class Model(nn.Module):
    def __init__(self, features=12 , h1=100, h2=100, h3=100, h4=100, output=2):
        super().__init__()
        self.fc1 = nn.Linear(features,h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, output) 

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x= F.relu(self.fc3(x))
        x= F.relu(self.fc4(x))
        x=self.out(x)

        return x


torch.manual_seed(41)

model = Model()

data_path = 'C:/Users/LuisC/OneDrive/Escritorio/Documentos/Data ML/hearth failure/heart_failure_clinical_records.csv'

df = pd.read_csv('heart_failure_clinical_records.csv')
features = df.columns.tolist()
features_x = features.copy()
features_x.remove('DEATH_EVENT')

x_values = df[features_x]
y_values = df['DEATH_EVENT']
x_values = x_values.values
y_values = y_values.values



X_train,X_test,y_train,y_test = train_test_split(x_values,y_values,test_size=0.5,random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epochs = 200

losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred,y_train)

    losses.append(loss.detach().numpy())

    if i% 10 == 0:
        print(f'Epoch: {i+1} and loss: {loss}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

#plt.plot(range(epochs), losses)
#plt.xlabel('epochs')
#plt.ylabel('loss/error')

model.eval()
with torch.no_grad():
    y_pred_train = model(X_train)
    y_pred_test = model(X_test)

_, y_pred_train = torch.max(y_pred_train,1)
_, y_pred_test = torch.max(y_pred_test,1)

y_pred_train = y_pred_train.numpy()
y_pred_test = y_pred_test.numpy()

train_accuracy = accuracy_score(y_pred_train,y_train)
test_accuracy = accuracy_score(y_pred_test,y_test)

print(f'Train accuracy: {train_accuracy:.3f} \t test_accuracy: {test_accuracy:.3f}')