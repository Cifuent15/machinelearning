import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Model(nn.Module):
    def __init__(self, in_features= 7, h1 = 5, h2=5,h3=5, out_features=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,h3)
        self.out = nn.Linear(h3,out_features)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)\
        
        return x

torch.manual_seed(41)

model = Model()

data_path = 'C:/Users/LuisC/OneDrive/Escritorio/Documentos/Data ML/Air_quality/air_quality_health_impact_data.csv'

df = pd.read_csv('air_quality_health_impact_data.csv')


features = ['PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity']

df_X = df[features]
df_y = df['HealthImpactClass']

X = df_X.values
y= df_y.values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epochs = 200

losses = []

for i in range(epochs):
    y_predict = model.forward(X_train)
    loss = criterion(y_predict,y_train)
    losses.append(loss.detach().numpy())

    if i%10 == 0:
        print(f'Epoch: {i+1} \t loss: {loss:.3f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#plt.plot(range(epochs),losses)
#plt.xlabel('epochs')
#plt.ylabel('loss/error')
#plt.show()

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