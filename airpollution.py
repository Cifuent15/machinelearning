import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def validation(n_estimators,max_depth,X_train,X_test,y_train,y_test):
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=41)
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    score = accuracy_score(predict,y_test)
    return score


data_path = 'C:/Users/LuisC/OneDrive/Escritorio/Documentos/Data ML/Air_quality/air_quality_health_impact_data.csv'

df = pd.read_csv('air_quality_health_impact_data.csv')




features = ['PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity']

df_X = df[features]
df_y = df['HealthImpactClass']

X_train,X_test,y_train,y_test = train_test_split(df_X,df_y, train_size=0.25,random_state=41)

for n_estimators in [10,25,30,35,40,45,50,70]:
    for max_depth in [5,10,15,20,25,30]:
        accuracy = validation(n_estimators,max_depth,X_train,X_test,y_train,y_test)
        print(f'numero de arboles: {n_estimators} \t numero de splits: {max_depth} \t precision: {accuracy:.2f}')


