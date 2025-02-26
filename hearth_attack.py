import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_path = 'C:/Users/LuisC/OneDrive/Escritorio/Documentos/Data ML/hearth failure/heart_failure_clinical_records.csv'

df = pd.read_csv('heart_failure_clinical_records.csv')

def validation(n_estimators,max_depth,X_train,X_test,y_train,y_test):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=1)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(prediction,y_test)
    return accuracy


features = df.columns.tolist()


total_features = features.copy()

total_features.remove('DEATH_EVENT')

X_values = df[total_features]

y_values = df['DEATH_EVENT']


X_train,X_test,y_train,y_test = train_test_split(X_values,y_values,train_size=0.30,random_state=41)

for n_estimators in [5,10,15,30,50,70]:
    for max_depth in [5,10,15,20,25]:
        precision = validation(n_estimators,max_depth,X_train,X_test,y_train,y_test)
        print(f'number of tree: {n_estimators} \t splits: {max_depth} \t accuracy: {precision:.3f}')


