import pandas as pd
import os
import random
import sklearn.model_selection as sk_model_selection
import sklearn.preprocessing as sk_preprocessing
import sklearn.metrics as sk_metrics
import sklearn.neural_network as sk_nn

data_dir = './Υλοποιητική Εργασία/harth'

files = os.listdir(data_dir)

concat = pd.DataFrame()

for root, _, files in os.walk(data_dir):
    for file in files:
        new = pd.read_csv(os.path.join(root, file), index_col = 'timestamp')
        file_name = os.path.splitext(file)[0]
        new['file_name'] = file_name
        concat = pd.concat([concat, new])
label = concat['label']
#concat.drop(labels = ['index', 'Unnamed: 0'], axis = 'columns', inplace = True)
concat.drop(labels = ['index'], axis = 'columns', inplace = True)
concat.reset_index(inplace = True)
concat['timestamp'] = pd.to_datetime(concat['timestamp']).apply(lambda x: x.timestamp())
time = concat['timestamp']

print(concat)

X = concat.drop(columns = ['timestamp', 'label', 'file_name'])
y = concat['label']

#scaler = sk_preprocessing.MinMaxScaler(feature_range = (-1,1))
#X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

mlp_classifier = sk_nn.MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000)
mlp_classifier.fit(X_train, y_train)

y_pred = mlp_classifier.predict(X_test)

# Calculate accuracy
accuracy = sk_metrics.accuracy_score(y_test, y_pred)

print(y_test)
print(y_pred)
print('Test accuracy: {:0.2f}'.format(accuracy))