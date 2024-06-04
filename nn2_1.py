import pandas as pd
import random
import os
import sklearn.model_selection as sk_model_selection
import sklearn.preprocessing as sk_preprocessing
import sklearn.metrics as sk_metrics
import sklearn.neural_network as sk_nn

data_dir = './Υλοποιητική Εργασία/harth'
random_file = random.choice(os.listdir(data_dir))
file_path = os.path.join(data_dir, random_file)
with open(file_path,'r') as csvfile: 
    df = pd.read_csv(csvfile)
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.time

label = df['label']
df = df.drop(columns=['label'])

label_encoder = sk_preprocessing.LabelEncoder()
label_labels_encoded = label_encoder.fit_transform(label)
x_train, x_test, y_train, y_test = sk_model_selection.train_test_split(df, label_labels_encoded, test_size=0.2, random_state=42)

mlp_classifier = sk_nn.MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000)
mlp_classifier.fit(x_train, y_train)

y_pred = mlp_classifier.predict(x_test)

# Calculate accuracy
accuracy = sk_metrics.accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)