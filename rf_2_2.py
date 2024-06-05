import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import datetime


# Φόρτωση όλων των αρχείων .csv στον φάκελο 'harth'
files = glob.glob('harth/*.csv')

# Φορτώστε τα δεδομένα σας
df = pd.concat([pd.read_csv(file) for file in files])
df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.timestamp())


# Διαχωρίστε τα χαρακτηριστικά και τις ετικέτες
X = df[['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']]
y = df['label']

# Διαχωρίστε τα δεδομένα σε σύνολα εκπαίδευσης και δοκιμής
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Δημιουργήστε και εκπαιδεύστε τον ταξινομητή
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Προβλέψτε τις ετικέτες για το σύνολο δοκιμής
y_pred = clf.predict(X_test)

# Υπολογίστε την ακρίβεια του μοντέλου
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

