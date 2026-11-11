import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/phishing_email.csv")

print(df.head())
print(df.shape)

# Checks for any null values in the dataset
print(df.isnull().sum())

# Checks for any duplicate rows in the dataset
print(df.duplicated().sum())

# Drop duplicate rows in the dataset
cleaned_df = df.drop_duplicates()

print(cleaned_df.duplicated(subset=["text_combined"]).sum())

print(f"Label counts: {cleaned_df["label"].value_counts()}")

X = cleaned_df['text_combined']
y = cleaned_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, stratify=y)

# max_iter is a hyperparameter that sets the maximum number of optimization iterations allowed to adjust the model's weights and bias
model = LogisticRegression(max_iter=1000)

# Builds a vocabulary and assigns numerical weights to words
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

model_metrics = confusion_matrix(y_test, predictions)
model_accuracy = accuracy_score(y_test, predictions)
model_precision = precision_score(y_test, predictions)
model_recall = recall_score(y_test, predictions)
model_f1 = f1_score(y_test, predictions)

print(model_metrics)
print(model_accuracy)
print(model_precision)
print(model_recall)
print(model_f1)