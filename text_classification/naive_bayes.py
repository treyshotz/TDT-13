import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import nltk
from nltk.stem import SnowballStemmer
import re
from tabulate import tabulate

# %%
nltk.download("stopwords")

# %% Global settings

number_removal = True
apply_stemming = True
apply_stopwords = True
vectorization = 'count'

# %% Load data
df_train = pd.read_csv('data/training_and_test_data/output_enc_concat_train.csv')
df_test = pd.read_csv('data/training_and_test_data/output_enc_concat_test.csv')

# %% Stop words and stemming
stopwords = list(stopwords.words('norwegian')) if apply_stopwords else None
stemmer = SnowballStemmer("norwegian")

# %% Pre-processing
df_train['text'] = df_train['text'].str.lower()
df_test['text'] = df_test['text'].str.lower()

if number_removal:
    df_train['text'] = df_train['text'].apply(lambda x: re.sub(r'\d+', '', x))
    df_test['text'] = df_test['text'].apply(lambda x: re.sub(r'\d+', '', x))

if apply_stemming:
    df_train['text'] = df_train['text'].apply(lambda x: ' '.join([stemmer.stem(token) for token in x.split()]))
    df_test['text'] = df_test['text'].apply(lambda x: ' '.join([stemmer.stem(token) for token in x.split()]))


# %% Function for printing accuracy
def print_accuracy(accuracy_value):
    variables = ["Number Removal", "Stemming", "Stopwords", "Vectorization", "Score (Accuracy)"]
    values = [[number_removal, apply_stemming, apply_stopwords, vectorization, accuracy_value]]

    print(tabulate(values, headers=variables, tablefmt='grid'))


# %% Printing the entries NB predicted incorrectly
def analyze_incorrect_texts(incorrect_pred, true_labels, pred_labels):
    id2label_dict = {0: 'Innbrudd', 1: 'Trafikk', 2: 'Brann', 3: 'Tyveri', 4: 'Ulykke', 5: 'Ro og orden',
                     6: 'Voldshendelse', 7: 'Andre hendelser', 8: 'Savnet', 9: 'Skadeverk', 10: 'Dyr', 11: 'Sjø',
                     12: 'Redning', 13: 'Arrangement'}
    true_labels_names = [id2label_dict[label] for label in true_labels]
    pred_labels_names = [id2label_dict[label] for label in pred_labels]

    df_misclassified = pd.DataFrame({
        'Text': incorrect_pred,
        'True Label ID': true_labels,
        'True Label': true_labels_names,
        'Predicted Label ID': pred_labels,
        'Predicted Label': pred_labels_names
    })

    # Save the DataFrame to a CSV file
    df_misclassified.to_csv('misclassified_texts.csv', index=False, encoding='ISO-8859-1')
    print("Misclassified texts saved to 'misclassified_texts.csv'")


# %% Run model
X_train = df_train['text']
X_test = df_test['text']
y_train = df_train['label']
y_test = df_test['label']

if vectorization == 'count':
    model = make_pipeline(CountVectorizer(stop_words=stopwords), MultinomialNB())
else:
    model = make_pipeline(TfidfVectorizer(stop_words=stopwords), MultinomialNB())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
# print(report)


print_accuracy(accuracy)
# Find the  predictions
incorrect_entries = [index for index, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
df_test.iloc[incorrect_entries].to_csv("incorrect_naive.csv")
incorrect_texts = X_test.iloc[incorrect_entries].tolist()
true_labels = y_test.iloc[incorrect_entries].tolist()
pred_labels = [y_pred[index] for index in incorrect_entries]

analyze_incorrect_texts(incorrect_texts, true_labels, pred_labels)
