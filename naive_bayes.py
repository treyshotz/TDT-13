import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from tabulate import tabulate


class NaiveBayesClassifier:
    def __init__(self, dataset, number_removal=True, apply_stemming=True, apply_stopwords=True, vectorization='count'):
        self.df = pd.read_csv(dataset)
        self.dataset = dataset
        self.number_removal = number_removal
        self.apply_stemming = apply_stemming
        self.apply_stopwords = apply_stopwords
        self.vectorization = vectorization
        self.stopwords = list(stopwords.words('norwegian')) if apply_stopwords else None
        self.stemmer = SnowballStemmer("norwegian")

    def pre_processing(self):
        self.df['text'] = self.df['text'].str.lower()

        if self.number_removal:
            self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\d+', '', x))

        if self.apply_stemming:
            self.df['text'] = self.df['text'].apply(
                lambda x: ' '.join([self.stemmer.stem(token) for token in x.split()]))

        return self.df

    def run_model(self):
        # Split into train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(self.df['text'], self.df['category'], test_size=0.2,
                                                            random_state=42)

        # Apply Naive Bayes
        if self.vectorization == 'count':
            model = make_pipeline(CountVectorizer(stop_words=self.stopwords), MultinomialNB())
        else:
            model = make_pipeline(TfidfVectorizer(stop_words=self.stopwords), MultinomialNB())

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        # report = classification_report(y_test, y_pred)

        self.print_accuracy(accuracy)

    def print_accuracy(self, accuracy):
        # number_removal=True, apply_stemming=True, apply_stopwords=True, vectorization='bow'
        variables = ["Dataset", "Number Removal", "Stemming", "Stopwords", "Vectorization", "Score (Accuracy)"]
        values = [[self.dataset, self.number_removal, self.apply_stemming, self.apply_stopwords, self.vectorization, accuracy]]

        print(tabulate(values, headers=variables, tablefmt='grid'))

    def main(self):
        self.pre_processing()
        self.run_model()


if __name__ == '__main__':
    classifier = NaiveBayesClassifier('data/output_concat.csv', number_removal=True, apply_stemming=True,
                                      apply_stopwords=True, vectorization='count')
    classifier.main()
    classifier = NaiveBayesClassifier('data/output_concat.csv', number_removal=False, apply_stemming=True,
                                      apply_stopwords=True, vectorization='count')
    classifier.main()
    classifier = NaiveBayesClassifier('data/output_concat.csv', number_removal=True, apply_stemming=False,
                                      apply_stopwords=False, vectorization='count')
    classifier.main()
    classifier = NaiveBayesClassifier('data/output_concat.csv', number_removal=True, apply_stemming=True,
                                      apply_stopwords=True, vectorization='tf_idf')
    classifier.main()
    classifier = NaiveBayesClassifier('data/output_concat.csv', number_removal=False, apply_stemming=True,
                                      apply_stopwords=True, vectorization='tf_idf')
    classifier.main()
    classifier = NaiveBayesClassifier('data/output_concat.csv', number_removal=True, apply_stemming=False,
                                      apply_stopwords=False, vectorization='tf_idf')
    classifier.main()

    classifier = NaiveBayesClassifier('data/logs_25oct.csv', number_removal=True, apply_stemming=True,
                                      apply_stopwords=True, vectorization='count')


