import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# %%

nltk.download('punkt')
nltk.download('stopwords')

# %% Load the data

train_data = pd.read_csv('../data/training_and_test_data/output_enc_concat_train.csv')
test_data = pd.read_csv('../data/training_and_test_data/output_enc_concat_test.csv')

id2label = {0: 'Innbrudd', 1: 'Trafikk', 2: 'Brann', 3: 'Tyveri', 4: 'Ulykke', 5: 'Ro og orden', 6: 'Voldshendelse',
            7: 'Andre hendelser', 8: 'Savnet', 9: 'Skadeverk', 10: 'Dyr', 11: 'Sjø', 12: 'Redning', 13: 'Arrangement'}

train_data['category'] = train_data['label'].map(id2label)
test_data['category'] = test_data['label'].map(id2label)

stopwords_norwegian = list(stopwords.words('norwegian'))


# %% Preprocessing

def preprocess_text(data):
    # lowercase, remove punctuations and numbers
    data['processed_text'] = data['text'].map(lambda x: re.sub(r'[\W_\d]+', ' ', x.lower()))

    # tokenize and remove stopwords

    data['processed_text'] = data['processed_text'].map(lambda x: ' '.join(
        [word for word in word_tokenize(x) if word not in stopwords_norwegian]
    ))

    # stemming
    stemmer = SnowballStemmer("norwegian")
    data['processed_text'] = data['processed_text'].apply(
        lambda x: ' '.join([stemmer.stem(token) for token in x.split()]))

    return data


train_data = preprocess_text(train_data)
test_data = preprocess_text(test_data)

# %% vectorize the preprocessed text

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stopwords_norwegian)
dtm_train = vectorizer.fit_transform(train_data['processed_text'])
dtm_test = vectorizer.transform(test_data['processed_text'])

# %% Run model and calculate NPMI score

texts_tokenized = [word_tokenize(text) for text in train_data['processed_text']]
dictionary = Dictionary(texts_tokenized)

num_topics_range = [5, 6, 7, 8, 9, 10, 15, 20]
no_top_words = 10
results = []


# Display the top words for each topic
def display_top_words(lda_model, vectorizer, no_top_words, num_topics):
    print(f"\nTop words for LDA model with {num_topics} topics:")
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        top_frequencies = [topic[i] for i in top_indices]

        print(f"Topic {topic_idx}:")
        for word, frequency in zip(top_words, top_frequencies):
            print(f"{word}: {frequency}")


# Finding similarities between real labels and the topics LDA predicted
def compare_with_real_labels(lda_model, data, dtm, num_topics, dataset_name):
    topic_assignments = lda_model.transform(dtm)
    data['LDA_Topic'] = topic_assignments.argmax(axis=1)
    cross_tab = pd.crosstab(data['category'], data['LDA_Topic'])
    cross_tab.to_csv(f'lda_model_real_label_{dataset_name}_{num_topics}.csv', index=True)


for num_topics in num_topics_range:
    print(f"LDA on topics = {num_topics}")
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dtm_train)

    topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_feature_indices = topic.argsort()[:-no_top_words - 1:-1][::-1]
        topic_words.append([vectorizer.get_feature_names_out()[i] for i in top_feature_indices])

    coherence_model = CoherenceModel(topics=topic_words, texts=texts_tokenized, dictionary=dictionary,
                                     coherence='c_npmi')
    coherence_score = coherence_model.get_coherence()

    results.append((num_topics, coherence_score))

    display_top_words(lda_model, vectorizer, no_top_words, num_topics)

    compare_with_real_labels(lda_model, train_data, dtm_train, num_topics, 'train')
    compare_with_real_labels(lda_model, test_data, dtm_test, num_topics, 'test')

for num_topics, score in results:
    print(f"NPMI Coherence Score for {num_topics} topics: {score}")
