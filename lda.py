import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% Load the data
data_path = 'data/output_concat.csv'
data = pd.read_csv(data_path)

nltk.download('punkt')
nltk.download('stopwords')

# %% Preprocessing
# lowercase, remove punctuations and numbers
data['processed_text'] = data['text'].map(lambda x: re.sub(r'[\W_\d]+', ' ', x.lower()))

# tokenize and remove stopwords
stopwords = list(stopwords.words('norwegian'))
data['processed_text'] = data['processed_text'].map(lambda x: ' '.join(
    [word for word in word_tokenize(x) if word not in stopwords]
))

# stemming
stemmer = SnowballStemmer("norwegian")
data['processed_text'] = data['processed_text'].apply(
    lambda x: ' '.join([stemmer.stem(token) for token in x.split()]))

# Vectorize the preprocessed text
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stopwords)
dtm = vectorizer.fit_transform(data['processed_text'])  # Document-Term Matrix

dtm.shape, vectorizer.get_feature_names_out()[:10]

# %%
num_topics = 14
no_top_words = 10

# run model
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(dtm)

# Display the top words for each topic
for topic_idx, topic in enumerate(lda_model.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([vectorizer.get_feature_names_out()[i]
                    for i in topic.argsort()[:-no_top_words  - 1:-1]]))


#%% Image of top words in each topic
feature_names = vectorizer.get_feature_names_out()

fig, axes = plt.subplots(2, 7, figsize=(30, 15), sharex=True)
axes = axes.flatten()
for topic_idx, topic in enumerate(lda_model.components_):
    top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
    top_features = [feature_names[i] for i in top_features_ind]
    weights = topic[top_features_ind]

    ax = axes[topic_idx]
    ax.barh(top_features, weights, height=0.7)
    ax.set_title(f'Topic {topic_idx +1}', fontdict={'fontsize': 30})
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=20)
    for i in 'top right left'.split():
        ax.spines[i].set_visible(False)
    fig.suptitle('Topics in LDA model', fontsize=40)

plt.savefig('topics_in_lda_model.png')


#%% Try to find some similarities between LDA topics and the existing labels

topic_assignments = lda_model.transform(dtm)
data['LDA_Topic'] = topic_assignments.argmax(axis=1)
pd.set_option('display.max_rows', None)
cross_tab = pd.crosstab(data['category'], data['LDA_Topic'])
cross_tab.to_csv('lda_model_real_label.csv', index=True)

print(cross_tab)