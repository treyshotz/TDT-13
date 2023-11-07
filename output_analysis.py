import pandas as pd
import matplotlib.pyplot as plt

nor_pd = pd.read_csv("data/incorrect_ltgnorbert3-base.csv")
mBERT_pd = pd.read_csv("data/incorrect_bert-base-multilingual-cased.csv")
BERT_pd = pd.read_csv("data/incorrect_bert-base-cased.csv")
naive_pd = pd.read_csv("data/incorrect_naive.csv")

label2id = {'Innbrudd': 0, 'Trafikk': 1, 'Brann': 2, 'Tyveri': 3, 'Ulykke': 4, 'Ro og orden': 5, 'Voldshendelse': 6,
            'Andre hendelser': 7, 'Savnet': 8, 'Skadeverk': 9, 'Dyr': 10, 'Sjø': 11, 'Redning': 12, 'Arrangement': 13}

# %%
nor_pd.groupby('label')['label'].count().plot(kind="bar", title="Plot")
plt.show()
# %%
mBERT_pd.groupby('label')['label'].count().plot(kind="bar", title="Plot")
plt.show()
# %%
BERT_pd.groupby('label')['label'].count().plot(kind="bar", title="Plot")
plt.show()
# %%
p = naive_pd.groupby('label')['label'].count().plot(kind="bar", title="Plot")
plt.show()
