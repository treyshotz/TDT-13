import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

raw_pd = pd.read_csv("data/raw_data/logs_25oct.csv")
test_pd = pd.read_csv("data/training_and_test_data/output_enc_concat_test.csv")
nor_pd = pd.read_csv("data/incorrect_predictions/incorrect_ltgnorbert3-base.csv")
mBERT_pd = pd.read_csv("data/incorrect_predictions/incorrect_bert-base-multilingual-cased.csv")
BERT_pd = pd.read_csv("data/incorrect_predictions/incorrect_bert-base-cased.csv")
naive_pd = pd.read_csv("data/incorrect_predictions/incorrect_naive.csv")

label2id = {'Innbrudd': 0, 'Trafikk': 1, 'Brann': 2, 'Tyveri': 3, 'Ulykke': 4, 'Ro og orden': 5, 'Voldshendelse': 6,
            'Andre hendelser': 7, 'Savnet': 8, 'Skadeverk': 9, 'Dyr': 10, 'Sjø': 11, 'Redning': 12, 'Arrangement': 13}
id2label = {0: 'Innbrudd', 1: 'Trafikk', 2: 'Brann', 3: 'Tyveri', 4: 'Ulykke', 5: 'Ro og orden', 6: 'Voldshendelse',
            7: 'Andre hendelser', 8: 'Savnet', 9: 'Skadeverk', 10: 'Dyr', 11: 'Sjø', 12: 'Redning', 13: 'Arrangement'}
# %%
# Plot distribution of test data
test_gr = test_pd.groupby('label')['label'].count()
test_gr.plot(kind="bar", title="Test data distribution")
plt.show()
# %%
# Plot incorrect NorBERT predictions as bar chart
nor_gr = nor_pd.groupby('label')['label'].count()
nor_gr.plot(kind="bar", title="NorBERT incorrect")
plt.show()
# %%
# Plot incorrect mBERT predictions as bar chart
mBert_gr = mBERT_pd.groupby('label')['label'].count()
mBert_gr.plot(kind="bar", title="mBERT incorrect")
plt.show()
# %%
# Plot incorrect BERT predictions as bar chart
bert_gr = BERT_pd.groupby('label')['label'].count()
bert_gr.plot(kind="bar", title="BERT incorrect")
plt.show()
# %%
# Plot incorrect Naive bayes predictions as bar chart
naive_gr = naive_pd.groupby('label')['label'].count()
naive_gr.plot(kind="bar", title="Naive Bayes incorrect")
plt.show()
# %%
# Plot the incorrect together
import numpy as np

width = 0.25
r = np.arange(len(label2id))
plt.bar(r - 2 * width, nor_gr.tolist(), color='r', width=width, )
plt.bar(r - width, mBert_gr.tolist(), color='b', width=width, )
plt.bar(r, bert_gr.tolist(), color='g', width=width, )
plt.bar(r + width, naive_gr.tolist(), width=width, )
plt.xticks(r + width, label2id.values())
plt.legend()
plt.show()

# %%
# Count by label in testsplit
test_grouped_by = test_pd.groupby('label')['label'].count()
# %%
print(f"NorBERT acc: {1 - nor_gr / test_grouped_by}")
print(f"mBERT acc: {1 - mBert_gr / test_grouped_by}")
print(f"BERT acc: {1 - bert_gr / test_grouped_by}")
print(f"Naive Bayes acc: {1 - naive_gr / test_grouped_by}")
# %%

m1 = pd.merge(nor_pd, mBERT_pd, how="inner", on=["message_id"])
berts = pd.merge(m1, BERT_pd, how="inner", on=["message_id"])
berts = berts.drop(columns=['Unnamed: 0_x', 'parent_id_x', 'district_x', 'districtId_x',
                            'municipality_x', 'text_x', 'createdOn_x',
                            'Unnamed: 0_y', 'parent_id_y', 'district_y', 'districtId_y',
                            'municipality_y', 'text_y', 'createdOn_y', 'Unnamed: 0',
                            'parent_id', 'district', 'districtId', 'municipality',
                            'createdOn', 'label_y', 'label'])

for i, r in berts.iterrows():
    berts.at[i, 'Category'] = id2label.get(r['label_x'])
    berts.at[i, 'NorBERT_pred'] = id2label.get(r['predicted_x'])
    berts.at[i, 'mBERT_pred'] = id2label.get(r['predicted_y'])
    berts.at[i, 'BERT_pred'] = id2label.get(r['predicted'])

m3 = pd.merge(berts, naive_pd, how="inner", on=["message_id"])

# %%
# Find a specific message in test dataset and raw dataset
ut = test_pd.loc[test_pd['message_id'] == "23rk6w-0"]
ut2 = raw_pd.loc[raw_pd['message_id'] == "23rk6w-0"]

# %%
# Check where all the BERT's predict differently
res = berts.loc[berts['NorBERT_pred'] != berts['mBERT_pred']].loc[berts['NorBERT_pred'] != berts['BERT_pred']].loc[
    berts['BERT_pred'] != berts['mBERT_pred']]
