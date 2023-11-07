import pandas as pd

df = pd.read_csv("data/logs_25oct.csv")

# %%
# rename column
df = df.rename(columns={'category': 'label'})

label2id = {'Innbrudd': 0, 'Trafikk': 1, 'Brann': 2, 'Tyveri': 3, 'Ulykke': 4, 'Ro og orden': 5, 'Voldshendelse': 6,
            'Andre hendelser': 7, 'Savnet': 8, 'Skadeverk': 9, 'Dyr': 10, 'Sjø': 11, 'Redning': 12, 'Arrangement': 13}

id2label = {0: 'Innbrudd', 1: 'Trafikk', 2: 'Brann', 3: 'Tyveri', 4: 'Ulykke', 5: 'Ro og orden', 6: 'Voldshendelse',
            7: 'Andre hendelser', 8: 'Savnet', 9: 'Skadeverk', 10: 'Dyr', 11: 'Sjø', 12: 'Redning', 13: 'Arrangement'}
# %%

# To change the labels to numeric values
for i, r in df.iterrows():
    df.at[i, 'label'] = label2id.get(r['label'])

# %%
# check if different messages with same parentid have different labels
unique_labels_per_parent = df.groupby('parent_id')['label'].nunique()
parent_with_multiple_labels = unique_labels_per_parent[unique_labels_per_parent > 1].index.tolist()
parent_with_multiple_labels

# %%
# concat messages with same parent id
df = df.groupby('parent_id').agg({
    'district': 'first',
    'districtId': 'first',
    'label': 'first',
    'municipality': 'first',
    'message_id': 'first',
    'text': ' '.join,
    'createdOn': 'first'

})

# %%
(df['label'] == 4).sum()

# %%
(df['label'] == 1).sum()

# %%
# To save the output
df.to_csv("output_concat.csv")

# %%

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=.2)
df_train.to_csv("output_enc_concat_train.csv")
df_test.to_csv("output_enc_concat_test.csv")

# %%

small_df = df.head(20)
small_df.to_csv("small.csv")
