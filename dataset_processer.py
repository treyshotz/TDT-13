import pandas as pd

df = pd.read_csv("data/logs_25oct.csv")

#%%
#rename column
df = df.rename(columns={'category': 'label'})

#%%
#Labels to id
labels = list(df['label'].unique())
label2id = dict(zip(labels, range(len(labels))))

# %%

#To change the labels to numeric values
for i, r in df.iterrows():
    df.at[i, 'label'] = label2id.get(r['label'])

#%%
#To save the output
df.to_csv("output_num_enc.csv")

#%%

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=.2)
df_train.to_csv("output_enc_train.csv")
df_test.to_csv("output_enc_test.csv")
