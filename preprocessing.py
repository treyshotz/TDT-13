import pandas as pd

#%%
# Load data
target = 'category'
df = pd.read_csv('data/raw_data/logs_25oct.csv')

#%%
# Remove updatedOn because it only has null values
df = df.drop(['updatedOn'], axis='columns')

#%%
# Remove updates on a message
df['createdOn'] = pd.to_datetime(df['createdOn'])
df_filtered = df.sort_values(by='createdOn').drop_duplicates(subset='parent_id', keep='first')

#%%
category_dist_filtered = df_filtered[target].value_counts()
category_dist_filtered

#%%
# How many entries have we "lost" after filtering?

category_dist = df[target].value_counts()
lost = category_dist - category_dist_filtered
lost