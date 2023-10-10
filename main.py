import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
# Load data
target = 'category'
df = pd.read_csv('output.csv')
print(df.head(10))

#%%
# Distribution of category
category_dist = df[target].value_counts()
print(category_dist)

#%%
# Distribution of district
district_dist = df['district'].value_counts()
print(district_dist)

#%%
# Distribution plot of the times the posts are posted
df['createdOn_datetime'] = pd.to_datetime(df['createdOn'])
df['hour'] = df['createdOn_datetime'].dt.hour
plt.figure()
sns.countplot(data=df, x='hour', color="blue")
plt.title('Distribution of Tweets by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Number of Tweets')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# Missing values
missing_values = df.isnull().sum()
print(missing_values)

#%%
# Encode labels
categories_list = df[target].unique()
categories_dict = {category: index for index, category in enumerate(categories_list)}

print(categories_dict)

#%%
# Replace category values with encoded labels

#%%
# First date and last date in dataset
earliest = df['createdOn_datetime'].min()
latest = df['createdOn_datetime'].max()
# 2023-09-24 08:45
# 2023-10-05 09:02
# We feched 1000 elements from 'dateTimeFrom': '2022-01-01T00:00:00.000Z', 'dateTimeTo': '2023-10-05T09:35:42.123Z',

print(earliest)
print(latest)




