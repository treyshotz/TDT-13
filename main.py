import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
# Load data
target = 'category'
df = pd.read_csv('output.csv')
df.head(10)

#%%
# Info
df.info()

#%%
# Distribution of district
district_dist = df['district'].value_counts()
district_dist

#%%
# Plot distribution of districts
plt.figure()
sns.countplot(data=df, x='district', palette="viridis", order=district_dist.index)
plt.title('Distribution of Districts')
plt.xlabel('Districts')
plt.ylabel('Number of Texts')
plt.xticks(ticks=range(len(district_dist)), labels=district_dist.keys(), rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('distribution_districts.png')

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
plt.savefig('img/distribution_hour_posted.png')

#%%
# Categories in specific hours
category_dist = df[target].value_counts()
plt.figure(figsize=(40,20))
sns.countplot(data=df, x='hour', hue=target, palette="viridis", hue_order=category_dist.index)
plt.title('Hourly Distribution of Tweets by Category')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Tweets')
plt.yscale("log")
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
# ugly figure
plt.savefig('img/distribution_hour_category_log_test2.png')


#%%
# Missing values
missing_values = df.isnull().sum()
print(missing_values)

#%%
# Remove updatedOn because it only has null values
df = df.drop(['updatedOn', 'hour'], axis='columns')

#%%
# Encode labels
categories_list = df[target].unique()
categories_dict = {category: index for index, category in enumerate(categories_list)}

# Replace category values with encoded labels
df[target] = df[target].map(categories_dict)
df.head()

#%%
# Distribution of category
category_dist = df[target].value_counts()
category_dist

#%%
# Plot distribution of categories
plt.figure(figsize=(12,6))
sns.countplot(data=df, x=target, palette="viridis", order=category_dist.index)
plt.title('Distribution of Categories')
plt.xlabel('Category (Encoded)')
plt.ylabel('Number of Texts')
plt.xticks(ticks=range(len(categories_dict)), labels=categories_dict.keys(), rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('img/distribution_catgoeries.png')
#%%
# First date and last date in dataset
earliest = df['createdOn_datetime'].min()
earliest
#%%
latest = df['createdOn_datetime'].max()
latest

#%%
# Text length
df['text_length'] = df['text'].apply(len)
plt.figure()
sns.histplot(df['text_length'], bins=50, kde=True, color="blue")
plt.title('Distribution of Text Lengths')
plt.xlabel('Length of Text')
plt.ylabel('Number of Texts')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('img/distribution_textlength.png')

#%%
# Text stats yo
text_length_stats = df['text_length'].describe()
text_length_stats


