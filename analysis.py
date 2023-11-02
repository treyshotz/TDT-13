import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load data
target = 'category'
df = pd.read_csv('data/logs_25oct.csv')
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
sns.countplot(data=df, x='district', hue='district', palette="viridis", order=district_dist.index, legend=False)
plt.title('Distribution of Districts')
plt.xlabel('Districts')
plt.ylabel('Number of Texts')
plt.xticks(ticks=range(len(district_dist)), labels=district_dist.keys(), rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('img/distribution_districts.png')

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
# Distribution of category
category_dist = df[target].value_counts()
category_dist

#%%
# Traffic category percentage
traffic_occurrence = df[target].value_counts()['Trafikk']/df[target].shape[0]
traffic_occurrence

#%%
# Plot distribution of categories
plt.figure(figsize=(12,6))
sns.countplot(data=df, x=target, order=category_dist.index)
plt.title('Distribution of Categories')
plt.xlabel('Category (Encoded)')
plt.ylabel('Number of Texts')
plt.xticks( rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('img/distribution_categories.png')
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
# Text stats
text_length_stats = df['text_length'].describe()
text_length_stats


#%%
short_tweets = df[df['text_length'] < 50]
pd.set_option('display.max_rows', None)
short_tweets['text']
