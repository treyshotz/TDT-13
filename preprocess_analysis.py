import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load data
target = 'category'
df = pd.read_csv('data/raw_data/logs2023-11-08T12:00:00.000Z.csv')
df.head(10)

#%%
# Info
df.info()
#%%

test = {'Trafikk': "Buss kjørt i bil bakfra. En person lettere skadet. Nødetatene er på stedet. Skaper en del kø på stedet. Kjøretøyene er nå flyttet og veien er åpen.",
 'Andre Hendelser': "Nødetatene  og Avinor gjennomfører øvelse ved Bodø lufthavn. Til info.",
'Brann': "09:25 Nødetater har rykket ut på melding om røykutvikling i et teknisk rom. Alle på stedet er evakuert ut. Ingen personer virker å være skadet.",
'Dyr': "Politiet har fått flere henvendelser fra folk som mener de har sett en ulv i Brumunddal sentrum. Observasjonene skal ha vært litt over kl 0200 i natt. Politiet har søkt i området, men ikke sett noe dyr.",
'Innbrud': "Politiet har vært på åstedsundersøkelse hos en bedrift i Fjordgaten som har hatt innbrudd iløpet av natten. Uvedkommende har tatt seg inn ved å knuse et vindu i en dør, og det er borttatt et mindre beløp fra kassaapparatet. ",
'Redning': "Politiet har iverksatt søk etter en kvinne i 60 årene som er savnet fra en helseinstitusjon. Frivillige fra Røde Kors, Norsk folkehjelp og Norske Redningshunder bistår i søket. Kvinnen har kort grått hår og er trolig iført en lys dunjakke.",
'Ro og orden': "Mann i 30 åra innbrakt til arrest. Ordensforstyrrelse. Drakk på offentlig sted. Når dette ble påtalt truet han politiet.",
'Savnet': "Politiet har startet en leteaksjon etter en savnet mann på Ørnes. Frivillige er kalt ut. I den forbindelse er politiet interessert i tips om aktuell person. Beskrivelse: Mørk jakke, Sort cowboyhatt, hvite bukser. Observasjoner meldes til politiet på 02800.",
'Sjø': "Meldt om båt på rek utenfor Kvamsøyna. HRS har fått noen til å ta hånd om båten. Den er nå fortøyd ved Masfjordnes. Båten er av typen åpen fritidsbåt med påhengsmotor. Eier kan kontakte politiet på 02800.",
'Skadeverk': "Klokken 0111 fikk politiet melding via vaktselskap om innbrudd på adresse i Sandviksbodene. Politiet på stedet fant et knust vindu i dør. Ikke meldt om noe som er stjålet. Uvisst om noen har kommet seg inn på adressen, eller kun snakk om skadeverk.",
'Tyveri': "Politiet har vært på åstedsundersøkelse hos en bedrift som har hatt innbrudd iløpet av helgen. Uvedkomne har brutt seg inn en dør og borttatt verktøy. Forholdet ble oppdaget av ansatte idag tidlig.",
'Ulykke': "Melding om trafikkulykke med to kjøretøy involvert på FV715 Austdalsveien like nord for kommunegrensa til Åfjord. Muligens 4 personer til sammen. En person klager over smerter i en arm. Nødetatene på vei. Oppdatteres.",
'Voldshendelse': ": kl. 0315: melding om krangling mellom flere gjester ved et utested i Lakselv. Politiet til stedet. Uoverensstemmelser mellom noen gjester. Politiet gir pålegg til partene om å forlate stedet. Ingen sak opprettes."
 }
#%%
# Distribution of district
district_dist = df['district'].value_counts()
district_dist

#%%

counted = df.groupby('category')['category'].count()
print(counted)
counted.plot(kind="bar", title="Plot")
plt.show()

#%%
pd.read_csv("data/output_concat.csv").groupby('label')['label'].count().plot(kind="bar", title="Plot")
plt.show()

#%%
weight_df = pd.read_csv("data/output_enc_train.csv")#.groupby('label')['label'].count()#.plot(kind="bar", title="Plot")
# plt.show()
#%%
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
#compute the class weights
class_wts = compute_class_weight('balanced', classes=np.unique(weight_df['label']), y=weight_df['label'])

#%%
pd.read_csv("data/kaggle_out.csv").groupby('label')['label'].count().plot(kind="bar", title="Plot")
plt.show()
pd.read_csv("data/kaggle_out.csv").count()
#%%
df.loc[df['category'] == 'Ulykke']

#%%
test_df = pd.read_csv("data/training_and_test_data/output_enc_concat_test.csv").groupby('label')['label'].count()#.plot(kind="bar", title="Plot")
# plt.show()
#%%
train_df = pd.read_csv("data/output_enc_concat_train.csv").groupby('label')['label'].count()#.plot(kind="bar", title="Plot")
# plt.show()
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
