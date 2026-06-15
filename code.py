import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = pd.read_csv(url)

df = df[['label', 'tweet']]
df.columns = ['label', 'text']

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_tokens(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['cleaned'] = df['text'].apply(clean_text)
df['processed'] = df['cleaned'].apply(process_tokens)

df = df[df['processed'].str.strip() != '']
df = df.dropna(subset=['processed'])
df = df.drop_duplicates(subset=['processed'])

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, palette='Set2')
plt.title('Sentiment Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()

pos_text = ' '.join(df[df['label'] == 1]['processed'])
wc = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Tweets - Common Words')
plt.tight_layout()
plt.savefig('wordcloud_positive.png')
plt.show()

neg_text = ' '.join(df[df['label'] == 0]['processed'])
wc2 = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
plt.figure(figsize=(10, 5))
plt.imshow(wc2, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Tweets - Common Words')
plt.tight_layout()
plt.savefig('wordcloud_negative.png')
plt.show()

df['length'] = df['processed'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='length', hue='label', bins=30, palette='Set1')
plt.title('Tweet Length by Sentiment')
plt.xlabel('Word Count')
plt.tight_layout()
plt.savefig('length_distribution.png')
plt.show()

X = df['processed']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

preds = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

def predict(text):
    c = clean_text(text)
    p = process_tokens(c)
    v = tfidf.transform([p])
    result = clf.predict(v)[0]
    return "Positive" if result == 1 else "Negative"

print(predict("i had a great time today"))
print(predict("this was a total disaster"))
