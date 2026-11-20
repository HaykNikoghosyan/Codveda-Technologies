from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'Data'

df = pd.read_csv(DATA_DIR / '3) Sentiment dataset.csv')

text_column = None
max_len = 0
for col in df.columns:
    avg_len = df[col].astype(str).str.len().mean()
    if avg_len > max_len:
        max_len = avg_len
        text_column = col

print(f'Text Column: "{text_column}"')

def get_sentiment(text):
    analysis = TextBlob(str(text))

    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df[text_column] = df[text_column].fillna('')

df['Sentiment_Result'] = df[text_column].apply(get_sentiment)

print('Sentiment Result:')
print(df['Sentiment_Result'].value_counts())

plt.figure(figsize=(7,5))
sns.countplot(x='Sentiment_Result', data=df, palette='pastel', hue='Sentiment_Result', legend=False)
plt.title('Sentiment Distribution Analysis', fontsize=14)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig(BASE_DIR / 'Sentiment_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()

all_words = ' '.join([str(text) for text in df[text_column]])

if len(all_words.strip()) > 0:
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(all_words)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Dataset', fontsize=16)
    plt.savefig(BASE_DIR / 'wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print('No Words Found')
