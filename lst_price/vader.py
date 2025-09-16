import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download("vader_lexicon")

FILE_NAME = "combined_news_bt_final21-25.csv"

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Load your CSV (adjust filename and column name if needed)
df = pd.read_csv(FILE_NAME)

# Assuming your headlines are in a column called 'headline'
def get_sentiment_scores(text):
    return sia.polarity_scores(str(text))  # str() ensures no crash if NaN

# Apply VADER to each headline
df["sentiment"] = df["title"].apply(get_sentiment_scores)

df["compound"] = df["sentiment"].apply(lambda x: x["compound"])
df["neg"] = df["sentiment"].apply(lambda x: x["neg"])
df["neu"] = df["sentiment"].apply(lambda x: x["neu"])
df["pos"] = df["sentiment"].apply(lambda x: x["pos"])

df.to_csv(FILE_NAME, index=False)

df_avg = df.groupby("date", as_index=False)["compound"].mean()
df_avg.to_csv("final_perday_bt.csv", index=False)