import pandas as pd
import re

data = pd.read_csv('IMDB_Dataset.csv')
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower() 
    return text

data['cleaned_reviews'] = data['review'].apply(clean_text)

print("Dataset preparation is done!")
