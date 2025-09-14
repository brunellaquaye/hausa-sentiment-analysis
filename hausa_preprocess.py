# Hausa Text Preprocessing (Comprehensive Class Version)
import re
import string
from typing import List
import pandas as pd

class HausaTextPreprocessor:
    def __init__(self):
        self.hausa_stopwords = {
            'da', 'ne', 'ce', 'na', 'ta', 'shi', 'ita', 'su', 'ni', 'ka', 'ki', 'ku', 'mu', 'wa', 'zuwa', 'akan',
            'amma', 'ko', 'kuma', 'saboda', 'don', 'ba', 'bai', 'bata', 'baiwa', 'bayan', 'cikin', 'ga', 'ina',
            'yana', 'yake', 'yayi', 'yake', 'yanzu', 'wannan', 'wancan', 'wata', 'wani', 'wasu', 'duk', 'kowa',
            'me', 'mece', 'mene', 'wace', 'wane', 'wacece', 'wanene', 'inda', 'lokacin', 'idan', 'kamar', 'saboda',
            'daidai', 'kawai', 'har', 'sai', 'tun', 'daga', 'zuwa', 'kuma', 'ko', 'amma', 'saboda', 'idan', 'ko',
            'da', 'ba', 'ce', 'ne', 'shi', 'ta', 'su', 'ni', 'ka', 'ki', 'ku', 'mu', 'wa', 'zuwa', 'akan', 'ga',
            'cikin', 'bayan', 'lokacin', 'inda', 'yanzu', 'kamar', 'saboda', 'kawai', 'har', 'sai', 'tun', 'daga'
        }
        self.punctuation = set(string.punctuation)
        self.hausa_chars = set('abcdefghijklmnopqrstuvwxyz’ʼƙɗɓçäöüÀÁÂÃÈÉÊÌÍÎÒÓÔÕÙÚÛÇÑ')

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@[\w_]+', '', text)
        text = re.sub(r'#[\w_]+', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed characters
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip('"\'')
        return text.strip()

    def remove_punctuation(self, text: str) -> str:
        return ''.join(char for char in text if char not in self.punctuation)

    def keep_hausa_chars(self, text: str) -> str:
        return ''.join(char for char in text if char in self.hausa_chars or char.isspace())

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.hausa_stopwords]

    def preprocess(self, text: str, remove_stopwords: bool = True, keep_only_hausa: bool = False) -> str:
        text = self.clean_text(text)
        text = self.remove_punctuation(text)
        if keep_only_hausa:
            text = self.keep_hausa_chars(text)
        tokens = self.tokenize(text)
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        return ' '.join(tokens)

# Instantiate the preprocessor
preprocessor = HausaTextPreprocessor()

# # Show before/after samples for verification
# print('Sample before cleaning:')
# print(train_df['tweet'].head(5))

# # Apply robust Hausa preprocessing
# train_df['tweet_clean'] = train_df['tweet'].apply(lambda x: preprocessor.preprocess(x))
# val_df['tweet_clean'] = val_df['tweet'].apply(lambda x: preprocessor.preprocess(x))
# test_df['tweet_clean'] = test_df['tweet'].apply(lambda x: preprocessor.preprocess(x))

# print('\nSample after cleaning:')
# print(train_df['tweet_clean'].head(5))

# # Show label distribution for verification
# if 'label' in train_df.columns:
#     print('\nTrain label distribution:')
#     print(train_df['label'].value_counts())
# if 'label' in val_df.columns:
#     print('\nValidation label distribution:')
#     print(val_df['label'].value_counts())
# if 'label' in test_df.columns:
#     print('\nTest label distribution:')
#     print(test_df['label'].value_counts())

# # Export cleaned data for training and evaluation
# export_cols = ['tweet_clean', 'label']
# train_df[export_cols].to_csv('data/afrisenti_twitter_hausa_train_clean.csv', index=False)
# val_df[export_cols].to_csv('data/afrisenti_twitter_hausa_validation_clean.csv', index=False)
# test_df[export_cols].to_csv('data/afrisenti_twitter_hausa_test_clean.csv', index=False)
# print('\nCleaned data exported for train, validation, and test sets.')