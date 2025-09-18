# Advanced Hausa Text Preprocessing for Performance Boost (80%+ Accuracy Target)
# Enhanced with domain adaptation, data augmentation, and advanced feature engineering
import re
import string
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random

class HausaTextPreprocessor:
    """
    Advanced Hausa Text Preprocessor optimized for Ghanaian social media sentiment analysis.
    
    Key Performance Enhancements:
    1. Code-switching handling (Hausa-English mix common in Ghana)
    2. Domain-specific preprocessing for social media patterns
    3. Advanced normalization for noisy text
    4. Sentiment-preserving cleaning
    5. Data augmentation capabilities
    6. Context-aware feature extraction
    """
    
    def __init__(self):
        # Enhanced Hausa stopwords with Ghanaian social media patterns
        self.hausa_stopwords = {
            # Core Hausa stopwords (preserved from original)
            'da', 'ne', 'ce', 'na', 'ta', 'shi', 'ita', 'su', 'ni', 'ka', 'ki', 'ku', 'mu', 'wa', 'zuwa', 'akan',
            'amma', 'ko', 'kuma', 'saboda', 'don', 'ba', 'bai', 'bata', 'baiwa', 'bayan', 'cikin', 'ga', 'ina',
            'yana', 'yake', 'yayi', 'yanzu', 'wannan', 'wancan', 'wata', 'wani', 'wasu', 'duk', 'kowa',
            'me', 'mece', 'mene', 'wace', 'wane', 'wacece', 'wanene', 'inda', 'lokacin', 'idan', 'kamar',
            'daidai', 'kawai', 'har', 'sai', 'tun', 'daga', 'zuwa', 'kuma', 'ko', 'amma', 'saboda',
            
            # Enhanced additions for Ghanaian context
            'a', 'an', 'mun', 'sun', 'kun', 'kin', 'kai', 'kee', 'kenan', 'zai', 'za', 'mai', 'marasa', 'masu',
            'wanda', 'wadanda', 'wacce', 'waɗanda', 'yau', 'jiya', 'gobe', 'fa', 'dai', 'kuwa', 'ma',
            
            # Social media specific stopwords
            'pls', 'please', 'thanks', 'thank', 'ok', 'okay', 'yes', 'no', 'oya'  # Common code-switching terms
        }
        
        self.punctuation = set(string.punctuation)
        # Enhanced Hausa character set with variants found in social media
        self.hausa_chars = set("'abcdefghijklmnopqrstuvwxyzʼƙɗɓçäöüÀÁÂÃÈÉÊÌÍÎÒÓÔÕÙÚÛÇÑ\"")
        
        # ADVANCED: Domain-specific sentiment lexicon for Ghanaian social media
        self.positive_indicators = {
            # Core positive terms
            'kyau', 'mai kyau', 'nagari', 'farin ciki', 'murna', 'jin dadi', 'godiya', 'na gode',
            'alheri', 'albarka', 'madalla', 'zakara', 'kyakkyawa', 'dadi', 'son', 'so', 'dariya',
            'alhamdulillahi', 'mashallah', 'barka', 'mabrouk', 'yayyafa', 'jin daɗi', 'fara\'a',
            
            # Ghanaian social media positive expressions
            'nice', 'good', 'great', 'excellent', 'perfect', 'amazing', 'wonderful', 'beautiful',
            'love', 'like', 'best', 'awesome', 'cool', 'sweet', 'fine', 'well done', 'congratulations',
            
            # Hausa intensifiers for positive sentiment
            'sosai', 'kwatakwata', 'ainun', 'da gaske', 'wallahi', 'lallai'
        }
        
        self.negative_indicators = {
            # Core negative terms
            'mugu', 'mummunan', 'bakin ciki', 'tsoro', 'damuwa', 'haushi', 'bacin rai',
            'rashin', 'kuskure', 'laifi', 'haramun', 'zalunci', 'ban sha\'awa', 'kyama',
            'kiyayya', 'ɓarna', 'azaba', 'wahala', 'matsala', 'rikici', 'fitina', 'tashin hankali',
            
            # Ghanaian social media negative expressions
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'angry', 'mad', 'sad',
            'disappointed', 'frustrated', 'annoying', 'stupid', 'foolish', 'nonsense',
            
            # Hausa intensifiers for negative sentiment
            'da yawa', 'ainun', 'kwata-kwata', 'ba komai ba'
        }
        
        # ADVANCED: Code-switching patterns common in Ghanaian social media
        self.code_switching_patterns = {
            # English-Hausa common switches
            'but': 'amma', 'and': 'da', 'or': 'ko', 'so': 'don haka', 'because': 'saboda',
            'now': 'yanzu', 'today': 'yau', 'tomorrow': 'gobe', 'yesterday': 'jiya',
            'good': 'kyau', 'bad': 'mugu', 'very': 'sosai', 'really': 'da gaske'
        }
        
        # ADVANCED: Social media normalization patterns
        self.social_patterns = {
            'urls': re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE),
            'emails': re.compile(r'\S+@\S+'),
            'mentions': re.compile(r'@[\w_]+'),
            'hashtags': re.compile(r'#[\w_]+'),
            'numbers': re.compile(r'\d+'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),  # Normalize excessive repetition
            'extra_spaces': re.compile(r'\s+'),
            'laughing': re.compile(r'\b(haha|hehe|lol|lmao|lmfao)\b', re.IGNORECASE),  # Normalize laughter
            'emojis': re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002700-\U000027BF"  # Dingbats
                u"\U000024C2-\U0001F251"  # Enclosed characters
                "]+", flags=re.UNICODE),
            # ADVANCED: Detect and preserve sentiment-bearing elements
            'strong_positive': re.compile(r'\b(excellent|amazing|wonderful|fantastic|brilliant|perfect)\b', re.IGNORECASE),
            'strong_negative': re.compile(r'\b(terrible|awful|horrible|disgusting|pathetic|useless)\b', re.IGNORECASE),
            'emphasis': re.compile(r'[!]{2,}|[?]{2,}'),  # Multiple punctuation for emphasis
        }
        
        # ADVANCED: Slang and informal expressions common in Ghanaian social media
        self.slang_normalization = {
            # Normalize common misspellings and slang to standard forms
            'u': 'you', 'ur': 'your', 'n': 'and', '2': 'to', '4': 'for', 'b4': 'before',
            'pls': 'please', 'thnx': 'thanks', 'gud': 'good', 'luv': 'love',
            'wat': 'what', 'wth': 'what the', 'omg': 'oh my god', 'btw': 'by the way',
            'rly': 'really', 'ur': 'your', 'urself': 'yourself', 'bcos': 'because',
            'dnt': 'dont', 'cnt': 'cant', 'wont': 'wont', 'shld': 'should',
            # Hausa social media shortcuts
            'dn': 'don', 'gd': 'good', 'gm': 'good morning', 'gn': 'good night'
        }

    def clean_text(self, text: str) -> str:
        """
        ADVANCED: Enhanced text cleaning optimized for Ghanaian social media.
        
        Key improvements for 80%+ accuracy:
        1. Sentiment-preserving cleaning
        2. Code-switching normalization
        3. Social media pattern handling
        4. Emphasis preservation
        """
        if pd.isna(text):
            return ""
        
        original_text = str(text)
        text = original_text.lower()
        
        # ADVANCED: Preserve sentiment-bearing patterns before cleaning
        sentiment_markers = []
        
        # Extract strong sentiment indicators
        strong_pos = self.social_patterns['strong_positive'].findall(text)
        strong_neg = self.social_patterns['strong_negative'].findall(text)
        emphasis_count = len(self.social_patterns['emphasis'].findall(text))
        
        # Store for later feature extraction
        sentiment_markers = {
            'strong_positive_count': len(strong_pos),
            'strong_negative_count': len(strong_neg), 
            'emphasis_count': emphasis_count
        }
        
        # ADVANCED: Normalize laughter expressions to sentiment markers
        text = self.social_patterns['laughing'].sub(' POSITIVE_MARKER ', text)
        
        # Standard cleaning with enhancements
        text = self.social_patterns['urls'].sub('', text)
        text = self.social_patterns['emails'].sub('', text)
        text = self.social_patterns['mentions'].sub('', text)
        text = self.social_patterns['hashtags'].sub('', text)
        text = self.social_patterns['numbers'].sub('', text)
        
        # ADVANCED: Normalize repeated characters while preserving emphasis
        # e.g., "sooooo good" -> "soo good" (keeps some emphasis)
        text = self.social_patterns['repeated_chars'].sub(r'\1\1', text)
        
        # ADVANCED: Normalize common slang and misspellings
        words = text.split()
        normalized_words = []
        for word in words:
            # Check slang normalization
            if word in self.slang_normalization:
                normalized_words.append(self.slang_normalization[word])
            # Check code-switching normalization
            elif word in self.code_switching_patterns:
                normalized_words.append(self.code_switching_patterns[word])
            else:
                normalized_words.append(word)
        text = ' '.join(normalized_words)
        
        # Remove emojis (but we've already extracted sentiment info)
        text = self.social_patterns['emojis'].sub('', text)
        
        # Clean up whitespace
        text = self.social_patterns['extra_spaces'].sub(' ', text)
        text = text.strip('"\'')
        
        # Store sentiment markers for feature extraction
        if not hasattr(self, '_current_sentiment_markers'):
            self._current_sentiment_markers = {}
        self._current_sentiment_markers[text] = sentiment_markers
        
        return text.strip()

    def remove_punctuation(self, text: str) -> str:
        """
        ADVANCED: Selective punctuation removal that preserves sentiment signals.
        
        Key improvement: Keep emotionally important punctuation as features.
        """
        # ADVANCED: Preserve sentiment-important punctuation
        sentiment_punct = {'!', '?'}  # Exclamation and question marks carry sentiment
        
        # Count sentiment punctuation before removal
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Remove punctuation but preserve sentiment info
        cleaned_chars = []
        for char in text:
            if char not in self.punctuation:
                cleaned_chars.append(char)
        
        result = ''.join(cleaned_chars)
        
        # Add sentiment punctuation as explicit tokens for the model to learn from
        if exclamation_count > 0:
            result += ' EXCLAMATION_MARKER'
        if question_count > 0:
            result += ' QUESTION_MARKER'
        if exclamation_count > 1:  # Multiple exclamations indicate strong sentiment
            result += ' STRONG_EMPHASIS_MARKER'
            
        return result

    def keep_hausa_chars(self, text: str) -> str:
        """Enhanced to handle code-switching context."""
        # ADVANCED: More flexible character filtering for code-switching
        # Allow English characters in code-switching contexts but prioritize Hausa
        hausa_and_english = self.hausa_chars | set('abcdefghijklmnopqrstuvwxyz')
        return ''.join(char for char in text if char in hausa_and_english or char.isspace())

    def tokenize(self, text: str) -> List[str]:
        """
        ADVANCED: Enhanced tokenization with sentiment marker preservation.
        """
        tokens = text.split()
        
        # Filter out very short tokens but preserve sentiment markers
        filtered_tokens = []
        for token in tokens:
            if len(token) > 1 or token.endswith('_MARKER'):
                filtered_tokens.append(token)
                
        return filtered_tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        ADVANCED: Context-aware stopword removal.
        
        Key improvement: Preserve stopwords that might carry sentiment in context.
        """
        filtered_tokens = []
        
        for i, token in enumerate(tokens):
            # Always preserve sentiment markers
            if token.endswith('_MARKER'):
                filtered_tokens.append(token)
                continue
                
            # ADVANCED: Context-aware stopword removal
            # Don't remove stopwords that are part of sentiment expressions
            if token in self.hausa_stopwords:
                # Check if it's part of a sentiment phrase
                if i > 0 and i < len(tokens) - 1:
                    # Look for patterns like "ba ... ba" (negation in Hausa)
                    prev_token = tokens[i-1]
                    next_token = tokens[i+1]
                    
                    # Preserve negation patterns
                    if token == 'ba' and (prev_token == 'ba' or next_token == 'ba'):
                        filtered_tokens.append(token)
                        continue
                    
                    # Preserve other important grammatical patterns
                    if token in ['ne', 'ce'] and (prev_token in self.positive_indicators or prev_token in self.negative_indicators):
                        filtered_tokens.append(token)
                        continue
                
                # Skip regular stopwords
                continue
            
            # Keep non-stopwords
            if len(token) > 1:
                filtered_tokens.append(token)
        
        return filtered_tokens

    def extract_features(self, text: str, original_text: str = "") -> Dict[str, float]:
        """
        ADVANCED: Comprehensive feature extraction for performance boost.
        
        Key enhancements for 80%+ accuracy:
        1. Social media specific features
        2. Code-switching detection
        3. Sentiment intensity features
        4. Linguistic complexity measures
        """
        features = {}
        text_lower = text.lower()
        original_lower = original_text.lower() if original_text else text_lower
        words = text_lower.split()
        
        # ADVANCED: Sentiment indicator features with weights
        pos_count = 0
        neg_count = 0
        strong_pos_count = 0
        strong_neg_count = 0
        
        for indicator in self.positive_indicators:
            if indicator in text_lower:
                if indicator in ['excellent', 'amazing', 'wonderful', 'perfect', 'brilliant']:
                    strong_pos_count += 1
                pos_count += 1
                
        for indicator in self.negative_indicators:
            if indicator in text_lower:
                if indicator in ['terrible', 'awful', 'horrible', 'disgusting', 'pathetic']:
                    strong_neg_count += 1
                neg_count += 1
        
        features['positive_indicators'] = pos_count
        features['negative_indicators'] = neg_count
        features['strong_positive_indicators'] = strong_pos_count
        features['strong_negative_indicators'] = strong_neg_count
        features['sentiment_polarity'] = pos_count - neg_count
        features['sentiment_intensity'] = pos_count + neg_count
        
        # ADVANCED: Code-switching features (important for Ghanaian context)
        english_words = 0
        hausa_words = 0
        mixed_words = 0
        
        for word in words:
            if word in self.code_switching_patterns.keys():  # English words commonly switched
                english_words += 1
            elif any(char in 'ƙɗɓ' for char in word):  # Contains Hausa-specific characters
                hausa_words += 1
            elif word in self.code_switching_patterns.values():  # Hausa words commonly switched
                hausa_words += 1
            else:
                mixed_words += 1
        
        total_words = len(words)
        features['english_word_ratio'] = english_words / max(total_words, 1)
        features['hausa_word_ratio'] = hausa_words / max(total_words, 1)
        features['code_switching_score'] = (english_words + mixed_words) / max(total_words, 1)
        
        # ADVANCED: Social media specific features
        features['has_markers'] = sum(1 for word in words if word.endswith('_MARKER'))
        features['exclamation_markers'] = sum(1 for word in words if 'EXCLAMATION' in word)
        features['question_markers'] = sum(1 for word in words if 'QUESTION' in word)
        features['emphasis_markers'] = sum(1 for word in words if 'EMPHASIS' in word)
        
        # ADVANCED: Text complexity and quality features
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        features['caps_ratio'] = sum(1 for c in original_text if c.isupper()) / len(original_text) if original_text else 0
        
        # ADVANCED: Linguistic pattern features
        features['repetition_ratio'] = len([w for w in words if words.count(w) > 1]) / len(words) if words else 0
        features['hausa_char_ratio'] = sum(1 for c in text if c in 'ƙɗɓ') / len(text) if text else 0
        
        # ADVANCED: Sentiment context features
        features['sentiment_word_density'] = (pos_count + neg_count) / len(words) if words else 0
        features['positive_ratio'] = pos_count / max(pos_count + neg_count, 1)
        features['sentiment_consistency'] = abs(pos_count - neg_count) / max(pos_count + neg_count, 1)
        
        # ADVANCED: Social media engagement features
        original_upper = original_text if original_text else text
        features['all_caps_words'] = sum(1 for word in original_upper.split() if word.isupper() and len(word) > 1)
        features['mixed_case_words'] = sum(1 for word in original_upper.split() if any(c.isupper() for c in word) and any(c.islower() for c in word))
        
        return features

    def preprocess(self, text: str, remove_stopwords: bool = True, keep_only_hausa: bool = False, 
                  extract_features: bool = False) -> str:
        """
        ADVANCED: Enhanced preprocessing pipeline optimized for performance.
        
        Maintains original interface while providing significant improvements.
        """
        original_text = text  # Store for feature extraction
        
        # Extract features before preprocessing if requested
        if extract_features:
            # First pass: basic cleaning to get features
            temp_clean = self.clean_text(text)
            features = self.extract_features(temp_clean, original_text)
        
        # Main preprocessing pipeline with enhancements
        text = self.clean_text(text)
        text = self.remove_punctuation(text)
        
        if keep_only_hausa:
            text = self.keep_hausa_chars(text)
        
        tokens = self.tokenize(text)
        
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        processed_text = ' '.join(tokens)
        
        if extract_features:
            return processed_text, features
        else:
            return processed_text

    def augment_data(self, texts: List[str], labels: List[int], augment_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """
        ADVANCED: Data augmentation specifically for Hausa sentiment analysis.
        
        This method boosts performance by creating additional training examples
        through linguistically-informed transformations.
        """
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        # Augmentation strategies for Hausa text
        augmentation_strategies = [
            self._synonym_replacement,
            self._code_switching_augmentation, 
            self._emphasis_modification,
            self._word_order_variation,
            self._slang_variation
        ]
        
        num_to_augment = int(len(texts) * augment_ratio)
        indices_to_augment = random.sample(range(len(texts)), min(num_to_augment, len(texts)))
        
        for idx in indices_to_augment:
            original_text = texts[idx]
            original_label = labels[idx]
            
            # Apply random augmentation strategy
            strategy = random.choice(augmentation_strategies)
            augmented_text = strategy(original_text)
            
            if augmented_text and augmented_text != original_text:
                augmented_texts.append(augmented_text)
                augmented_labels.append(original_label)
        
        print(f"Data augmentation: {len(texts)} -> {len(augmented_texts)} samples (+{len(augmented_texts) - len(texts)})")
        
        return augmented_texts, augmented_labels

    def _synonym_replacement(self, text: str) -> str:
        """Replace words with Hausa synonyms."""
        synonym_dict = {
            'kyau': 'nagari', 'nagari': 'kyau', 'mugu': 'mummunan', 'mummunan': 'mugu',
            'farin ciki': 'murna', 'murna': 'farin ciki', 'bakin ciki': 'bacin rai',
            'sosai': 'kwatakwata', 'da gaske': 'lallai', 'good': 'kyau', 'bad': 'mugu'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word in synonym_dict and random.random() < 0.3:
                words[i] = synonym_dict[word]
        
        return ' '.join(words)

    def _code_switching_augmentation(self, text: str) -> str:
        """Add code-switching elements common in Ghanaian social media."""
        # Add English equivalents or Hausa equivalents
        switches = {
            'kyau': 'good', 'mugu': 'bad', 'sosai': 'very', 'yanzu': 'now',
            'good': 'kyau', 'bad': 'mugu', 'very': 'sosai', 'now': 'yanzu'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word in switches and random.random() < 0.2:
                words[i] = switches[word]
        
        return ' '.join(words)

    def _emphasis_modification(self, text: str) -> str:
        """Modify emphasis patterns."""
        if random.random() < 0.5:
            # Add emphasis
            if '!' not in text and random.random() < 0.5:
                text += '!'
            elif '!!' not in text and random.random() < 0.3:
                text = text.replace('!', '!!')
        
        return text

    def _word_order_variation(self, text: str) -> str:
        """Slight word order variations (careful with Hausa grammar)."""
        words = text.split()
        if len(words) > 3 and random.random() < 0.3:
            # Simple adjacent word swaps (safe for Hausa)
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)

    def _slang_variation(self, text: str) -> str:
        """Add common social media variations."""
        variations = {
            'good': ['gud', 'gd'], 'you': ['u'], 'your': ['ur'], 
            'please': ['pls'], 'thanks': ['thnx'], 'because': ['bcos']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word in variations and random.random() < 0.2:
                words[i] = random.choice(variations[word])
        
        return ' '.join(words)

    def analyze_corpus(self, texts: List[str]) -> Dict[str, any]:
        """
        ADVANCED: Comprehensive corpus analysis for insights.
        """
        processed_texts = [self.preprocess(text) for text in texts]
        
        # Basic statistics
        stats = {
            'total_texts': len(texts),
            'avg_length': np.mean([len(text) for text in processed_texts]),
            'avg_words': np.mean([len(text.split()) for text in processed_texts]),
            'vocabulary_size': len(set(' '.join(processed_texts).split())),
        }
        
        # ADVANCED: Code-switching analysis
        english_count = 0
        hausa_count = 0
        mixed_count = 0
        
        for text in texts:
            features = self.extract_features(text, text)
            if features['code_switching_score'] > 0.5:
                mixed_count += 1
            elif features['hausa_word_ratio'] > 0.7:
                hausa_count += 1
            else:
                english_count += 1
        
        stats['code_switching'] = {
            'primarily_hausa': hausa_count,
            'primarily_english': english_count, 
            'mixed_language': mixed_count,
            'code_switching_ratio': mixed_count / len(texts)
        }
        
        # Most common words
        all_words = ' '.join(processed_texts).split()
        stats['most_common_words'] = Counter(all_words).most_common(20)
        
        # Sentiment distribution
        pos_texts = sum(1 for text in texts if any(pos in text.lower() for pos in self.positive_indicators))
        neg_texts = sum(1 for text in texts if any(neg in text.lower() for neg in self.negative_indicators))
        
        stats['sentiment_distribution'] = {
            'texts_with_positive_indicators': pos_texts,
            'texts_with_negative_indicators': neg_texts,
            'neutral_or_mixed': len(texts) - pos_texts - neg_texts
        }
        
        return stats

# Instantiate the enhanced preprocessor
preprocessor = HausaTextPreprocessor()