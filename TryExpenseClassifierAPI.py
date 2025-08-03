import unicodedata
import sys
import re
import string
import pandas as pd
import torch
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to your model folder and number dictionary CSV using absolute paths
saved_model_path = os.path.join(current_dir, "CAMeLBERT_Classifier_SelfTrained_v1")
number_csv_path_for_predict = os.path.join(current_dir, "yemeni_numbers.csv")

print(f"Model path: {saved_model_path}")
print(f"Number dict path: {number_csv_path_for_predict}")


if os.path.exists(saved_model_path):
    print("the path is exist")
else:
    print("the path not found")
    

df = pd.read_csv(number_csv_path_for_predict)
df.head()

# Arabic Text Preprocessor Class
class ArabicTextPreprocessor:
    _EASTERN_ARABIC_NUMERALS = '٠١٢٣٤٥٦٧٨٩'
    _WESTERN_ARABIC_NUMERALS = '0123456789'
    _ARABIC_CHAR_MAP = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
        'ة': 'ه',
        'ى': 'ي'
    }
    _ARABIC_DIACRITICS_TATWEEL_REGEX = re.compile(r'[\u064B-\u0652\u0640]')
    _CHARS_TO_PRESERVE = '.-/'
    _ARABIC_PUNCTUATIONS_BASE = '`÷×؛<>_()*&^%][ـ،:"؟\'{}~¦+|!”…“–ـ«»'
    _ENGLISH_PUNCTUATIONS_BASE = string.punctuation
    _MULTI_WHITESPACE_REGEX = re.compile(r'\s+')
    _EMOJI_PATTERN = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F" u"\U0001F780-\U0001F7FF" u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF" u"\U0001FA70-\U0001FAFF" u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)

    def __init__(self):
        self.numeral_translation_table = None
        self.char_norm_translation_table = None
        self.punctuation_removal_table = None
        try:
            self.numeral_translation_table = str.maketrans(
                self._EASTERN_ARABIC_NUMERALS,
                self._WESTERN_ARABIC_NUMERALS
            )
        except Exception as e:
          pass
        try:
            self.char_norm_translation_table = str.maketrans(self._ARABIC_CHAR_MAP)
        except Exception as e:
          pass
        try:
            _english_punctuations_to_remove_str = ''.join(
                c for c in self._ENGLISH_PUNCTUATIONS_BASE if c not in self._CHARS_TO_PRESERVE
            )
            _punctuations_to_remove_str = self._ARABIC_PUNCTUATIONS_BASE + _english_punctuations_to_remove_str
            self.punctuation_removal_table = str.maketrans('', '', _punctuations_to_remove_str)
        except Exception as e:
          pass
    def _normalize_unicode(self, text: str, form: str = 'NFC') -> str:
        if not isinstance(text, str): return text
        try: return unicodedata.normalize(form, text)
        except Exception: return text

    def _remove_emojis(self, text:str) -> str:
        if not isinstance(text, str): return text
        try: return self._EMOJI_PATTERN.sub('', text)
        except Exception: return text
    def _normalize_arabic_chars(self, text: str) -> str:
        if self.char_norm_translation_table is None: return text
        if not isinstance(text, str): return text
        try: return text.translate(self.char_norm_translation_table)
        except Exception: return text
    def _standardize_numerals(self, text: str) -> str:
        if self.numeral_translation_table is None: return text
        if not isinstance(text, str): return text
        try: return text.translate(self.numeral_translation_table)
        except Exception: return text
    def _remove_diacritics_and_tatweel(self, text: str) -> str:
        if not isinstance(text, str): return text
        try: return self._ARABIC_DIACRITICS_TATWEEL_REGEX.sub('', text)
        except Exception: return text
    def _remove_punctuations(self, text: str) -> str:
        if self.punctuation_removal_table is None: return text
        if not isinstance(text, str): return text
        try: return text.translate(self.punctuation_removal_table)
        except Exception: return text
    def _lowercase_latin(self, text: str) -> str:
        if not isinstance(text, str): return text
        try: return text.lower()
        except Exception: return text
    def _normalize_whitespace(self, text: str) -> str:
        if not isinstance(text, str): return text
        try:
            text = text.strip()
            return self._MULTI_WHITESPACE_REGEX.sub(' ', text)
        except Exception: return text
    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        if self.char_norm_translation_table is None or \
           self.numeral_translation_table is None or \
           self.punctuation_removal_table is None:
             return text
        processed_text = self._normalize_unicode(text, 'NFC')
        processed_text = self._remove_emojis(processed_text)
        processed_text = self._normalize_arabic_chars(processed_text)
        processed_text = self._standardize_numerals(processed_text)
        processed_text = self._remove_diacritics_and_tatweel(processed_text)
        processed_text = self._remove_punctuations(processed_text)
        processed_text = self._lowercase_latin(processed_text)
        processed_text = self._normalize_whitespace(processed_text)
        return processed_text



# Utility Function: Load Number Dictionary
def load_number_dictionary(csv_path: str) -> dict:
    """Loads the textual number mapping from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'yemeni_textual_number' not in df.columns or 'numerical_value' not in df.columns:
            raise ValueError("CSV must contain 'yemeni_textual_number' and 'numerical_value' columns.")
        preprocessor_for_dict = ArabicTextPreprocessor()
        df['normalized_text'] = df['yemeni_textual_number'].apply(preprocessor_for_dict.preprocess)
        number_dict = pd.Series(df.numerical_value.values, index=df.normalized_text).astype(str).to_dict()
        number_dict.pop("", None)
        return number_dict
    except FileNotFoundError:
        print(f"Error: Number dictionary CSV not found at {csv_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading number dictionary: {e}", file=sys.stderr)
        return None



# Utility Function: Convert Textual Numbers
def convert_textual_numbers(text: str, number_map: dict) -> str:
    """Converts textual number phrases using a single regex pass, prioritizing longer matches."""
    if not number_map or not isinstance(text, str):
        return text
    sorted_keys = sorted(number_map.keys(), key=len, reverse=True)
    valid_escaped_keys = [re.escape(key) for key in sorted_keys if key]
    if not valid_escaped_keys:
         return text
    pattern_str = r'\b(' + '|'.join(valid_escaped_keys) + r')\b'
    try:
        pattern_regex = re.compile(pattern_str)
    except Exception as e:
        print(f"Error compiling regex for number conversion: {e}", file=sys.stderr)
        return text
    def replace_func(match):
        matched_text = match.group(1)
        return number_map.get(matched_text, matched_text)
    try:
        processed_text = pattern_regex.sub(replace_func, text)
        return processed_text
    except Exception as e:
        print(f"Error applying regex substitution for number conversion: {e}", file=sys.stderr)
        return text




# Financial Text Classifier Class (for loading and prediction)
class FinancialTextClassifier:
    def __init__(self, base_model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix", num_labels=None, id2label=None, label2id=None, preprocessor=None, number_dictionary=None):
        self.base_model_name = base_model_name
        self.tokenizer = None # Initialize as None, loaded with model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.preprocessor = preprocessor # Store preprocessor as instance attribute
        self.number_dictionary = number_dictionary # Store number_dictionary as instance attribute

    def load_model(self, model_path):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            mapping_path = f"{model_path}/category_mappings.json"
            if os.path.exists(mapping_path):
                 with open(mapping_path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                    self.num_labels = mappings.get('num_labels')
                    self.id2label = {int(k): v for k, v in mappings.get('id2label', {}).items()}
                    self.label2id = mappings.get('label2id')
                    if not all([self.num_labels, self.id2label, self.label2id]):
                         print("Warning: Loaded model but category mappings incomplete/missing.", file=sys.stderr)
            else:
                 print(f"Warning: category_mappings.json not found in {model_path}. Using model config labels.", file=sys.stderr)
                 try:
                    self.id2label = self.model.config.id2label
                    self.label2id = self.model.config.label2id
                    self.num_labels = self.model.config.num_labels
                 except AttributeError:
                    print("Error: Could not load labels from model config.", file=sys.stderr)
                    self.id2label = None
                    self.label2id = None
                    self.num_labels = None
            if self.num_labels != self.model.config.num_labels:
                 print(f"Warning: Mismatch num_labels ({self.num_labels}) vs config ({self.model.config.num_labels}). Using config.", file=sys.stderr)
                 self.num_labels = self.model.config.num_labels
            return True
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}", file=sys.stderr)
            self.model = None
            self.tokenizer = None
            return False

    def predict(self, text):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or Tokenizer not loaded.")
        if self.id2label is None:
             print("Warning: Label names map unavailable. Returning ID.", file=sys.stderr)

        # Use instance attributes for preprocessor and number_dictionary
        if self.preprocessor is None:
             raise RuntimeError("Preprocessor not initialized.")
        if self.number_dictionary is None:
             temp_num_dict = {}
        else:
             temp_num_dict = self.number_dictionary

        clean_text = self.preprocessor.preprocess(str(text))
        text_with_numbers = convert_textual_numbers(clean_text, temp_num_dict)

        inputs = self.tokenizer(
            text_with_numbers,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_id].item()

        category = self.id2label.get(pred_id, f"ID_{pred_id}") if self.id2label else pred_id
        return {'category': category, 'confidence': confidence}

    def batch_predict(self, texts):
        results = []
        # Use instance attributes for preprocessor and number_dictionary
        if self.preprocessor is None:
             print("Error: Preprocessor not initialized for batch prediction.", file=sys.stderr)
             for text in texts:
                 results.append({'text': text, 'category': 'PREDICTION_ERROR', 'confidence': 0.0})
             return results

        if self.number_dictionary is None:
             temp_num_dict = {}
        else:
             temp_num_dict = self.number_dictionary

        for text in texts:
            try:
                # Use instance attributes in predict method
                result = self.predict(text)
                result['text'] = text
                results.append(result)
            except Exception as e:
                 print(f"Error predicting for text: '{text[:50]}...': {e}", file=sys.stderr)
                 results.append({'text': text, 'category': 'PREDICTION_ERROR', 'confidence': 0.0})
        return results


# --- Main Execution ---


if __name__ == "__main__":
    # Initialize preprocessor and number_dictionary
    try:
        preprocessor = ArabicTextPreprocessor()
    except Exception as e:
        print(f"Error initializing preprocessor: {e}", file=sys.stderr)
        preprocessor = None

    try:
        number_dictionary = load_number_dictionary(number_csv_path_for_predict)
        if number_dictionary is None:
            # Handle case where number dictionary loading failed (e.g., file not found)
            print("Warning: Could not load number dictionary. Textual number conversion will not be performed.", file=sys.stderr)
            number_dictionary = {} # Ensure it's a dictionary even if loading fails
    except Exception as e:
        print(f"Error loading number dictionary: {e}", file=sys.stderr)
        number_dictionary = {} # Ensure it's a dictionary even if loading fails

    # Pass preprocessor and number_dictionary to the FinancialTextClassifier during initialization
    classifier_loaded = FinancialTextClassifier(preprocessor=preprocessor, number_dictionary=number_dictionary)
    model_loaded_successfully = classifier_loaded.load_model(saved_model_path)

    if model_loaded_successfully and preprocessor is not None:
        print("Model loaded successfully. Ready for predictions.")

        # Example usage: Make predictions on new text data
        new_expenses = [
            "شليت لي قات بالف ريال يمني اليوم",
            "10000 عشاء بالف ريال ",
            "   دخل اضافي من تاجير شقة بالعمارة.. ٢٠٠ الف ريال شهريا",
            " وصلتني ارباح من اسهم بنك ابوظبي الاول. ١٠٠ درهم اماراتي. ",
            "ارباح وصلتني من مصنع هائل سعيد ",
            "غاز للبيت الحين بـ ٤٥٠٠ ! كان ارخص قبل فتره",
            " تعبئة رصيد بـ 10000 ",
            "دفعت ايجار البيت خمسين الف",
            " دفعت النت مقدم سنة .. ثلاثون الف ريال يمني دفع بنكي",
            "عبيت بترول للسيارة ب ٨٠٠٠",
            "استلمت راتب نوفمبر ١٥ الف ريال",
            " 100000 سددت الدين لصاحب البقالة ",
            "فاتورة كهرباء ٢٥٠٠",
            "10000 غداء",
            "غداء في مطعم السعيد بالف وميتين",
            "اشتريت جزمة جديدة بـ ٣ الف من السوق",
            "حولت زكاة المال ٥٠٠٠ ريال",
            "تحويل  عبر بنك كريمي 30000 مقابل دفع مشتريات  ",
            "مية دولار حق العشاء",
            "مواصلات باص ٥٠ ريال",
            "فاتورة النت ب 1000",
        ]

        predictions = classifier_loaded.batch_predict(new_expenses)

        print("\n--- Prediction Results ---")
        for result in predictions:
            original_text = result.get('text', 'N/A')
            predicted_cat = result.get('category', 'ERROR')
            confidence_score = result.get('confidence', 0.0) * 100
            print(f"Input Text:      '{original_text}'")
            print(f"Predicted Class: {predicted_cat}")
            print(f"Confidence:      {confidence_score:.2f}%")
            print("-" * 30)
    else:
        print("\nSkipping prediction test due to errors loading model or prerequisites.")