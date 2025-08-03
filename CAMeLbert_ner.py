# Setup and Installations
# !pip install numpy==1.26.4 -q

# !pip install transformers==4.48.3 datasets seqeval scikit-learn accelerate torch -q
# ensure transformer version
import transformers

print(transformers.__version__)
# Import Libraries and Configuration
import unicodedata
import sys
import re
import string
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
import torch
import numpy as np
from seqeval.metrics import classification_report
import os
import json
from google.colab import drive

# Mount Google Drive
from google.colab import drive

drive.mount("/content/drive")
# Define Paths and Constants
annotated_csv_path = "drive/MyDrive/datasets/full_dataset_with_ner_annotations.csv"

if os.path.exists(annotated_csv_path):
    print(f"File '{annotated_csv_path}' exists.")
else:
    print(f"File '{annotated_csv_path}' does not exist.")

number_csv_path = "drive/MyDrive/datasets/yemeni_numbers.csv"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

local_ner_model_path = "./fine_tuned_camelbert_ner_model"
drive_ner_model_save_base = "drive/MyDrive/models/CAMeLBERT_Expense_NER_v1_first30"

# Create a final path for saving to Drive
os.makedirs(drive_ner_model_save_base, exist_ok=True)
final_ner_model_on_drive_path = os.path.join(
    drive_ner_model_save_base, os.path.basename(local_ner_model_path)
)

df = pd.read_csv(annotated_csv_path)
df.head()

df.loc[:50, "text":"category"]


# Arabic Text Preprocessor Class
class ArabicTextPreprocessor:
    _EASTERN_ARABIC_NUMERALS = "٠١٢٣٤٥٦٧٨٩"
    _WESTERN_ARABIC_NUMERALS = "0123456789"
    _ARABIC_CHAR_MAP = {"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"}
    _ARABIC_DIACRITICS_TATWEEL_REGEX = re.compile(r"[\u064B-\u0652\u0640]")
    _CHARS_TO_PRESERVE = ".-/"
    _ARABIC_PUNCTUATIONS_BASE = "`÷×؛<>_()*&^%][ـ،:\"؟'{}~¦+|!”…“–ـ«»"
    _ENGLISH_PUNCTUATIONS_BASE = string.punctuation
    _MULTI_WHITESPACE_REGEX = re.compile(r"\s+")
    _EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    def __init__(self):
        self.numeral_translation_table = None
        self.char_norm_translation_table = None
        self.punctuation_removal_table = None
        try:
            self.numeral_translation_table = str.maketrans(
                self._EASTERN_ARABIC_NUMERALS, self._WESTERN_ARABIC_NUMERALS
            )
        except Exception as e:
            pass
        try:
            self.char_norm_translation_table = str.maketrans(self._ARABIC_CHAR_MAP)
        except Exception as e:
            pass
        try:
            _english_punctuations_to_remove_str = "".join(
                c
                for c in self._ENGLISH_PUNCTUATIONS_BASE
                if c not in self._CHARS_TO_PRESERVE
            )
            _punctuations_to_remove_str = (
                self._ARABIC_PUNCTUATIONS_BASE + _english_punctuations_to_remove_str
            )
            self.punctuation_removal_table = str.maketrans(
                "", "", _punctuations_to_remove_str
            )
        except Exception as e:
            pass

    def _normalize_unicode(self, text: str, form: str = "NFC") -> str:
        if not isinstance(text, str):
            return text
        try:
            return unicodedata.normalize(form, text)
        except Exception:
            return text

    def _remove_emojis(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        try:
            return self._EMOJI_PATTERN.sub("", text)
        except Exception:
            return text

    def _normalize_arabic_chars(self, text: str) -> str:
        if self.char_norm_translation_table is None:
            return text
        if not isinstance(text, str):
            return text
        try:
            return text.translate(self.char_norm_translation_table)
        except Exception:
            return text

    def _standardize_numerals(self, text: str) -> str:
        if self.numeral_translation_table is None:
            return text
        if not isinstance(text, str):
            return text
        try:
            return text.translate(self.numeral_translation_table)
        except Exception:
            return text

    def _remove_diacritics_and_tatweel(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        try:
            return self._ARABIC_DIACRITICS_TATWEEL_REGEX.sub("", text)
        except Exception:
            return text

    def _remove_punctuations(self, text: str) -> str:
        if self.punctuation_removal_table is None:
            return text
        if not isinstance(text, str):
            return text
        try:
            return text.translate(self.punctuation_removal_table)
        except Exception:
            return text

    def _lowercase_latin(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        try:
            return text.lower()
        except Exception:
            return text

    def _normalize_whitespace(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        try:
            text = text.strip()
            return self._MULTI_WHITESPACE_REGEX.sub(" ", text)
        except Exception:
            return text

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        if (
            self.char_norm_translation_table is None
            or self.numeral_translation_table is None
            or self.punctuation_removal_table is None
        ):
            return text
        processed_text = self._normalize_unicode(text, "NFC")
        processed_text = self._remove_emojis(processed_text)
        processed_text = self._normalize_arabic_chars(processed_text)
        processed_text = self._standardize_numerals(processed_text)
        processed_text = self._remove_diacritics_and_tatweel(processed_text)
        processed_text = self._remove_punctuations(processed_text)
        processed_text = self._lowercase_latin(processed_text)
        processed_text = self._normalize_whitespace(processed_text)
        return processed_text


# Utility Function: Load Number Dictionary


def load_number_dictionary(csv_path: str, preprocessor: ArabicTextPreprocessor) -> dict:
    """Loads the textual number mapping from a CSV and preprocesses the keys."""
    try:
        df = pd.read_csv(csv_path)
        if (
            "yemeni_textual_number" not in df.columns
            or "numerical_value" not in df.columns
        ):
            raise ValueError(
                "CSV must contain 'yemeni_textual_number' and 'numerical_value' columns."
            )

        df["normalized_text_key"] = df["yemeni_textual_number"].apply(
            lambda x: preprocessor.preprocess(str(x))
        )

        number_dict = (
            pd.Series(df.numerical_value.values, index=df.normalized_text_key)
            .astype(str)
            .to_dict()
        )
        number_dict.pop(
            "", None
        )  # Remove potential empty keys that might result from preprocessing
        return number_dict
    except Exception as e:
        print(
            f"Error loading or processing textual number dictionary from {csv_path}: {e}",
            file=sys.stderr,
        )
        return None


# Utility Function: Convert Textual Numbers


def convert_textual_numbers(text: str, number_map: dict) -> str:
    """Converts textual number phrases using a single regex pass."""
    if not number_map or not isinstance(text, str):
        return text
    sorted_keys = sorted(number_map.keys(), key=len, reverse=True)
    valid_escaped_keys = [re.escape(key) for key in sorted_keys if key]
    if not valid_escaped_keys:
        return text

    pattern = re.compile(r"\b(" + "|".join(valid_escaped_keys) + r")\b")
    replace_func = lambda match: number_map.get(match.group(1), match.group(1))

    try:
        return pattern.sub(replace_func, text)
    except Exception as e:
        print(f"Error during textual number conversion: {e}", file=sys.stderr)
        return text


########################### Load and Initialize ###########################
try:
    preprocessor = ArabicTextPreprocessor()
    number_dictionary = load_number_dictionary(number_csv_path, preprocessor)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Preprocessor, number dictionary, and tokenizer initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}", file=sys.stderr)
    # Set to None to prevent subsequent code from running with errors
    tokenizer = None

# ########################### Initialize variables ###########################
tokenized_datasets_ner = None
ner_label_list = None
label2id_ner = None
id2label_ner = None

############################# Load and Parse Annotated Data ###########################
if tokenizer:
    try:

        df = pd.read_csv(annotated_csv_path)
        df_ner = df[["tokens", "ner_tags"]].copy()
        df_ner.dropna(inplace=True)

        def parse_column_to_list(column_entry):
            if isinstance(column_entry, list):
                return column_entry
            if isinstance(column_entry, str):
                try:
                    return json.loads(column_entry.replace("'", '"'))
                except (json.JSONDecodeError, AttributeError):
                    return [item.strip() for item in column_entry.split(",")]
            return []

        df_ner["tokens_list"] = df_ner["tokens"].apply(parse_column_to_list)
        df_ner["ner_tags_list"] = df_ner["ner_tags"].apply(parse_column_to_list)
        df_filtered = df_ner[
            df_ner.apply(
                lambda r: len(r["tokens_list"]) == len(r["ner_tags_list"]), axis=1
            )
        ]
        raw_ner_dataset = Dataset.from_pandas(
            df_filtered[["tokens_list", "ner_tags_list"]]
        )
    except Exception as e:
        print(f"Failed to load or process NER dataset: {e}", file=sys.stderr)
        raw_ner_dataset = None

    if raw_ner_dataset:
        ner_label_list = sorted(
            list(
                set(
                    tag
                    for tag_list in raw_ner_dataset["ner_tags_list"]
                    for tag in tag_list
                )
            )
        )
        label2id_ner = {label: i for i, label in enumerate(ner_label_list)}
        id2label_ner = {i: label for label, i in label2id_ner.items()}
        num_ner_labels = len(ner_label_list)

        def tokenize_and_align_labels_smart(examples):
            tokenized_inputs = tokenizer(
                examples["tokens_list"], truncation=True, is_split_into_words=True
            )
            labels = []
            for i, label in enumerate(examples["ner_tags_list"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label2id_ner[label[word_idx]])
                    else:

                        original_label = label[word_idx]
                        if original_label.startswith("B-"):

                            i_label = "I-" + original_label[2:]
                            label_ids.append(
                                label2id_ner.get(i_label, label2id_ner[original_label])
                            )
                        else:

                            label_ids.append(label2id_ner[original_label])
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        # --- 5. Apply the Final Processing to the Dataset ---
        tokenized_datasets_ner = raw_ner_dataset.map(
            tokenize_and_align_labels_smart, batched=True
        )
        tokenized_datasets_ner = tokenized_datasets_ner.remove_columns(
            ["tokens_list", "ner_tags_list"]
        )
        tokenized_datasets_ner.set_format("torch")
        print("\nNER dataset successfully prepared with SMART alignment.")


## Test CAMeLBERT Tokenizer Behavior on Numbers
# ---
#  We got an issue: at first fine-tuning .
# ---

# # * the ouput of the model for numbers more than 3 digits ... for the amount entity . the modle fragments these amount into 'tow' separated toknization .. this behavior by default in the model:  *e.g :*
# - 4500 -> ['450', '##0']. Fragmented.
# - 8000 -> ['800', '##0']. Fragmented.
# -10000 -> ['1000', '##0']. Fragmented.
# - 100000 -> ['1000', '##00']. Fragmented.


if "tokenizer" in locals() and tokenizer is not None:
    print("--- Testing CAMeLBERT Tokenizer Behavior on Various Numbers ---\n")

    test_cases = [
        "100",
        "999",
        "1000",
        "4500",
        "8000",
        "10000",
        "50000",
        "100000",
        "ريال",
        "الف",
        "خمسين الف",
        " بـ 10000 ريال",
    ]

    for text in test_cases:
        # Tokenize the text
        tokens = tokenizer.tokenize(text)

        # Get the corresponding numerical IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        print(f"Original Text: '{text}'")
        print(f"  -> Tokens: {tokens}")
        print(f"  -> IDs:    {input_ids}\n")
        print("-" * 40)

else:
    print(
        "Tokenizer not initialized. Please run the cell that defines the 'tokenizer' object first."
    )
# ---
# NER Fine-tuning phase
# ---
## NER Model Class Definition

from transformers import DataCollatorForTokenClassification
from datasets import concatenate_datasets
from seqeval.metrics import classification_report


class FinancialEntityRecognizer:
    def __init__(self, base_model_name, num_labels, id2label, label2id):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        ########### verifying data before training ############
        # print("--- FinancialEntityRecognizer Initialized With: ---")
        # print(f"num_labels: {self.num_labels}")
        # print(f"id2label: {self.id2label}")
        # print("-------------------------------------------------")

        self.model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        report = classification_report(
            true_labels, true_predictions, output_dict=True, zero_division=0
        )

        return {
            "precision": report["micro avg"]["precision"],
            "recall": report["micro avg"]["recall"],
            "f1-score": report["micro avg"]["f1-score"],
        }

    def fine_tune_model(
        self,
        train_dataset,
        eval_dataset,
        output_dir,
        epochs=10,
        batch_size=8,
        learning_rate=3e-5,
    ):
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=200,
            logging_strategy="steps",
            logging_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1-score",
            greater_is_better=True,
            fp16=True if self.device.type == "cuda" else False,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("--- Starting NER Model Fine-Tuning ---")
        trainer.train()

        print("\n--- Evaluating Final NER Model ---")
        metrics = trainer.evaluate()

        print(f"\n--- Final NER Evaluation Metrics ---")
        for key, value in metrics.items():
            if key.startswith("eval_"):
                print(f"  {key.replace('eval_', '')}: {value:.4f}")

        print(f"\n--- Saving Final NER Model to {output_dir} ---")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        with open(
            os.path.join(output_dir, "ner_mappings.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"id2label": self.id2label, "label2id": self.label2id},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Trigger self-training if F1 score is below a threshold
        if metrics.get("eval_f1-score", 0) < 0.90:
            print("\nF1-score below 0.90, initiating self-training...")
            self.self_training(train_dataset, eval_dataset, output_dir)

        return metrics

    def self_training(self, current_train_dataset, eval_dataset, output_dir):
        print("Loading best model from initial fine-tuning for self-training...")
        model_for_pseudo = AutoModelForTokenClassification.from_pretrained(output_dir)
        model_for_pseudo.to(self.device)
        model_for_pseudo.eval()

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        if hasattr(eval_dataset, "set_format"):
            eval_dataset.set_format(
                "torch", columns=["input_ids", "attention_mask", "labels"]
            )

        eval_loader = DataLoader(
            eval_dataset, batch_size=16, collate_fn=data_collator, shuffle=False
        )

        new_data_list = []
        confidence_threshold = 0.90  # Confidence per-token

        with torch.no_grad():
            for batch in eval_loader:
                batch_labels = batch.pop(
                    "labels", None
                )  # Keep original labels for comparison if needed
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = model_for_pseudo(**batch).logits
                probs = torch.softmax(logits, dim=-1)
                confidence, preds = torch.max(probs, dim=-1)

                for i in range(batch["input_ids"].size(0)):
                    # Check if all token predictions in the sequence meet the confidence threshold
                    # We ignore padding tokens where the original label is -100
                    original_labels_for_seq = batch_labels[i]
                    is_confident = True
                    for j in range(len(original_labels_for_seq)):
                        if (
                            original_labels_for_seq[j] != -100
                            and confidence[i][j].item() < confidence_threshold
                        ):
                            is_confident = False
                            break

                    if is_confident:
                        new_data_list.append(
                            {
                                "input_ids": batch["input_ids"][i].cpu().tolist(),
                                "attention_mask": batch["attention_mask"][i]
                                .cpu()
                                .tolist(),
                                "labels": preds[i].cpu().tolist(),
                            }
                        )

        if not new_data_list:
            print("No high-confidence sequences found for self-training. Stopping.")
            return

        print(
            f"Added {len(new_data_list)} new high-confidence sequences via self-training."
        )

        pseudo_dataset = Dataset.from_list(new_data_list)
        pseudo_dataset.set_format("torch")

        # Combine original and pseudo-labeled data
        print("Combining original training data and pseudo-labeled data...")
        combined_train_dataset = concatenate_datasets(
            [current_train_dataset, pseudo_dataset]
        )
        print(f"Combined dataset size: {len(combined_train_dataset)}")

        # Re-initialize the model from the best checkpoint to continue training
        self.model = AutoModelForTokenClassification.from_pretrained(output_dir)
        self.model.to(self.device)

        self_training_output_dir = f"{output_dir}/self_trained"
        st_args = TrainingArguments(
            output_dir=self_training_output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=2e-5,  # Slightly lower LR
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1-score",
            greater_is_better=True,
            fp16=True if self.device.type == "cuda" else False,
            report_to="none",
            save_total_limit=1,
        )

        st_trainer = Trainer(
            model=self.model,
            args=st_args,
            train_dataset=combined_train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("\n--- Starting Self-Training Phase for NER ---")
        st_trainer.train()

        st_metrics = st_trainer.evaluate()
        print(f"\n--- NER Metrics After Self-Training ---")
        for k, v in st_metrics.items():
            if k.startswith("eval_"):
                print(f"  {k.replace('eval_', '')}: {v:.4f}")

        print(f"\n--- Saving Self-Trained NER Model to {self_training_output_dir} ---")
        st_trainer.save_model(self_training_output_dir)
        self.tokenizer.save_pretrained(self_training_output_dir)
        self.model = st_trainer.model  # Update to the final, best model

    def predict(self, text, preprocessor, number_map):
        # 1. Preprocess the raw input text using the preprocessor
        clean_text = preprocessor.preprocess(str(text))

        # 2. Tokenize the clean text
        inputs = self.tokenizer(clean_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 3. Get model predictions (logits)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # 4. Get the predicted label IDs
        predictions = torch.argmax(logits, dim=2)

        # 5. Group tokens and their predicted IDs together
        results = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        for token, prediction_id in zip(tokens, predictions[0]):
            if token in [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ]:
                continue
            results.append({"token": token, "tag": self.id2label[prediction_id.item()]})

        # 6. Group consecutive tokens with the same entity type (B-TAG, I-TAG)
        final_entities = []
        current_entity_tokens = []
        current_entity_tag = None

        for res in results:
            token = res["token"]
            tag = res["tag"]

            if tag.startswith("B-"):

                if current_entity_tag:
                    word = self.tokenizer.convert_tokens_to_string(
                        current_entity_tokens
                    )
                    final_entities.append({"entity": current_entity_tag, "word": word})

                current_entity_tokens = [token]
                current_entity_tag = tag[2:]

            elif tag.startswith("I-") and current_entity_tag == tag[2:]:
                # Continue the current entity
                current_entity_tokens.append(token)

            else:
                if current_entity_tag:
                    word = self.tokenizer.convert_tokens_to_string(
                        current_entity_tokens
                    )
                    final_entities.append({"entity": current_entity_tag, "word": word})

                # Reset
                current_entity_tokens = []
                current_entity_tag = None

        # Add the last entity if it exists after the loop
        if current_entity_tag:
            word = self.tokenizer.convert_tokens_to_string(current_entity_tokens)
            final_entities.append({"entity": current_entity_tag, "word": word})

        return final_entities


## NER Training and Evaluation Execution
from transformers import EarlyStoppingCallback


def train_and_evaluate_ner_model(tokenized_ds, lbl2id, id2lbl, num_lbls):

    if isinstance(tokenized_ds, Dataset):
        train_test_split_ds = tokenized_ds.train_test_split(test_size=0.15, seed=42)
        train_dataset = train_test_split_ds["train"]
        eval_dataset = train_test_split_ds["test"]
    else:  # Assumes DatasetDict
        train_dataset = tokenized_ds["train"]
        eval_dataset = tokenized_ds.get("validation") or tokenized_ds.get("test")
        if not eval_dataset:
            train_test_split_ds = tokenized_ds["train"].train_test_split(
                test_size=0.15, seed=42
            )
            train_dataset = train_test_split_ds["train"]
            eval_dataset = train_test_split_ds["test"]

    recognizer = FinancialEntityRecognizer(
        base_model_name=MODEL_NAME,
        num_labels=num_lbls,
        id2label=id2lbl,
        label2id=lbl2id,
    )

    metrics = recognizer.fine_tune_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=local_ner_model_path,
        epochs=10,
        batch_size=8,
        learning_rate=3e-5,
    )

    print("\n--- Testing Trained NER Model ---")
    test_examples = [
        "دفعت ايجار البيت خمسين الف ريال",
        "حق القات اليوم الف ريال يمني",
        "تذكرة طيران الى جدة ب ٥٠ الف من الخطوط اليمنية",
        "استلمت حوالة امس بمبلغ مية دولار امريكي كاش",
        "فاتورة النت بتاريخ 2024-02-15",
    ]

    for text in test_examples:
        try:
            entities = recognizer.predict(text, preprocessor, number_dictionary)
            print(f"Text: {text}")
            print(f"Predicted Entities: {entities}\n")
        except Exception as e:
            print(f"Error predicting for text: '{text}'. Error: {e}", file=sys.stderr)

    return recognizer, metrics


## Main Entery Execution ----> بسم الله

if __name__ == "__main__":
    if "prepared_ner_data" in locals() and prepared_ner_data is not None:

        # Verifying Data  befoer training
        # print("--- Verifying Data Before Training ---")
        # print(f"Number of unique labels (num_ner_labels): {prepared_ner_data['num_labels']}")
        # print(f"Label to ID mapping (label2id_ner): {prepared_ner_data['label2id']}")
        # print(f"ID to Label mapping (id2label_ner): {prepared_ner_data['id2label']}")
        # print("------------------------------------")

        # # Check if the key '8' is actually missi
        # if 8 not in prepared_ner_data['id2label']:
        #     print("CRITICAL ERROR: The key '8' is MISSING from the id2label dictionary.")

        print("Starting NER model training and evaluation...")

        trained_recognizer, final_metrics = train_and_evaluate_ner_model(
            tokenized_ds=prepared_ner_data["dataset"],
            lbl2id=prepared_ner_data["label2id"],
            id2lbl=prepared_ner_data["id2label"],
            num_lbls=prepared_ner_data["num_labels"],
        )

        if trained_recognizer:
            print("\n--- NER Training and Evaluation Finished ---")
        else:
            print("\n--- NER Training and Evaluation Failed ---")

    else:
        print("\nError: Required data object 'prepared_ner_data' not found.")
        print("Ensure the NER data preparation cell has been run successfully.")
annotated_df = pd.read_csv(annotated_csv_path, encoding="utf-8")
display(annotated_df.head(15))
# Save Model and Tokenizer
import shutil
import os


drive_ner_model_save_final = final_ner_model_on_drive_path

print(
    f"Attempting to copy the trained NER model from '{local_ner_model_path}' to '{drive_ner_model_save_final}'..."
)

try:

    if os.path.exists(local_ner_model_path):
        # Remove the destination directory if it already exists to avoid FileExistsError
        if os.path.exists(drive_ner_model_save_final):
            print(
                f"Destination path '{drive_ner_model_save_final}' already exists. Removing it."
            )
            shutil.rmtree(drive_ner_model_save_final)

        shutil.copytree(local_ner_model_path, drive_ner_model_save_final)
        print(
            f"NER model successfully copied to Google Drive: {drive_ner_model_save_final}"
        )
    else:
        print(f"Error: Source model directory '{local_ner_model_path}' not found.")
except Exception as e:
    print(f"An error occurred during copying the NER model: {e}")
# Load Fine-tuned Model and Test
import os
import json
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)  # Import for NER model


saved_ner_model_path = final_ner_model_on_drive_path  #


ner_mappings_path = os.path.join(saved_ner_model_path, "ner_mappings.json")
id2label_loaded = None
label2id_loaded = None

try:
    with open(ner_mappings_path, "r", encoding="utf-8") as f:
        ner_mappings = json.load(f)
        # Ensure keys are integers for id2label if they were saved as strings
        id2label_loaded = {int(k): v for k, v in ner_mappings["id2label"].items()}
        label2id_loaded = ner_mappings["label2id"]
    print("NER mappings loaded successfully.")
except FileNotFoundError:
    print(f"Error: NER mappings file not found at {ner_mappings_path}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {ner_mappings_path}")
except Exception as e:
    print(f"An error occurred loading NER mappings: {e}")

# Determine the number of labels from the loaded mappings
num_ner_labels_loaded = len(id2label_loaded) if id2label_loaded else None


# Load the tokenizer and the fine-tuned NER model
tokenizer_loaded = None
model_loaded_successfully = False
loaded_recognizer = None

if (
    os.path.exists(saved_ner_model_path)
    and id2label_loaded is not None
    and label2id_loaded is not None
    and num_ner_labels_loaded is not None
):
    try:
        # Load the tokenizer
        tokenizer_loaded = AutoTokenizer.from_pretrained(saved_ner_model_path)

        # Load the model using the correct class for token classification
        # Pass the loaded mappings and num_labels using the correct parameter names
        loaded_recognizer = FinancialEntityRecognizer(
            base_model_name=saved_ner_model_path,  # Load from the saved path
            num_labels=num_ner_labels_loaded,
            id2label=id2label_loaded,  # Use the correct parameter name
            label2id=label2id_loaded,  # Use the correct parameter name
        )
        # The FinancialEntityRecognizer constructor loads the model from base_model_name
        model_loaded_successfully = True
        print("NER model and tokenizer loaded successfully.")

    except Exception as e:
        print(f"Error loading the NER model or tokenizer: {e}")
        model_loaded_successfully = False
        loaded_recognizer = None
        tokenizer_loaded = None
else:
    print(
        f"Skipping model loading: Saved model path not found ({saved_ner_model_path}), or mappings failed to load."
    )


# Test the loaded NER model with new examples
if (
    model_loaded_successfully
    and loaded_recognizer
    and preprocessor is not None
    and number_dictionary is not None
):
    print("\n--- Testing Loaded NER Model ---")
    new_expenses = [
        "   شليت لي قات مليون ريال يمني اليوم من المقوت",
        "1000 عشاء بالف ريال ",
        " عشاء بالف 1000 ريال ",
        "دخل اضافي من تاجير شقة بالعمارة.. ٢٠٠ الف ريال شهريا",
        "وصلتني ارباح من اسهم بنك ابوظبي الاول. ١٠٠ درهم اماراتي. ",
        "ارباح وصلتني من مصنع هائل سعيد ",
        "غاز للبيت الحين بـ ٤٥٠٠ ! كان ارخص قبل فتره",
        "تعبئة رصيد بـ 10000 ",
        "دفعت ايجار البيت خمسين الف",
        "دفعت النت مقدم سنة .. ثلاثون الف ريال يمني دفع بنكي",
        "عبيت بترول للسيارة ب ٨٠٠٠",
        "استلمت راتب نوفمبر ١٥ الف ريال",
        "100000 سددت الدين لصاحب البقالة ",
        "فاتورة كهرباء ٢٥٠٠",
        "10000 غداء",
        "غداء في مطعم السعيد بالف وميتين",
        "اشتريت جزمة جديدة بـ ٣ الف من السوق",
        "حولت زكاة المال ٥٠٠٠ ريال",
        "تحويل عبر بنك كريمي 30000 مقابل دفع مشتريات ",
        "مية دولار حق العشاء",
        "مواصلات باص ٥٠ ريال",
        "اديت لاخي الصغير مصروف المدرسة ميتين ريال",
        "فاتورة النت ب 1000",
    ]

    predictions = []
    for text in new_expenses:
        try:
            # Use the predict method of the loaded FinancialEntityRecognizer instance
            entities = loaded_recognizer.predict(text, preprocessor, number_dictionary)
            predictions.append({"text": text, "entities": entities})
        except Exception as e:
            print(f"Error predicting for text: '{text}'. Error: {e}", file=sys.stderr)
            predictions.append({"text": text, "entities": [], "error": str(e)})

    # Print results neatly
    for result in predictions:
        original_text = result.get("text", "N/A")
        predicted_entities = result.get("entities", [])
        error_msg = result.get("error", None)

        print(f"Input Text:         '{original_text}'")
        if error_msg:
            print(f"Prediction Error:   {error_msg}")
        else:
            print(f"Predicted Entities: {predicted_entities}")
        print("-" * 30)

else:
    print(
        "\nSkipping prediction test due to errors loading model, tokenizer, preprocessor, or number dictionary."
    )
