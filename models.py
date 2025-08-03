
import torch, json, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch.nn.functional as F

class ExpenseClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        with open(os.path.join(model_path, "category_mappings.json"), 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            self.id2label = {int(k): v for k, v in mappings['id2label'].items()}

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            pred_id = torch.argmax(probs).item()
            confidence = probs[0, pred_id].item()

        return {'category': self.id2label[pred_id], 'confidence': confidence}

class EntityRecognizer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        results = []
        current_entity_tokens = []
        current_entity_tag = None

        for token, pred_id in zip(tokens, predictions[0]):
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue

            tag = self.id2label[pred_id.item()]

            if tag.startswith("B-"):
                if current_entity_tag:
                    word = self.tokenizer.convert_tokens_to_string(current_entity_tokens)
                    results.append({"entity": current_entity_tag, "word": word})
                current_entity_tokens = [token]
                current_entity_tag = tag[2:]
            elif tag.startswith("I-") and current_entity_tag == tag[2:]:
                current_entity_tokens.append(token)
            else:
                if current_entity_tag:
                    word = self.tokenizer.convert_tokens_to_string(current_entity_tokens)
                    results.append({"entity": current_entity_tag, "word": word})
                current_entity_tokens = []
                current_entity_tag = None

        if current_entity_tag:
            word = self.tokenizer.convert_tokens_to_string(current_entity_tokens)
            results.append({"entity": current_entity_tag, "word": word})

        return results
