import unicodedata, sys, re, string


class ArabicTextPreprocessor:
    # This is the complete and final version of the class
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
        self.numeral_translation_table = str.maketrans(
            self._EASTERN_ARABIC_NUMERALS, self._WESTERN_ARABIC_NUMERALS
        )
        self.char_norm_translation_table = str.maketrans(self._ARABIC_CHAR_MAP)
        _english_punctuations_to_remove = "".join(
            c
            for c in self._ENGLISH_PUNCTUATIONS_BASE
            if c not in self._CHARS_TO_PRESERVE
        )
        _punctuations_to_remove = (
            self._ARABIC_PUNCTUATIONS_BASE + _english_punctuations_to_remove
        )
        self.punctuation_removal_table = str.maketrans("", "", _punctuations_to_remove)

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        processed_text = unicodedata.normalize("NFC", text)
        processed_text = self._EMOJI_PATTERN.sub("", processed_text)
        processed_text = processed_text.translate(self.char_norm_translation_table)
        processed_text = processed_text.translate(self.numeral_translation_table)
        processed_text = self._ARABIC_DIACRITICS_TATWEEL_REGEX.sub("", processed_text)
        processed_text = processed_text.translate(self.punctuation_removal_table)
        processed_text = processed_text.lower()
        processed_text = self._MULTI_WHITESPACE_REGEX.sub(" ", processed_text).strip()
        return processed_text
