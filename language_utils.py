import re
import unicodedata

ARABIC_RANGES = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
URDU_MARKERS = set("ٹڈڑںھپچژگے")
URDU_WORDS = set("""
ہے ہیں میں کیا کا کی کے سے کو اور یہ وہ اس ان ایک بھی نہیں نہی رخصت چھٹی
چھٹیوں سالانہ پالیسی تنخواہ ملازم ملازمت کام قابل منتقل اقامہ
""".split())
ARABIC_WORDS = set("""
هل ما ماذا كيف من في على عن هو هي هذا هذه ذلك التي الذي الإقامة قابلة للنقل
سياسة إجازة الإجازات السنوية راتب موظف عمل
""".split())
ROMAN_URDU = set("""
hai hain kya ka ki ke ko se mein mai main mujhe mera meri mere hum ham tum
aap ye yeh woh wo aur par pe nahi nahin kyun kyu kyunke kaise kis kab kahan
kitna kitni kitne chutti chuttiyon chuttiyan policy salary iqama transferable
batao btao iska iski isko uska uski usko
""".split())

def normalize_text(text):
    return unicodedata.normalize("NFKC", str(text or "")).strip()

def contains_arabic_script(text):
    return bool(re.search(f"[{ARABIC_RANGES}]", normalize_text(text)))

def _tokens(text):
    return re.findall(
        r"[A-Za-z']+|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+",
        normalize_text(text).lower(),
    )

def detect_language(text):
    text = normalize_text(text)
    if not text:
        return "English"

    tokens = _tokens(text)
    arabic_chars = re.findall(f"[{ARABIC_RANGES}]", text)

    if arabic_chars:
        urdu_score = sum(1 for ch in arabic_chars if ch in URDU_MARKERS)
        words = set(tokens)
        urdu_score += 2 * len(words & URDU_WORDS)
        arabic_score = 2 * len(words & ARABIC_WORDS)
        return "Urdu" if urdu_score >= max(1, arabic_score) else "Arabic"

    lower = text.lower()
    roman_hits = sum(1 for token in tokens if token in ROMAN_URDU)
    if roman_hits >= 2 or re.search(
        r"\b(meri|mera|mere|mujhe|mujhy|hai|hain|kya|ka|ki|ke|ko|mein|yeh|woh|chutti|chuttiyon|kyun|kaise|kitni|kitna)\b",
        lower,
    ):
        return "Roman Urdu"

    return "English"

def language_instruction(language):
    if language == "Urdu":
        return (
            "Respond ONLY in natural Pakistani Urdu using Urdu script. "
            "Do not switch to Arabic, Chinese, Persian, or English sentences. "
            "Technical terms and file names may remain in English when necessary."
        )
    if language == "Roman Urdu":
        return (
            "Respond ONLY in natural Roman Urdu written with Latin letters. "
            "Do NOT use Urdu/Arabic script, Chinese, or Arabic. "
            "Keep technical terms and file names in English when useful."
        )
    if language == "Arabic":
        return (
            "Respond ONLY in clear Modern Standard Arabic. "
            "Do not switch to Urdu, Chinese, or English sentences. "
            "Technical terms and file names may remain in English when necessary."
        )
    return "Respond ONLY in clear English. Do not switch languages."

def is_wrong_script(text, language):
    text = normalize_text(text)
    if not text:
        return True
    if re.search(r"[\u4E00-\u9FFF]", text):
        return True
    if language == "Roman Urdu":
        arabic_count = len(re.findall(f"[{ARABIC_RANGES}]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))
        return arabic_count > max(2, latin_count * 0.08)
    return False
