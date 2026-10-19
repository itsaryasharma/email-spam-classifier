"""Text normalisation used inside the TF-IDF vectoriser.

The same function runs at training time and at prediction time, because it
is stored inside the saved scikit-learn Pipeline. Replacing URLs, e-mail
addresses, money symbols and numbers with placeholder tokens lets the model
learn "this message contains a link" instead of memorising specific links.
"""
import html
import re

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
MONEY_RE = re.compile(r"[$£€]")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")

# The Enron-Spam files were tokenised by their authors, so links and addresses
# appear with spaces around punctuation: "http : / / www . site . com / x" and
# "vince . kaminski @ enron . com". Without these patterns the two corpora would
# encode links differently, and "contains a link" would become a corpus signal.
_URL_TAIL = r"(?:\s*[./?=&;%~+#:-]+\s*[\w-]*)*"
SPACED_URL_RE = re.compile(r"(?:https?\s*:\s*/\s*/|www\s*\.)\s*[\w-]+" + _URL_TAIL, re.IGNORECASE)
SPACED_EMAIL_RE = re.compile(r"[\w.+-]+\s*@\s*[\w-]+(?:\s*\.\s*[\w-]+)+")

PLACEHOLDERS = {
    "urltoken": "a web link",
    "emailtoken": "an e-mail address",
    "moneytoken": "a currency symbol",
    "numtoken": "a number",
}

# Only alphabetic tokens of length >= 2 (placeholders are alphabetic too).
TOKEN_PATTERN = r"(?u)\b[a-z][a-z]+\b"


def normalize(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = SPACED_URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = SPACED_EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    return text


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BREAK_RE = re.compile(r"<\s*(br|/p|/div|/tr|/li)[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")


def html_to_text(markup: str) -> str:
    """Very small HTML-to-text converter (good enough for spam bodies)."""
    text = _SCRIPT_STYLE_RE.sub(" ", markup)
    text = _BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _SPACE_RE.sub(" ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()
