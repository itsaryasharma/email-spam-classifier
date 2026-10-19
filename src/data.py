"""Download and parse the two e-mail corpora.

    python -m src.data      # download (if needed) and build data/processed/emails.csv.gz

1. **Apache SpamAssassin public corpus**: raw e-mails (full headers + MIME).
   We keep only the Subject line and the readable body text.
2. **Enron-Spam (pre-processed)**: one text file per e-mail; the first line is
   "Subject: ...", the rest is the body (already lower-cased and stripped of
   headers and HTML by the dataset authors).

Why only subject + body? That is what a user can paste into the app. Headers
(routing servers, dates, mailing-list IDs) would let the model learn
shortcuts that do not exist in pasted text.

Licence note: neither corpus is published under an open licence (copyright in
the message text remains with the original senders), so raw and processed data
are NOT committed to the repository; they are downloaded on demand.
See docs/DATA_SOURCES.md.
"""
import email
import email.policy
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    ENRON_ARCHIVES,
    ENRON_BASE_URL,
    PROCESSED_PATH,
    RAW_DIR,
    SPAMASSASSIN_ARCHIVES,
    SPAMASSASSIN_BASE_URL,
)
from src.text import html_to_text


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        print(f"Downloading {path.name} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# SpamAssassin: raw MIME e-mails
# --------------------------------------------------------------------------- #
def _decode_part(part) -> str:
    """Decode one MIME part to str, tolerating broken or unknown charsets."""
    try:
        return part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "latin-1"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:  # unknown charset name such as "unknown-8bit"
            return payload.decode("latin-1", errors="replace")


def parse_email(raw: bytes) -> tuple[str, str]:
    """Return (subject, body_text) for one raw MIME e-mail."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    try:
        subject = str(msg.get("Subject", "") or "")
    except Exception:
        subject = ""
    plain, htmls = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(_decode_part(part))
        elif ctype == "text/html":
            htmls.append(html_to_text(_decode_part(part)))
    # Prefer the plain-text version; fall back to text extracted from HTML.
    body = "\n".join(plain) if any(p.strip() for p in plain) else "\n".join(htmls)
    return subject.strip(), body.strip()


def read_spamassassin(raw_dir: Path) -> list[dict]:
    rows = []
    for name, label in SPAMASSASSIN_ARCHIVES.items():
        path = _download(SPAMASSASSIN_BASE_URL + name, raw_dir / name)
        with tarfile.open(path, "r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.name.endswith("cmds"):
                    continue  # each archive also contains a "cmds" helper file
                subject, body = parse_email(tar.extractfile(member).read())
                rows.append({"corpus": "SpamAssassin", "source": member.name.split("/")[0],
                             "label": label, "subject": subject, "body": body})
    return rows


# --------------------------------------------------------------------------- #
# Enron-Spam: pre-processed text files
# --------------------------------------------------------------------------- #
def parse_enron_file(raw: bytes) -> tuple[str, str]:
    text = raw.decode("utf-8", errors="replace")
    first, _, rest = text.partition("\n")
    subject = first[len("Subject:"):].strip() if first.startswith("Subject:") else ""
    body = rest if first.startswith("Subject:") else text
    return subject, body.strip()


def read_enron(raw_dir: Path) -> list[dict]:
    rows = []
    for name in ENRON_ARCHIVES:
        path = _download(ENRON_BASE_URL + name, raw_dir / name)
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                parts = member.name.split("/")  # e.g. enron1/ham/0001.1999-12-10.farmer.ham.txt
                if not member.isfile() or len(parts) != 3 or parts[1] not in ("ham", "spam"):
                    continue  # skips Summary.txt
                subject, body = parse_enron_file(tar.extractfile(member).read())
                rows.append({"corpus": "Enron-Spam", "source": f"{parts[0]}/{parts[1]}",
                             "label": int(parts[1] == "spam"), "subject": subject, "body": body})
    return rows


# --------------------------------------------------------------------------- #
# Combined dataset
# --------------------------------------------------------------------------- #
def build_dataset(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Parse both corpora, drop empty and duplicate messages.

    Returns the cleaned DataFrame and a dictionary of cleaning statistics.
    """
    df = pd.DataFrame(read_spamassassin(raw_dir) + read_enron(raw_dir))
    df["text"] = ("Subject: " + df["subject"] + "\n\n" + df["body"]).str.strip()

    def counts(frame):
        return {f"{c} / {'spam' if l else 'legitimate'}": int(n)
                for (c, l), n in frame.groupby(["corpus", "label"]).size().items()}

    stats = {"parsed": int(len(df)), "parsed_by_corpus_label": counts(df)}
    empty = df["body"].str.strip().eq("") & df["subject"].str.strip().eq("")
    df = df[~empty]
    stats["dropped_empty"] = int(empty.sum())

    # Identical texts in both train and test would inflate test scores (leakage),
    # and identical texts with different labels would be contradictory.
    before = len(df)
    conflicting = df.groupby("text")["label"].nunique().gt(1).sum()
    df = df.drop_duplicates(subset="text", keep="first")
    stats["dropped_duplicates"] = int(before - len(df))
    stats["duplicate_texts_with_conflicting_labels"] = int(conflicting)
    df = df.reset_index(drop=True)
    stats["final"] = int(len(df))
    stats["final_by_corpus_label"] = counts(df)
    stats["final_by_label"] = {("spam" if k else "legitimate"): int(v)
                               for k, v in df["label"].value_counts().items()}
    return df, stats


def load_dataset() -> pd.DataFrame:
    """Load the processed dataset, building it first if necessary."""
    if not PROCESSED_PATH.exists():
        df, _ = build_dataset()
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False, compression="gzip")
    return pd.read_csv(PROCESSED_PATH, keep_default_na=False)


if __name__ == "__main__":
    data, info = build_dataset()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False, compression="gzip")
    for key, value in info.items():
        print(f"{key}: {value}")
