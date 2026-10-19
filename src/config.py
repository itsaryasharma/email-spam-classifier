"""Central configuration: paths, dataset sources and fixed settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_PATH = DATA_DIR / "processed" / "emails.csv.gz"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "spam_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Corpus 1: Apache SpamAssassin public corpus (2002-2005), raw RFC-822 e-mails.
# Label 1 = spam, 0 = legitimate ("ham").
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_ARCHIVES = {
    "20030228_easy_ham.tar.bz2": 0,
    "20030228_easy_ham_2.tar.bz2": 0,
    "20030228_hard_ham.tar.bz2": 0,
    "20030228_spam.tar.bz2": 1,
    "20050311_spam_2.tar.bz2": 1,
}

# Corpus 2: Enron-Spam (Metsis, Androutsopoulos & Paliouras, CEAS 2006),
# pre-processed form. Ham = real business e-mail of six Enron employees.
ENRON_BASE_URL = "https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/"
ENRON_ARCHIVES = [f"enron{i}.tar.gz" for i in range(1, 7)]

LABELS = {0: "legitimate", 1: "spam"}

# Words that identify which mailbox/corpus an e-mail came from rather than
# whether it is spam: the company name "enron", Enron's internal mail routing
# codes "ect"/"hou", its head-office city, the names of the six Enron mailbox
# owners, and SpamAssassin's anonymisation domain. Left in, they let the model
# learn "mentions Vince => legitimate", a shortcut that is useless on anyone
# else's e-mail.
CORPUS_IDENTITY_TERMS = [
    "enron", "ect", "hou", "houston",
    "vince", "kaminski", "louise", "sally", "beck", "daren", "michelle", "lokay",
    "spamassassin", "taint",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# Parallel workers for the grid search. Each worker builds its own bigram
# vocabulary over ~23,000 e-mails; using every core (n_jobs=-1, 12 workers on
# the development laptop) caused a MemoryError on a 16 GB machine. Changing
# this affects only speed, never the results.
N_JOBS = 4
DEFAULT_THRESHOLD = 0.5

# Input limits for the app.
MIN_WORDS_RELIABLE = 5        # fewer words -> warn that the prediction is unreliable
MAX_CHARS = 50_000            # longer input is truncated
