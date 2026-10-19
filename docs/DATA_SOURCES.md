# Data Sources: Email Spam Classifier

## 1. Apache SpamAssassin public corpus

| | |
|---|---|
| URL | https://spamassassin.apache.org/old/publiccorpus/ |
| Files used | `20030228_easy_ham.tar.bz2`, `20030228_easy_ham_2.tar.bz2`, `20030228_hard_ham.tar.bz2`, `20030228_spam.tar.bz2`, `20050311_spam_2.tar.bz2` |
| Format | One raw RFC-822 e-mail per file (full headers + MIME body) |
| Content | Ham: mostly technical mailing lists and newsletters (2002). Spam: received by the maintainers |
| Terms | The corpus readme states the messages "were posted to public fora, were sent to me in the knowledge that they may be made public, ..." and "**Copyright for the text in the messages remains with the original senders.**" It also asks users **not** to send the messages into live e-mail systems. No open licence is given |

## 2. Enron-Spam (pre-processed)

| | |
|---|---|
| URL | https://www2.aueb.gr/users/ion/data/enron-spam/ (files `preprocessed/enron1.tar.gz` … `enron6.tar.gz`) |
| Citation | V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with Naive Bayes – Which Naive Bayes?", *Proc. 3rd Conference on Email and Anti-Spam (CEAS 2006)*, Mountain View, CA, USA, 2006 |
| Format | One text file per message; first line `Subject: ...`; already lower-cased and tokenised (spaces around punctuation) by the authors |
| Content | Ham: business e-mail of six Enron employees (released publicly during the Enron investigation). Spam: several spam traps and personal collections |
| Terms | Distributed by the authors for research; no explicit licence. Cite the paper above |

## Why the data is not in the repository
Neither source grants an open redistribution licence, and copyright in each message remains with its
sender. The repository therefore contains only the **code to download and process** the data and the
**trained model**. `data/` is listed in `.gitignore`.

## Integrity (SHA-256 of the files downloaded on 19 Oct 2026)
```
2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438  20030228_easy_ham.tar.bz2
b4bd3dc5ae5b40f38e99a0e41ad7d16b428b56e818e3a6ffa07330d62004dd38  20030228_easy_ham_2.tar.bz2
ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9  20030228_hard_ham.tar.bz2
c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743  20030228_spam.tar.bz2
44280a0e28bf7645b2279e8e42659271ad82c6f40bf8d364dbda7f1df344a765  20050311_spam_2.tar.bz2
3d6a1fcced6c30b1701a3fd04c6b912d7046208f6ef6f4fdcf022f64bdc6c32f  enron1.tar.gz
9b5bf5c4844eb4ecead52de8c676ff8761c2d7037fdc175851c88ef52c827315  enron2.tar.gz
9321ca1c195b45b6f0fb795963a8020ff317e7ebac2c102205272f0b91a1ac0b  enron3.tar.gz
61be71c59d7e9fdf99df63b814ddbeb18e0c9e8a69ce691d8e4e8127ed3588aa  enron4.tar.gz
d7b839c2a7071b6d52f1d76be8d05e66733d872614475fad7522740d7c60540d  enron5.tar.gz
283f70c45a7d819e0b988536fbb8e21cc17c17a22998403c2255d2c38cbfc5a0  enron6.tar.gz
```

## Processing summary (actual counts)

| Stage | Count |
|---|---|
| Messages parsed | 39,762 (SpamAssassin 6,046 · Enron 33,716) |
| Empty messages removed | 51 |
| Exact duplicate texts removed | 3,328 (none had conflicting labels) |
| **Final** | **36,383** (20,019 legitimate · 16,364 spam) |
| Train / test split | 29,106 / 7,277 (stratified by corpus and label) |

SpamAssassin parsed 6,046 messages, while its readme says 6,047. The difference is one message that the
maintainers removed from `spam_2` in March 2005 because it was mislabelled.
