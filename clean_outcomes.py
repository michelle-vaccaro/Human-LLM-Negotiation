#!/usr/bin/env python3
"""Clean and enrich negotiation transcripts.
...
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# =============================================================================
# Constants
# =============================================================================

NAME_COLS = ["name_1", "name_2", "prompt_1", "prompt_2", "prompt_name_1", "prompt_name_2"]
SCORE_COLS = ["email_cleaned", "warmth_score", "dominance_score"]

HEDGE_WORDS = [
    "i think", "i believe", "i feel", "i guess", "i suppose", "i hope",
    "a little", "quite", "somewhat", "kind of", "sort of", "relatively",
    "fairly", "maybe", "perhaps", "possibly", "ideally", "hopefully",
    "might", "probably",
]
APOLOGIES = [
    "sorry", "i apologize", "we apologize", "my apologies", "forgive me",
    "excuse me", "pardon me", "i regret", "i didn't mean to",
]
SUBJUNCTIVES = [
    "i wish", "if only", "i would", "it is important that", "i suggest that",
    "i propose that", "i recommend that", "i hope that", "it would be better if",
    "if i were", "would you mind",
]
GRATITUDE = [
    "thank you", "thanks", "i'm thankful", "we're thankful", "i appreciate",
    "we appreciate", "i'm grateful", "we're grateful", "appreciated",
]
PLURAL_PRONOUNS = ["we", "our", "ours", "us", "ourselves", "let's"]

METRIC_DEFS = {
    "hedge": HEDGE_WORDS,
    "apology": APOLOGIES,
    "subjunctive": SUBJUNCTIVES,
    "gratitude": GRATITUDE,
    "fpp": PLURAL_PRONOUNS,
}

QUESTION_WORDS = {
    "what", "why", "when", "where", "who", "which", "whom", "whose", "how",
    "is", "am", "are", "do", "does", "did", "can", "could", "shall", "should",
    "will", "would", "have", "has", "had", "don't",
}

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =============================================================================
# Functions
# =============================================================================

def deserialize_conversation(column: pd.Series) -> pd.Series:
    return column.apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x))

def merge_scores(df: pd.DataFrame, score_path: Path, role1: str, role2: str) -> pd.DataFrame:
    df_scores = pd.read_csv(score_path)
    df_scores["email_cleaned"] = df_scores["email"].str.lower()
    df["name_1"] = df["name_1"].str.lower()
    df["name_2"] = df["name_2"].str.lower()
    role1_map = {"warmth_score": "role1_warmth_score", "dominance_score": "role1_dominance_score"}
    role2_map = {"warmth_score": "role2_warmth_score", "dominance_score": "role2_dominance_score"}
    out = pd.merge(df_scores[SCORE_COLS], df, left_on="email_cleaned", right_on="name_1", how="inner")
    out = out.rename(columns=role1_map)
    out = pd.merge(df_scores[SCORE_COLS], out, left_on="email_cleaned", right_on="name_2", how="left")
    out = out.rename(columns=role2_map)
    return out

def check_exit(conversation: Sequence[str], role: str) -> int | None:
    if(type(conversation) == float):
        return None
    if not conversation:
        return None
    for msg in conversation:
        if msg.startswith(f"{role}:") and "[NO DEAL]" in msg:
            return 1
    return 0

def mimicry_scores(messages: Sequence[str]) -> List[float]:
    if len(messages) <= 1:
        return [0.0] * len(messages)
    vec = TfidfVectorizer(min_df=1, max_features=500)
    emb = vec.fit_transform(messages).toarray()
    scores = [0.0]
    for prev, curr in zip(emb, emb[1:]):
        scores.append(float(cosine_similarity([curr], [prev])[0][0]))
    return scores

def role_based_mimicry(df: pd.DataFrame, role1: str, role2: str, column: str = "conversation") -> pd.DataFrame:
    role1_scores, role2_scores = [], []
    for convo in df[column]:
        if(type(convo) == float):
            return None
    
        roles, texts = [], []
        for utt in convo:
            if utt.startswith(role1):
                roles.append(role1)
                texts.append(utt[len(role1) + 1 :].lstrip(": "))
            elif utt.startswith(role2):
                roles.append(role2)
                texts.append(utt[len(role2) + 1 :].lstrip(": "))
        if len(texts) <= 1:
            role1_scores.append(np.nan)
            role2_scores.append(np.nan)
            continue
        mim = mimicry_scores(texts)
        prev_role = roles[0]
        r1_list, r2_list = [], []
        for this_role, this_score in zip(roles[1:], mim[1:]):
            if this_role == role1 and prev_role == role2:
                r1_list.append(this_score)
            elif this_role == role2 and prev_role == role1:
                r2_list.append(this_score)
            prev_role = this_role
        role1_scores.append(float(np.mean(r1_list)) if r1_list else np.nan)
        role2_scores.append(float(np.mean(r2_list)) if r2_list else np.nan)
    df["role1_mimicry"] = role1_scores
    df["role2_mimicry"] = role2_scores
    return df

def count_phrase_occurrences(text: str, phrases: Iterable[str]) -> int:
    text_low = text.lower()
    return sum(text_low.count(p.lower()) for p in phrases)

def is_question(sentence: str) -> bool:
    s = sentence.strip().lower()
    return s.endswith("?") or any(s.startswith(q) for q in QUESTION_WORDS)

def compute_conversation_metrics(df: pd.DataFrame, role1: str, role2: str, column: str = "conversation") -> pd.DataFrame:
    metrics = {f"role1_{m}": [] for m in METRIC_DEFS}
    metrics.update({f"role2_{m}": [] for m in METRIC_DEFS})
    metrics["role1_message_length"], metrics["role2_message_length"] = [], []
    metrics["role1_questions"], metrics["role2_questions"] = [], []

    for convo in df[column]:
        if(type(convo) == float):
            return None
        
        role_data = {
            role1: {"tokens": 0, "utters": 0, **{m: 0 for m in METRIC_DEFS}, "questions": 0},
            role2: {"tokens": 0, "utters": 0, **{m: 0 for m in METRIC_DEFS}, "questions": 0},
        }
        for utt in convo:
            if utt.startswith(role1):
                who, txt = role1, utt[len(role1) + 1 :].lstrip(": ")
            elif utt.startswith(role2):
                who, txt = role2, utt[len(role2) + 1 :].lstrip(": ")
            else:
                continue
            role_data[who]["utters"] += 1
            role_data[who]["tokens"] += len(txt.split())
            role_data[who]["questions"] += count_questions(txt)
            for m, phrases in METRIC_DEFS.items():
                role_data[who][m] += count_phrase_occurrences(txt, phrases)

        for role_label, role_key in ((role1, "role1"), (role2, "role2")):
            data = role_data[role_label]
            denom = max(data["utters"], 1)
            metrics[f"{role_key}_message_length"].append(data["tokens"] / denom)
            metrics[f"{role_key}_questions"].append(data["questions"] / denom)
            for m in METRIC_DEFS:
                metrics[f"{role_key}_{m}"].append(data[m] / denom)

    for k, v in metrics.items():
        df[k] = v
    return df

def count_questions(text: str) -> int:
    sentences = re.split(r"(?<=[.!?]) +", text)
    return sum(is_question(s) for s in sentences)

def sentiment_scores(df: pd.DataFrame, role1: str, role2: str) -> pd.DataFrame:
    r1_scores, r2_scores = [], []
    for convo in df["conversation"]:
        if(type(convo) == float):
            return None
        
        r1_list, r2_list = [], []
        for utt in convo:
            if utt.startswith(role1 + ":"):
                blob = TextBlob(utt[len(role1) + 1 :].lstrip(": "))
                r1_list.append(blob.sentiment.polarity)
            elif utt.startswith(role2 + ":"):
                blob = TextBlob(utt[len(role2) + 1 :].lstrip(": "))
                r2_list.append(blob.sentiment.polarity)
        r1_scores.append(float(np.mean(r1_list)) if r1_list else 0.0)
        r2_scores.append(float(np.mean(r2_list)) if r2_list else 0.0)
    df["role1_sentiment"], df["role2_sentiment"] = r1_scores, r2_scores
    return df

def reshape_long(df: pd.DataFrame, role1: str, role2: str) -> pd.DataFrame:
    rows_focal_role1, rows_focal_role2 = [], []
    for (_, row) in df.iterrows():
        r1, r2 = {}, {}
        for col, val in row.items():
            if col.startswith("role1_"):
                base = col[len("role1_") :]
                r1[f"focal_{base}"] = val
                r2[f"counterpart_{base}"] = val
            elif col.startswith("role2_"):
                base = col[len("role2_") :]
                r1[f"counterpart_{base}"] = val
                r2[f"focal_{base}"] = val
            else:
                if col == "name_1":
                    r1["focal_name"], r2["counterpart_name"] = val, val
                elif col == "name_2":
                    r1["counterpart_name"], r2["focal_name"] = val, val
                elif col == "prompt_1":
                    r1["focal_prompt"], r2["counterpart_prompt"] = val, val
                elif col == "prompt_2":
                    r1["counterpart_prompt"], r2["focal_prompt"] = val, val
                elif col == "prompt_name_1":
                    r1["focal_prompt_name"], r2["counterpart_prompt_name"] = val, val
                elif col == "prompt_name_2":
                    r1["counterpart_prompt_name"], r2["focal_prompt_name"] = val, val
                else:
                    r1[col] = val
                    r2[col] = val
        r1["role"], r2["role"] = role1, role2
        rows_focal_role1.append(r1)
        rows_focal_role2.append(r2)
    df_long = pd.DataFrame(rows_focal_role1 + rows_focal_role2)
    df_long["dyad"] = df_long.apply(
        lambda r: " & ".join(
            sorted([str(r.get("focal_name", "")), str(r.get("counterpart_name", ""))])
        ),
        axis=1,
    )
    return df_long


def run_pipeline(transcript_path: Path, score_path: Path, role1: str, role2: str, test: bool) -> None:
    logging.info("Loading transcripts %s", transcript_path)
    df = pd.read_csv(transcript_path)

    if test:
        logging.info("Running in test mode, limiting to 10 rows")
        df = df.head(10)

    logging.info("Merging rater scores %s", score_path)
    df = merge_scores(df, score_path, role1, role2)

    df["conversation"] = deserialize_conversation(df["conversation"])

    logging.info("Computing exit flags, mimicry, lexical metrics, sentiment")
    df["role1_exit"] = df["conversation"].apply(lambda x: check_exit(x, role1))
    df["role2_exit"] = df["conversation"].apply(lambda x: check_exit(x, role2))
    df = role_based_mimicry(df, role1, role2)
    df = compute_conversation_metrics(df, role1, role2)
    df = sentiment_scores(df, role1, role2)

    out_wide = transcript_path.with_name(transcript_path.stem + "_cleaned.csv.gz")
    logging.info("Saving wide data → %s", out_wide.name)
    df.to_csv(out_wide, index=False, compression="gzip")

    logging.info("Reshaping to focal/counterpart view → long format")
    df_long = reshape_long(df, role1, role2)
    out_long = transcript_path.with_name(transcript_path.stem + "_cleaned_long.csv.gz")
    df_long.to_csv(out_long, index=False, compression="gzip")
    logging.info("All done ✔︎")

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean negotiation transcripts")
    p.add_argument("--transcript", type=Path, required=True, help="CSV file with conversations")
    p.add_argument("--scores", type=Path, required=True, help="CSV with rater scores")
    p.add_argument("--role1", default="Buyer", help="Name of role 1 (default: Buyer)")
    p.add_argument("--role2", default="Seller", help="Name of role 2 (default: Seller)")
    return p.parse_args(argv)

if __name__ == "__main__":
    round = "1"
    exercise = "table"
    transcript = Path(f"outcomes/round{round}_{exercise}_outcomes.csv")
    scores = Path(f"prompts/round{round}_prompts_scored_4.5.csv")
    role1 = "Buyer"
    role2 = "Seller"
    test = False
    run_pipeline(transcript, scores, role1, role2, test=test)
    # args = parse_args()
    # run_pipeline(args.transcript, args.scores, args.role1, args.role2)