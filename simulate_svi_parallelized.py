import asyncio
import os, re
import pandas as pd
import numpy as np
from openai import AsyncClient   # pip install --upgrade openai>=1.14.0
from dotenv import load_dotenv
import openai

# ── Settings ───────────────────────────────────────────────────────────
MODEL          = "gpt-4o-mini"  # or "gpt-4o" for full model
TEMPERATURE    = 0.2
INPUT_FILE     = "outcomes/pilot_data_cleaned.csv"
CONSULANT_PATH     = "prompts/pilot_consultant_instructions.txt"
COO_PATH     = "prompts/pilot_coo_instructions.txt"
SURVEY_PATH    = "prompts/svi_survey.txt"
TEST_MODE      = False           # True → first 10 rows only
OUTPUT_FILE    = ("outcomes/pilot_data_cleaned_full_test.csv"
                  if TEST_MODE else
                  "outcomes/pilot_data_cleaned_full.csv")
CONCURRENCY    = 8               # tweak to match your rate-limit & cores
CHECKPOINT_EVERY = 500           # rows

# ── File loading ───────────────────────────────────────────────────────
def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

consultant_instr   = read_text(CONSULANT_PATH)
coo_instr   = read_text(COO_PATH)
survey_prompt = read_text(SURVEY_PATH)

df = pd.read_csv(OUTPUT_FILE if os.path.exists(OUTPUT_FILE) else INPUT_FILE)
if TEST_MODE:
    df = df.head(10)
if "simulated_svi_human" not in df.columns:
    df["simulated_svi_human"] = np.nan      # placeholder column

# ── Fast SVI-parser  (vectorised & pre-compiled) ───────────────────────
_strip_num_re = re.compile(r"^\d+\.\s*")
score_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Somewhat disagree": 3,
    "Neither agree nor disagree": 4,
    "Somewhat agree": 5,
    "Agree": 6,
    "Strongly agree": 7,
}

def parse_svi(raw: str):
    # discard anything before a colon, then split
    part = raw.split(":", 1)[-1] if ":" in raw else raw
    answers = [
        score_map.get(_strip_num_re.sub("", ln.strip()), np.nan)
        for ln in part.splitlines() if ln.strip()
    ]
    if len(answers) < 12:            # malformed – keep NaNs so np.nanmean works
        answers += [np.nan] * (12 - len(answers))

    # reverse-score Q3 & Q5  (0-index → 2,4)
    for idx in (2, 4):
        if not np.isnan(answers[idx]):
            answers[idx] = 8 - answers[idx]

    # sub-scales unused below; keep if you need them
    sv_instrumental = np.nanmean(answers[0:4])     # Q1-Q4
    sv_self         = np.nanmean(answers[4:8])     # Q5-Q8
    sv_process      = np.nanmean(answers[8:12])    # Q9-Q12
    sv_relationship = np.nanmean(answers[12:16])   # Q13-Q16
    sv_global       = np.nanmean(answers)

    return {
        "simulated_sv_instrumental":   sv_instrumental,
        "simulated_sv_self":           sv_self,
        "simulated_sv_process":        sv_process,
        "simulated_sv_relationship":   sv_relationship,
        "simulated_sv_global":         sv_global,
    }

# ── Async OpenAI calls ─────────────────────────────────────────────────
load_dotenv()
client = AsyncClient()   # respects OPENAI_API_KEY env-var

def build_messages(transcript, role):
    sys_prompt = (coo_instr if role == "COO" else consultant_instr)
    sys_full   = f"{sys_prompt}\n\nHere is the negotiation you just participated in:\n{transcript}"
    return [{"role": "system", "content": sys_full},
            {"role": "user",   "content": survey_prompt}]

async def fetch_svi(transcript, role):
    resp = await client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=build_messages(transcript, role),
    )
    return resp.choices[0].message.content

# ── Main async loop with batched checkpoints ───────────────────────────
async def run():
    sem   = asyncio.Semaphore(CONCURRENCY)
    tasks = []

    async def process(idx, trans, role):
        if pd.notna(df.at[idx, "simulated_svi_human"]):
            return          # already done from previous run

        async with sem:
            try:
                raw  = await fetch_svi(trans, role)
                scores = parse_svi(raw)

                for k, v in scores.items():
                    df.at[idx, k] = v          # prefix keeps them together
            except Exception as e:
                df.at[idx, "simulated_svi_human"] = f"ERROR: {e}"

    for idx, row in df.iterrows():
        tasks.append(process(idx, row.cleaned_messages, row.role))

    # gather in slices so we can checkpoint frequently without
    # waiting for *all* rows to finish first
    for i in range(0, len(tasks), CHECKPOINT_EVERY):
        await asyncio.gather(*tasks[i:i+CHECKPOINT_EVERY])
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✔ saved up to row {min(i+CHECKPOINT_EVERY, len(df))}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"🏁 all done → {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(run())