"""
Generic AI Text Classification Template
========================================

A reusable, dataset-agnostic template for classifying a column of free text
(survey answers, social media posts, support tickets, public-consultation
feedback, etc.) using an LLM, with built-in cost estimation and an accuracy
verification workflow.

Methodology and structure follow the approach taught by Jeremy Merrill
(data/AI reporter, The Washington Post) in the Lede Program (2026 cohort).

HOW TO REUSE THIS FILE FOR A NEW PROJECT
-----------------------------------------
1. Edit the CONFIG block below: your CSV path, text column name, and categories.
2. Rewrite SYSTEM_PROMPT with instructions + examples for your own task.
3. Create a `.env` file (add it to `.gitignore`!) with your API key(s), e.g.:
       OPENROUTER_API_KEY=sk-or-...
4. Run this script section by section (or paste into a Jupyter notebook).

WORKFLOW
--------
1. Load data, take a random sample.
2. Estimate cost of running the model on the sample and on the full dataset.
3. Classify the sample.
4. Export the sample for hand-coding (manual ground-truth labeling).
5. Re-import the hand-coded sample and measure accuracy vs. a baseline,
   plus a confusion matrix and (optionally) precision/recall for one category.
6. Iterate on the prompt until accuracy is good enough for your use case.
7. Classify the full dataset.
"""

import os
from enum import Enum

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
from dotenv import load_dotenv

tqdm.pandas()
load_dotenv()


# ---------------------------------------------------------------------------
# CONFIG — edit this block for your own project
# ---------------------------------------------------------------------------

CSV_PATH = "your_data.csv"          # path to your input data
TEXT_COLUMN = "text"                # name of the column containing the text to classify
ID_COLUMN = None                    # optional: a unique-id column to set as index, or None
SAMPLE_SIZE_FOR_HANDCODING = 50     # size of the random sample used to check accuracy
RANDOM_STATE = 613                  # fixed seed so the sample is reproducible

USE_OPENROUTER = True               # True = route all models through OpenRouter
                                     # False = call each provider's SDK directly


class Category(str, Enum):
    """Replace with your own categories."""
    CATEGORY_A = "category_a"
    CATEGORY_B = "category_b"
    OTHER = "other"


class ClassificationResult(BaseModel):
    classification: Category


SYSTEM_PROMPT_TEMPLATE = """You are classifying short texts into one of these categories: {categories}.

Choose exactly one label. Return only the label.

Examples
--------
"<example text 1>" = {example_label_1}
"<example text 2>" = {example_label_2}
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
    categories=", ".join(f'"{c.value}"' for c in Category),
    example_label_1=Category.CATEGORY_A.value,
    example_label_2=Category.OTHER.value,
)

PROMPT_BASE = """Now classify the following text:
"{text}"
"""


# ---------------------------------------------------------------------------
# API CLIENT SETUP
# ---------------------------------------------------------------------------

from anthropic import Anthropic

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))  # used for token counting

if USE_OPENROUTER:
    from openrouter import OpenRouter
    openrouter_client = OpenRouter(api_key=os.environ.get("OPENROUTER_API_KEY"))
else:
    from openai import OpenAI
    from mistralai.client import Mistral
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

# Map your friendly model name -> provider-specific model id (edit as needed)
OPENROUTER_MODEL_NAMES = {
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5.4": "openai/gpt-5.4",
    "mistral-large-latest": "mistralai/mistral-large-2512",
    "claude-4.5-haiku": "anthropic/claude-4.5-haiku",
    "claude-4.6-sonnet": "anthropic/claude-4.6-sonnet",
}

# Published per-input-token prices (USD) — update these before estimating costs
INPUT_TOKEN_COSTS = {
    "gpt-5-mini": 0.40 / 1_000_000,
    "gpt-5.4": 2.5 / 1_000_000,
    "mistral-large-latest": 0.5 / 1_000_000,
    "claude-4.5-haiku": 1 / 1_000_000,
    "claude-4.6-sonnet": 3 / 1_000_000,
}

MODEL_TO_USE = "gpt-5-mini"  # pick one of the keys above


# ---------------------------------------------------------------------------
# COST ESTIMATION
# ---------------------------------------------------------------------------

def count_tokens(model: str, text: str) -> int:
    if "claude" in model and os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic_client.messages.count_tokens(
            model=model, messages=[{"content": text, "role": "user"}]
        )
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4o")  # most modern models share this tokenizer
    return len(encoding.encode(text))


def estimate_cost(model: str, token_count: int) -> float:
    return token_count * INPUT_TOKEN_COSTS[model]


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------

def openrouter_classify(prompt_including_text: str) -> str:
    response = openrouter_client.chat.send(
        model=OPENROUTER_MODEL_NAMES[MODEL_TO_USE],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_including_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": ClassificationResult.model_json_schema(),
            },
        },
    )
    return ClassificationResult.model_validate_json(
        response.choices[0].message.content
    ).classification.value


def classify(prompt_including_text: str) -> str:
    """Provider-agnostic wrapper. Extend with direct OpenAI/Anthropic/Mistral
    calls here if USE_OPENROUTER is False."""
    if USE_OPENROUTER:
        return openrouter_classify(prompt_including_text)
    raise NotImplementedError(
        "Add a direct-provider classify function for MODEL_TO_USE, or set USE_OPENROUTER=True."
    )


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
    df.dropna(subset=[TEXT_COLUMN], inplace=True)
    if ID_COLUMN:
        df.set_index(ID_COLUMN, inplace=True)
    return df


def build_prompts(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: PROMPT_BASE.format(text=row[TEXT_COLUMN]), axis="columns")


def run_cost_estimate(prompts: pd.Series, label: str) -> None:
    joined = "SAMPLE RESPONSE SAMPLE RESPONSE".join(prompts)
    tokens = count_tokens(MODEL_TO_USE, joined)
    print(f"{label} would cost: ${estimate_cost(MODEL_TO_USE, tokens):.6f}")


def classify_sample(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.sample(n=SAMPLE_SIZE_FOR_HANDCODING, random_state=RANDOM_STATE).copy()
    prompts = build_prompts(sample)
    run_cost_estimate(prompts, "Sample")
    sample["ai_guess"] = prompts.progress_apply(classify)
    return sample


def export_for_handcoding(sample: pd.DataFrame, path: str = "handcoded.csv") -> None:
    sample["groundtruth"] = ""
    sample.to_csv(path)
    print(f"Fill in the 'groundtruth' column in {path}, then re-import it to measure accuracy.")


def evaluate_accuracy(handcoded_path: str = "handcoded.csv"):
    """Requires scikit-learn. Compares hand-coded ground truth to ai_guess."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        ConfusionMatrixDisplay,
    )

    handcoded = pd.read_csv(handcoded_path)
    assert "groundtruth" in handcoded.columns, "handcoded.csv needs a 'groundtruth' column"
    assert not handcoded["groundtruth"].isna().any(), "fill in every row of 'groundtruth' first"

    accuracy = accuracy_score(handcoded["groundtruth"], handcoded["ai_guess"])
    most_common = handcoded["groundtruth"].mode().iloc[0]
    baseline = accuracy_score(handcoded["groundtruth"], [most_common] * len(handcoded))

    print(f"Accuracy: {accuracy:.1%}")
    print(f"Baseline (always guessing '{most_common}'): {baseline:.1%}")

    ConfusionMatrixDisplay.from_predictions(
        handcoded["groundtruth"], handcoded["ai_guess"], xticks_rotation="vertical"
    )

    # Optional: precision/recall for one category of interest
    # category_of_interest = Category.CATEGORY_A.value
    # precision = precision_score(handcoded["groundtruth"] == category_of_interest,
    #                              handcoded["ai_guess"] == category_of_interest)
    # recall = recall_score(handcoded["groundtruth"] == category_of_interest,
    #                        handcoded["ai_guess"] == category_of_interest)
    # print(f"Precision: {precision:.1%}  |  Recall: {recall:.1%}")

    return accuracy, baseline


def classify_full_dataset(df: pd.DataFrame, output_path: str = "classified_output.csv") -> pd.DataFrame:
    prompts = build_prompts(df)
    run_cost_estimate(prompts, "Full dataset")
    df["ai_guess"] = prompts.progress_apply(classify)
    df.to_csv(output_path)
    return df


if __name__ == "__main__":
    data = load_data()
    sample_df = classify_sample(data)
    export_for_handcoding(sample_df)
    # After manually filling in handcoded.csv:
    # evaluate_accuracy("handcoded.csv")
    # Once accuracy is good enough:
    # classify_full_dataset(data)
