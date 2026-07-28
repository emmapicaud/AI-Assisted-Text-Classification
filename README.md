# AI-Assisted Text Classification 

This repository classifies free-text submissions to the EU public consultation on the **European Union Deforestation Regulation (EUDR)** into three categories, using a Large Language Model (LLM) with structured (schema-validated) outputs:

- `include_leather` — the comment supports **keeping** leather within the scope of the EUDR
- `exclude_leather` — the comment supports **excluding** leather from the scope of the EUDR
- `other_topic` — the comment does not address the leather provision

The classification prompt explicitly instructs the model to recognize the word "leather" across more than 20 EU languages (French, German, Italian, Portuguese, Polish, Greek, etc.), since consultation responses were submitted in many national languages.

## Methodology and credit

The code architecture, prompting approach, and the workflow for evaluating classification quality (cost estimation → sampling → hand-coding → accuracy/precision/recall → confusion matrix → iterative prompt refinement) follow the methodology taught by **Jeremy Merrill**, data/AI reporter at the *Washington Post*, as part of the Lede Program (2026 cohort). The notebook this repository is based on is adapted from his teaching template for AI-assisted classification of large text corpora.

The dataset and classification prompt (leather / EUDR) are specific to this project; the underlying pipeline is generic and reusable (see [Reuse](#reuse-as-a-template) below).

## How it works

1. **Data loading** — feedback submissions are loaded from a scrapped CSV export (originally sourced from a public Google Sheet of EUDR consultation responses). This step is described in full detail, with reusable code, in the companion repository:
   👉 **[scrape-the-EU-consultations-Have-You-Say-portal](../Scrape-the-EU-consultations-Have-You-Say-portal)** .
2. **Structured output schema** — a `pydantic` model (`FeedbackOptions` / `FeedbackCategory` enum) constrains the model's answer to exactly one of the three valid labels, requested via a JSON schema in the API call.
3. **System prompt** — a single, fixed system prompt defines the task, the three categories, the multilingual "leather" rule, and a handful of worked examples in different languages.
4. **Multi-provider API access** — classification calls are routed through [OpenRouter](https://openrouter.ai/), which can reach OpenAI (GPT-5 family), Anthropic (Claude), and Mistral models through a single interface. Direct provider clients (`openai`, `anthropic`, `mistralai`) are also supported if you prefer not to use OpenRouter.
5. **Cost estimation** — before running the model on the full dataset, the notebook estimates token counts (via `tiktoken` or the Anthropic token counter) and multiplies by each model's published per-token price, both for a small sample and for the full corpus.
6. **Classification** — a `classify()` wrapper dispatches each row's text to the selected model and returns the validated category.
7. **Accuracy verification** — a random sample is manually labeled (ground truth) and compared against the model's guesses. This step is described in full detail, with reusable code, in the companion repository:
   👉 **[classification-accuracy-verification](../classification-accuracy-verification)** (see that repo's README for accuracy scores, baseline comparison, confusion matrices, precision and recall).
8. **Full-dataset run and visualization** — once the prompt is validated on the sample, the model is run on the entire dataset, and results are visualized (e.g., a stacked bar chart of category share over time).

## Tech stack

- Python 3, `pandas`, `pydantic`, `tqdm`
- `openrouter` (or `openai` / `anthropic` / `mistralai` directly)
- `tiktoken` (token counting for cost estimation)
- `python-dotenv` (API key management via a local `.env` file, excluded from version control via `.gitignore`)
- `scikit-learn` (used in the verification repo for accuracy metrics)

## Sample results

Below is an illustrative excerpt of the kind of output the pipeline produces (comment text truncated; actual submissions are longer and in multiple languages):

| Comment (excerpt) | Language | AI classification |
|---|---|---|
| "We support excluding leather from the EUDR." | EN | `exclude_leather` |
| "Leather must be included because cattle production drives deforestation." | EN | `include_leather` |
| "Le cuir est une source majeure de déforestation trop souvent ignorée..." | FR | `include_leather` |
| "Apoio a proposta da Comissão Europeia de excluir o couro do escopo da EUDR..." | PT | `exclude_leather` |
| "We support the preservation of biodiversity in the EUDR." | EN | `other_topic` |

The full pipeline outputs one row per submission with the original text plus an `ai_guess` column, exported as `EUDR_feedback_ai_classified.csv`, and an aggregated monthly breakdown of category share.

## Setup

1. Clone this repository.
2. Create a `.env` file (not committed — add it to `.gitignore`) containing your API key(s), e.g.:
   ```
   OPENROUTER_API_KEY_LEDE=sk-or-...
   ```
3. Install dependencies:
   ```bash
   pip install pandas pydantic tqdm python-dotenv tiktoken openrouter anthropic scikit-learn
   ```
4. Point `data_full_with_feedbacks` at your own CSV of feedback text.
5. Run the notebook / script cell by cell: sample → cost estimate → classify sample → hand-code → measure accuracy → adjust prompt → classify full dataset.

## Reuse as a template

The pipeline is designed to be **generic**: swap out the category `Enum`, the system prompt, and the input CSV column names, and it can classify any batch of short texts (survey responses, social media posts, support tickets, etc.). A stripped-down, dataset-agnostic version of the code is provided in [`generic_classifier_template.py`](./generic_classifier_template.py) for reuse in new projects.

## License

MIT 
