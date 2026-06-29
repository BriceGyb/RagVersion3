"""
Evaluation RAGAS — reproduction article RAGIntel (CTI-ATTACK, 229 requetes)

Compromis appliques (voir DECISIONS_COMPROMIS.txt) :
  - LLMs : GPT-4o et GPT-3.5 uniquement (BART et Flan-T5 exclus)
  - Metriques : context_recall, context_precision, answer_relevancy, noise_sensitivity
  - context_recall/precision calcules 1x globalement (independant du LLM)
  - Checkpoint toutes les 20 lignes pour reprendre en cas de crash
"""

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import os
import ast
import pandas as pd
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import (
    LLMContextRecall, ContextPrecision,
    AnswerRelevancy, NoiseSensitivity
)

INPUT_FILE          = "RAG-evaluation-results.xlsx"
CHECKPOINT_FILE     = "ragas_checkpoint.csv"
CHECKPOINT_CTX_FILE = "ragas_checkpoint_context.csv"
OUTPUT_FILE         = "RAG-evaluation-results-final.xlsx"
SUMMARY_FILE        = "ragas_scores_by_llm.xlsx"
BATCH_SIZE          = 20
LLMS_TO_EVAL        = ["openai-gpt-4o"]

print("=" * 60)
print("Evaluation RAGAS — CTI-ATTACK")
print("=" * 60)

# --- Chargement des donnees ---
print("\nChargement des donnees...")
df = pd.read_excel(INPUT_FILE)
df["retrieved_contexts"] = df["retrieved_contexts"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)
df["response"] = df["response"].fillna("").astype(str)
df = df.reset_index(drop=True)
print(f"  {len(df)} lignes totales")

# Filtrer : GPT-4o et GPT-3.5 uniquement
df_eval = df[df["llm_name"].isin(LLMS_TO_EVAL)].copy().reset_index(drop=True)
print(f"  {len(df_eval)} lignes retenues ({', '.join(LLMS_TO_EVAL)})")

# Dataset de base pour context metrics (1 seul LLM suffit car context est identique)
df_context = df[df["llm_name"] == "openai-gpt-4o"].copy().reset_index(drop=True)
print(f"  {len(df_context)} requetes pour context_recall/precision (calcul global)")

# --- Configuration LLM / Embeddings ---
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
evaluator_llm = llm_factory("gpt-4o-mini", client=openai_client)
evaluator_emb = LangchainEmbeddingsWrapper(
    LCOpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])
)
run_config = RunConfig(max_workers=32, timeout=120, max_retries=3)

generator_metrics = [
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
    NoiseSensitivity(llm=evaluator_llm),
]
context_metrics = [
    LLMContextRecall(llm=evaluator_llm),
    ContextPrecision(llm=evaluator_llm),
]
NOISE_COL        = "noise_sensitivity(mode=relevant)"
all_metric_names = ["context_recall", "context_precision", "answer_relevancy", NOISE_COL]

# ================================================================
# ETAPE 1 : Context recall + precision (calcul global, 1x)
# ================================================================
print("\n--- Etape 1/2 : context_recall + context_precision (global) ---")

if os.path.exists(CHECKPOINT_CTX_FILE):
    ctx_df = pd.read_csv(CHECKPOINT_CTX_FILE)
    ctx_done = set(ctx_df["original_index"].tolist())
    print(f"  Checkpoint contexte trouve : {len(ctx_done)}/{len(df_context)} deja evalues")
else:
    ctx_df = pd.DataFrame()
    ctx_done = set()

ctx_remaining = [i for i in range(len(df_context)) if i not in ctx_done]
print(f"  {len(ctx_remaining)} requetes restantes")

for batch_start in range(0, len(ctx_remaining), BATCH_SIZE):
    batch_indices = ctx_remaining[batch_start:batch_start + BATCH_SIZE]
    batch = df_context.iloc[batch_indices]
    pct = (len(ctx_done) + batch_start) / len(df_context) * 100
    print(f"  [{pct:.0f}%] Context batch {batch_start//BATCH_SIZE + 1}...")

    data = [
        {
            "user_input": str(r["user_input"]),
            "reference": str(r["reference"]) if pd.notna(r["reference"]) else "",
            "retrieved_contexts": r["retrieved_contexts"],
            "response": str(r["response"]),
        }
        for _, r in batch.iterrows()
    ]
    try:
        results = evaluate(EvaluationDataset.from_list(data), metrics=context_metrics, run_config=run_config)
        res_df = results.to_pandas()
        res_df["original_index"] = batch_indices
        ctx_df = pd.concat([ctx_df, res_df[["original_index", "context_recall", "context_precision"]]], ignore_index=True)
        ctx_df.to_csv(CHECKPOINT_CTX_FILE, index=False)
    except Exception as e:
        print(f"  Erreur batch contexte: {e}")
        empty = pd.DataFrame({"original_index": batch_indices, "context_recall": float("nan"), "context_precision": float("nan")})
        ctx_df = pd.concat([ctx_df, empty], ignore_index=True)
        ctx_df.to_csv(CHECKPOINT_CTX_FILE, index=False)

ctx_scores = pd.read_csv(CHECKPOINT_CTX_FILE)
avg_ctx_recall = ctx_scores["context_recall"].mean()
avg_ctx_precision = ctx_scores["context_precision"].mean()
print(f"\n  context_recall    (global) = {avg_ctx_recall:.4f}")
print(f"  context_precision (global) = {avg_ctx_precision:.4f}")

# ================================================================
# ETAPE 2 : answer_relevancy + noise_sensitivity (par LLM)
# ================================================================
print("\n--- Etape 2/2 : answer_relevancy + noise_sensitivity (par LLM) ---")

if os.path.exists(CHECKPOINT_FILE):
    checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
    done_indices = set(checkpoint_df["original_index"].tolist())
    print(f"  Checkpoint trouve : {len(done_indices)}/{len(df_eval)} deja evalues")
else:
    checkpoint_df = pd.DataFrame()
    done_indices = set()

remaining = [i for i in range(len(df_eval)) if i not in done_indices]
print(f"  {len(remaining)} echantillons restants")

for batch_start in range(0, len(remaining), BATCH_SIZE):
    batch_indices = remaining[batch_start:batch_start + BATCH_SIZE]
    batch = df_eval.iloc[batch_indices]
    pct = (len(done_indices) + batch_start) / len(df_eval) * 100
    print(f"  [{pct:.0f}%] Generator batch {batch_start//BATCH_SIZE + 1} ({batch['llm_name'].iloc[0]})...")

    data = [
        {
            "user_input": str(r["user_input"]),
            "reference": str(r["reference"]) if pd.notna(r["reference"]) else "",
            "retrieved_contexts": r["retrieved_contexts"],
            "response": str(r["response"]),
        }
        for _, r in batch.iterrows()
    ]
    try:
        results = evaluate(EvaluationDataset.from_list(data), metrics=generator_metrics, run_config=run_config)
        res_df = results.to_pandas()
        res_df["original_index"] = batch_indices
        res_df["llm_name"] = batch["llm_name"].values
        checkpoint_df = pd.concat([checkpoint_df, res_df[["original_index", "llm_name", "answer_relevancy", NOISE_COL]]], ignore_index=True)
        checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)
    except Exception as e:
        print(f"  Erreur batch: {e}")
        empty = pd.DataFrame({
            "original_index": batch_indices,
            "llm_name": batch["llm_name"].values,
            "answer_relevancy": float("nan"),
            NOISE_COL: float("nan"),
        })
        checkpoint_df = pd.concat([checkpoint_df, empty], ignore_index=True)
        checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)

# ================================================================
# ASSEMBLAGE FINAL
# ================================================================
print("\n--- Assemblage des resultats finaux ---")

gen_scores = pd.read_csv(CHECKPOINT_FILE)
final_df = df_eval.copy()
final_df["context_recall"] = avg_ctx_recall
final_df["context_precision"] = avg_ctx_precision
final_df["answer_relevancy"] = float("nan")
final_df[NOISE_COL] = float("nan")

for _, row in gen_scores.iterrows():
    idx = int(row["original_index"])
    final_df.at[idx, "answer_relevancy"] = row["answer_relevancy"]
    final_df.at[idx, NOISE_COL] = row[NOISE_COL]

final_df.to_excel(OUTPUT_FILE, index=False)
print(f"Resultats sauves dans : {OUTPUT_FILE}")

print("\n=== SCORES MOYENS PAR LLM ===")
summary = final_df.groupby("llm_name")[all_metric_names].mean().round(4)
print(summary.to_string())
summary.to_excel(SUMMARY_FILE)
print(f"Resume sauves dans : {SUMMARY_FILE}")

print("\n=== SCORES GLOBAUX RETRIEVER ===")
print(f"  context_recall    = {avg_ctx_recall:.4f}")
print(f"  context_precision = {avg_ctx_precision:.4f}")
