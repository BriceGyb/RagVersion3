"""
Evaluation RAGAS sur les resultats de l'agent MCP.
Memes metriques que pour le RAG : context_recall, context_precision,
answer_relevancy, noise_sensitivity.
Permet la comparaison directe RAG vs MCP.
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

INPUT_FILE          = "mcp_agent_results.xlsx"
CHECKPOINT_FILE     = "ragas_mcp_checkpoint.csv"
CHECKPOINT_CTX_FILE = "ragas_mcp_checkpoint_context.csv"
OUTPUT_FILE         = "mcp_evaluation_results.xlsx"
SUMMARY_FILE        = "mcp_ragas_scores.xlsx"
BATCH_SIZE          = 20
NOISE_COL           = "noise_sensitivity(mode=relevant)"

print("=" * 60)
print("Evaluation RAGAS — Agent MCP CTI-ATTACK")
print("=" * 60)

df = pd.read_excel(INPUT_FILE)
df["retrieved_contexts"] = df["retrieved_contexts"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)
df["response"] = df["response"].fillna("").astype(str)
df = df.reset_index(drop=True)
print(f"  {len(df)} resultats MCP charges")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
evaluator_llm = llm_factory("gpt-4o-mini", client=openai_client)
evaluator_emb = LangchainEmbeddingsWrapper(
    LCOpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])
)
run_config = RunConfig(max_workers=32, timeout=120, max_retries=3)

context_metrics  = [LLMContextRecall(llm=evaluator_llm), ContextPrecision(llm=evaluator_llm)]
generator_metrics = [AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb), NoiseSensitivity(llm=evaluator_llm)]

# ── Etape 1 : context_recall + context_precision ──
print("\n--- Etape 1/2 : context_recall + context_precision ---")

if os.path.exists(CHECKPOINT_CTX_FILE):
    ctx_df = pd.read_csv(CHECKPOINT_CTX_FILE)
    ctx_done = set(ctx_df["original_index"].tolist())
else:
    ctx_df = pd.DataFrame()
    ctx_done = set()

ctx_remaining = [i for i in range(len(df)) if i not in ctx_done]
print(f"  {len(ctx_remaining)} requetes restantes")

for batch_start in range(0, len(ctx_remaining), BATCH_SIZE):
    batch_indices = ctx_remaining[batch_start:batch_start + BATCH_SIZE]
    batch = df.iloc[batch_indices]
    pct = (len(ctx_done) + batch_start) / len(df) * 100
    print(f"  [{pct:.0f}%] Context batch {batch_start//BATCH_SIZE + 1}...")

    data = [{"user_input": str(r["user_input"]), "reference": str(r["reference"]) if pd.notna(r["reference"]) else "", "retrieved_contexts": r["retrieved_contexts"], "response": str(r["response"])} for _, r in batch.iterrows()]
    try:
        results = evaluate(EvaluationDataset.from_list(data), metrics=context_metrics, run_config=run_config)
        res_df = results.to_pandas()
        res_df["original_index"] = batch_indices
        ctx_df = pd.concat([ctx_df, res_df[["original_index", "context_recall", "context_precision"]]], ignore_index=True)
        ctx_df.to_csv(CHECKPOINT_CTX_FILE, index=False)
    except Exception as e:
        print(f"  Erreur: {e}")
        empty = pd.DataFrame({"original_index": batch_indices, "context_recall": float("nan"), "context_precision": float("nan")})
        ctx_df = pd.concat([ctx_df, empty], ignore_index=True)
        ctx_df.to_csv(CHECKPOINT_CTX_FILE, index=False)

ctx_scores = pd.read_csv(CHECKPOINT_CTX_FILE)
avg_ctx_recall    = ctx_scores["context_recall"].mean()
avg_ctx_precision = ctx_scores["context_precision"].mean()
print(f"\n  context_recall    = {avg_ctx_recall:.4f}")
print(f"  context_precision = {avg_ctx_precision:.4f}")

# ── Etape 2 : answer_relevancy + noise_sensitivity ──
print("\n--- Etape 2/2 : answer_relevancy + noise_sensitivity ---")

if os.path.exists(CHECKPOINT_FILE):
    chk = pd.read_csv(CHECKPOINT_FILE)
    done = set(chk["original_index"].tolist())
else:
    chk = pd.DataFrame()
    done = set()

remaining = [i for i in range(len(df)) if i not in done]
print(f"  {len(remaining)} echantillons restants")

for batch_start in range(0, len(remaining), BATCH_SIZE):
    batch_indices = remaining[batch_start:batch_start + BATCH_SIZE]
    batch = df.iloc[batch_indices]
    pct = (len(done) + batch_start) / len(df) * 100
    print(f"  [{pct:.0f}%] Generator batch {batch_start//BATCH_SIZE + 1}...")

    data = [{"user_input": str(r["user_input"]), "reference": str(r["reference"]) if pd.notna(r["reference"]) else "", "retrieved_contexts": r["retrieved_contexts"], "response": str(r["response"])} for _, r in batch.iterrows()]
    try:
        results = evaluate(EvaluationDataset.from_list(data), metrics=generator_metrics, run_config=run_config)
        res_df = results.to_pandas()
        res_df["original_index"] = batch_indices
        res_df["llm_name"] = batch["llm_name"].values
        chk = pd.concat([chk, res_df[["original_index", "llm_name", "answer_relevancy", NOISE_COL]]], ignore_index=True)
        chk.to_csv(CHECKPOINT_FILE, index=False)
    except Exception as e:
        print(f"  Erreur: {e}")
        empty = pd.DataFrame({"original_index": batch_indices, "llm_name": batch["llm_name"].values, "answer_relevancy": float("nan"), NOISE_COL: float("nan")})
        chk = pd.concat([chk, empty], ignore_index=True)
        chk.to_csv(CHECKPOINT_FILE, index=False)

# ── Assemblage final ──
print("\n--- Assemblage final ---")
gen_scores = pd.read_csv(CHECKPOINT_FILE)
final_df = df.copy()
final_df["context_recall"]    = avg_ctx_recall
final_df["context_precision"] = avg_ctx_precision
final_df["answer_relevancy"]  = float("nan")
final_df[NOISE_COL]           = float("nan")

for _, row in gen_scores.iterrows():
    idx = int(row["original_index"])
    final_df.at[idx, "answer_relevancy"] = row["answer_relevancy"]
    final_df.at[idx, NOISE_COL]          = row[NOISE_COL]

final_df.to_excel(OUTPUT_FILE, index=False)
print(f"Resultats sauves dans : {OUTPUT_FILE}")

all_metric_names = ["context_recall", "context_precision", "answer_relevancy", NOISE_COL]
print("\n=== SCORES MCP ===")
summary = final_df[all_metric_names].mean().round(4)
print(summary.to_string())
pd.DataFrame([summary], index=["claude-mcp"]).to_excel(SUMMARY_FILE)
print(f"\nResume sauves dans : {SUMMARY_FILE}")

# ── Comparaison RAG vs MCP ──
print("\n=== COMPARAISON RAG vs MCP ===")
rag_scores = pd.read_excel("ragas_scores_by_llm.xlsx", index_col=0)
print("\nRAG (GPT-4o):")
print(rag_scores.to_string())
print("\nMCP (Claude):")
print(summary.to_string())
