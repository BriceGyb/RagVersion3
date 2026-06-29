"""
Agent MCP — Traitement des 229 requetes CTI-ATTACK
Utilise GPT-4o via OpenAI function calling (equivalent MCP tool use).
Meme modele que le RAG pour une comparaison equitable.
Resultats sauvegardes dans mcp_agent_results.xlsx avec checkpoint.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import ast
import time
import pandas as pd
from openai import OpenAI
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mcp_mitre_server import (
    decompose_query, extract_keywords, identify_context_type,
    search_techniques, get_technique, get_techniques_by_tactic,
    get_subtechniques, get_group, get_software, get_mitigations,
    cross_reference, filter_by_platform, rank_candidates,
    validate_technique, format_final_answer,
)

INPUT_FILE       = "RAG-evaluation-results.xlsx"
OUTPUT_FILE      = "mcp_agent_results.xlsx"
CHECKPOINT_FILE  = "mcp_checkpoint.csv"
MODEL            = "gpt-4o-mini"

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "decompose_query",
            "description": "Decompose une requete CTI complexe en sous-questions ciblees. A appeler en premier.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_keywords",
            "description": "Extrait les termes techniques cles (comportements, outils, plateformes) d'une requete CTI.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identify_context_type",
            "description": "Identifie la tactique MITRE probable et le type de contexte CTI.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_techniques",
            "description": "Recherche les techniques MITRE ATT&CK les plus pertinentes pour une requete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technique",
            "description": "Retourne le detail complet d'une technique MITRE par son ID (ex: T1059, T1110).",
            "parameters": {"type": "object", "properties": {"technique_id": {"type": "string"}}, "required": ["technique_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_techniques_by_tactic",
            "description": "Retourne les techniques d'une tactique MITRE (ex: initial-access, execution, persistence, credential-access).",
            "parameters": {
                "type": "object",
                "properties": {
                    "tactic_name": {"type": "string"},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["tactic_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subtechniques",
            "description": "Retourne les sous-techniques d'une technique parent (ex: T1059 -> T1059.001, T1059.003...).",
            "parameters": {"type": "object", "properties": {"technique_id": {"type": "string"}}, "required": ["technique_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_group",
            "description": "Retourne les informations sur un groupe de menace MITRE (APT28, Lazarus, etc.).",
            "parameters": {"type": "object", "properties": {"group_name": {"type": "string"}}, "required": ["group_name"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_software",
            "description": "Retourne les informations sur un logiciel malveillant ou outil MITRE (Mimikatz, Cobalt Strike...).",
            "parameters": {"type": "object", "properties": {"software_name": {"type": "string"}}, "required": ["software_name"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mitigations",
            "description": "Retourne les mitigations recommandees pour une technique MITRE.",
            "parameters": {"type": "object", "properties": {"technique_id": {"type": "string"}}, "required": ["technique_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_reference",
            "description": "Analyse les relations entre plusieurs techniques (tactiques communes, chaine d'attaque).",
            "parameters": {
                "type": "object",
                "properties": {"technique_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["technique_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_platform",
            "description": "Filtre une liste de techniques par plateforme cible (Windows, Linux, macOS, Cloud...).",
            "parameters": {
                "type": "object",
                "properties": {
                    "technique_ids": {"type": "array", "items": {"type": "string"}},
                    "platform": {"type": "string"},
                },
                "required": ["technique_ids", "platform"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_candidates",
            "description": "Classe une liste de techniques candidates par pertinence par rapport a la requete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {"type": "array", "items": {"type": "object"}},
                    "query": {"type": "string"},
                },
                "required": ["candidates", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_technique",
            "description": "Valide si une technique MITRE correspond au comportement decrit. Retourne un score de confiance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "technique_id": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["technique_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_final_answer",
            "description": "Formate la reponse finale avec les IDs MITRE et le raisonnement. TOUJOURS appeler en dernier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "technique_ids": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string"},
                },
                "required": ["technique_ids", "reasoning"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "decompose_query": decompose_query,
    "extract_keywords": extract_keywords,
    "identify_context_type": identify_context_type,
    "search_techniques": search_techniques,
    "get_technique": get_technique,
    "get_techniques_by_tactic": get_techniques_by_tactic,
    "get_subtechniques": get_subtechniques,
    "get_group": get_group,
    "get_software": get_software,
    "get_mitigations": get_mitigations,
    "cross_reference": cross_reference,
    "filter_by_platform": filter_by_platform,
    "rank_candidates": rank_candidates,
    "validate_technique": validate_technique,
    "format_final_answer": format_final_answer,
}

SYSTEM_PROMPT = """You are a cybersecurity expert specialized in MITRE ATT&CK framework.
Your task is to analyze attack descriptions and identify the corresponding MITRE ATT&CK technique IDs.

Use the available tools systematically:
1. Call decompose_query and extract_keywords first to understand the query
2. Call search_techniques and/or get_techniques_by_tactic to find candidates
3. Call get_technique to get details on promising candidates
4. Call validate_technique and rank_candidates to confirm your choices
5. Always call format_final_answer last

The final line of your response must contain ONLY the main technique IDs (no sub-techniques),
separated by commas. Example: T1110, T1078, T1059"""


def call_tool(name: str, inputs: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Outil inconnu: {name}"})
    try:
        result = func(**inputs)
        return json.dumps(result, ensure_ascii=False)[:4000]
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_agent(client: OpenAI, query: str, max_iterations: int = 10) -> tuple:
    """
    Execute l'agent sur une requete.
    Retourne (reponse_finale, contextes_utilises, nb_tool_calls).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    contexts_used = []
    tool_call_count = 0

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if response.choices[0].finish_reason == "stop":
            return msg.content or "", contexts_used, tool_call_count

        if response.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_count += 1
                try:
                    inputs = json.loads(tc.function.arguments)
                except Exception:
                    inputs = {}

                result_str = call_tool(tc.function.name, inputs)

                if tc.function.name in ("search_techniques", "get_technique", "get_techniques_by_tactic"):
                    contexts_used.append(result_str[:500])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
        else:
            break

    return msg.content or "", contexts_used, tool_call_count


def main():
    print("=" * 60)
    print(f"Agent MCP MITRE ATT&CK — CTI-ATTACK (229 requetes)")
    print(f"Modele : {MODEL}")
    print("=" * 60)

    rag_df = pd.read_excel(INPUT_FILE)
    rag_df = rag_df[rag_df["llm_name"] == "openai-gpt-4o"].copy().reset_index(drop=True)
    rag_df["retrieved_contexts"] = rag_df["retrieved_contexts"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    print(f"  {len(rag_df)} requetes chargees")

    if Path(CHECKPOINT_FILE).exists():
        chk = pd.read_csv(CHECKPOINT_FILE)
        done = set(chk["index"].tolist())
        print(f"  Checkpoint: {len(done)}/{len(rag_df)} deja traites")
    else:
        chk = pd.DataFrame()
        done = set()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    remaining = [i for i in range(len(rag_df)) if i not in done]
    print(f"  {len(remaining)} requetes restantes\n")

    for i in remaining:
        row = rag_df.iloc[i]
        query = str(row["user_input"])
        reference = str(row["reference"]) if pd.notna(row["reference"]) else ""

        pct = (len(done) + (remaining.index(i))) / len(rag_df) * 100
        print(f"  [{pct:.0f}%] Requete {i+1}/{len(rag_df)}...")

        try:
            response, contexts, n_tools = run_agent(client, query)
            status = "ok"
        except Exception as e:
            print(f"    Erreur: {e}")
            response, contexts, n_tools = "", [], 0
            status = f"error: {str(e)[:100]}"

        record = pd.DataFrame([{
            "index": i,
            "user_input": query,
            "reference": reference,
            "retrieved_contexts": str(contexts),
            "response": response,
            "llm_name": f"mcp-{MODEL}",
            "tool_calls": n_tools,
            "status": status,
        }])
        chk = pd.concat([chk, record], ignore_index=True)
        chk.to_csv(CHECKPOINT_FILE, index=False)
        time.sleep(0.3)

    chk.to_excel(OUTPUT_FILE, index=False)
    print(f"\nResultats sauves dans : {OUTPUT_FILE}")
    print(f"Reponses valides : {(chk['response'].str.strip() != '').sum()}/{len(chk)}")
    print(f"Moyenne appels outils/requete : {chk['tool_calls'].mean():.1f}")


if __name__ == "__main__":
    main()
