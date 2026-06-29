"""
Serveur MCP — MITRE ATT&CK
Expose 15 outils organises en 3 couches :
  - Couche 1 : Comprehension de la requete (decompose, extract, identify)
  - Couche 2 : Recuperation MITRE (search, get_technique, tactic, group, software...)
  - Couche 3 : Raffinement & Agregation (cross_ref, filter, rank, validate, format)
"""

import json
import re
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

import fastmcp

MITRE_JSON = Path("enterprise-attack.json")

# ─── Chargement MITRE ───────────────────────────────────────────────────────

def _load_mitre():
    with open(MITRE_JSON, encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects", [])

    techniques, tactics, groups, software, campaigns, mitigations, relationships = {}, {}, {}, {}, {}, {}, []

    for obj in objects:
        t = obj.get("type", "")
        if t == "attack-pattern":
            ext = obj.get("external_references", [])
            tid = next((r["external_id"] for r in ext if r.get("source_name") == "mitre-attack"), None)
            if tid:
                techniques[tid] = obj
        elif t == "x-mitre-tactic":
            short = obj.get("x_mitre_shortname", "")
            tactics[short] = obj
        elif t == "intrusion-set":
            groups[obj["name"].lower()] = obj
            for alias in obj.get("aliases", []):
                groups[alias.lower()] = obj
        elif t in ("tool", "malware"):
            software[obj["name"].lower()] = obj
            for alias in obj.get("x_mitre_aliases", []):
                software[alias.lower()] = obj
        elif t == "campaign":
            campaigns[obj["name"].lower()] = obj
        elif t == "course-of-action":
            ext = obj.get("external_references", [])
            mid = next((r["external_id"] for r in ext if r.get("source_name") == "mitre-attack"), None)
            if mid:
                mitigations[mid] = obj
        elif t == "relationship":
            relationships.append(obj)

    return techniques, tactics, groups, software, campaigns, mitigations, relationships

print("Chargement MITRE ATT&CK...")
TECHNIQUES, TACTICS, GROUPS, SOFTWARE, CAMPAIGNS, MITIGATIONS, RELATIONSHIPS = _load_mitre()
print(f"  {len(TECHNIQUES)} techniques, {len(TACTICS)} tactiques, {len(GROUPS)} groupes, {len(SOFTWARE)} logiciels")

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _search_text(query: str, obj: dict, top_k: int = 5) -> float:
    q = query.lower()
    name = obj.get("name", "").lower()
    desc = obj.get("description", "").lower()
    score = 0.0
    for word in q.split():
        if word in name:
            score += 2.0
        if word in desc:
            score += 1.0
    return score

# ─── Serveur MCP ─────────────────────────────────────────────────────────────

mcp = fastmcp.FastMCP("MITRE ATT&CK MCP Server")

# ══════════════════════════════════════════════════════════════
# COUCHE 1 — Comprehension de la requete
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def decompose_query(query: str) -> dict:
    """
    Decompose une requete CTI complexe en sous-questions ciblees.
    Retourne une liste de sous-questions et le type d'analyse recommande.
    """
    keywords = extract_keywords(query)
    context_type = identify_context_type(query)

    sub_questions = []
    if keywords.get("behaviors"):
        sub_questions.append(f"Quelles techniques MITRE correspondent aux comportements : {', '.join(keywords['behaviors'][:3])} ?")
    if keywords.get("tools"):
        sub_questions.append(f"Quels logiciels ou outils MITRE sont lies a : {', '.join(keywords['tools'][:3])} ?")
    if keywords.get("platforms"):
        sub_questions.append(f"Quelles techniques ciblent la plateforme : {', '.join(keywords['platforms'][:2])} ?")
    if context_type.get("tactic"):
        sub_questions.append(f"Quelles techniques appartiennent a la tactique : {context_type['tactic']} ?")

    if not sub_questions:
        sub_questions.append(f"Rechercher des techniques MITRE liees a : {query[:100]}")

    return {
        "original_query": query,
        "sub_questions": sub_questions,
        "recommended_tools": ["search_techniques", "get_techniques_by_tactic", "get_software"],
        "analysis_type": context_type.get("type", "technique_identification"),
    }


@mcp.tool()
def extract_keywords(query: str) -> dict:
    """
    Extrait les termes techniques cles d'une requete CTI :
    comportements, outils, plateformes, indicateurs.
    """
    q = query.lower()

    behavior_patterns = [
        "login", "password", "credential", "brute force", "phishing", "injection",
        "execution", "persistence", "privilege", "lateral", "exfiltration", "command",
        "control", "discovery", "collection", "ransomware", "encrypt", "backdoor",
        "dropper", "loader", "shellcode", "exploit", "vulnerability", "scan", "recon",
        "spearphishing", "attachment", "macro", "powershell", "script", "registry",
        "process", "memory", "dump", "token", "bypass", "evasion", "obfuscat",
        "network", "traffic", "dns", "http", "smb", "rdp", "ssh", "ftp",
    ]
    platform_patterns = ["windows", "linux", "macos", "cloud", "azure", "aws", "android", "ios"]
    tool_patterns = ["mimikatz", "cobalt strike", "metasploit", "nmap", "wireshark",
                     "psexec", "bloodhound", "empire", "beacon", "meterpreter"]

    behaviors = [p for p in behavior_patterns if p in q]
    platforms = [p for p in platform_patterns if p in q]
    tools = [p for p in tool_patterns if p in q]

    words = re.findall(r'\b[A-Z][A-Za-z0-9]{2,}\b', query)
    technique_ids = re.findall(r'\bT\d{4}(?:\.\d{3})?\b', query)

    return {
        "behaviors": behaviors[:8],
        "platforms": platforms,
        "tools": tools,
        "named_entities": words[:10],
        "technique_ids": technique_ids,
    }


@mcp.tool()
def identify_context_type(query: str) -> dict:
    """
    Identifie le type de contexte CTI de la requete :
    tactique probable, type d'analyse, niveau de specificite.
    """
    q = query.lower()

    tactic_keywords = {
        "initial-access": ["phishing", "exploit", "supply chain", "valid account", "external", "spearphish"],
        "execution": ["execute", "run", "powershell", "script", "macro", "command"],
        "persistence": ["persist", "startup", "registry", "scheduled task", "service", "backdoor"],
        "privilege-escalation": ["privilege", "escalat", "admin", "root", "uac", "bypass"],
        "defense-evasion": ["evad", "obfuscat", "bypass", "disable", "log", "clear", "hide"],
        "credential-access": ["credential", "password", "hash", "token", "kerberos", "login", "brute"],
        "discovery": ["scan", "enum", "discover", "recon", "list", "network", "host"],
        "lateral-movement": ["lateral", "spread", "move", "rdp", "smb", "psexec", "pass the"],
        "collection": ["collect", "captur", "screen", "keylog", "clipboard", "file"],
        "command-and-control": ["c2", "c&c", "command", "control", "beacon", "callback", "dns tunnel"],
        "exfiltration": ["exfil", "steal", "upload", "transfer", "leak"],
        "impact": ["ransom", "encrypt", "destroy", "wipe", "disrupt", "dos"],
    }

    scores = {tactic: sum(1 for kw in kws if kw in q) for tactic, kws in tactic_keywords.items()}
    best_tactic = max(scores, key=scores.get) if max(scores.values()) > 0 else None

    return {
        "tactic": best_tactic,
        "tactic_score": scores.get(best_tactic, 0),
        "type": "technique_identification",
        "specificity": "high" if len(q.split()) > 20 else "medium" if len(q.split()) > 10 else "low",
    }


# ══════════════════════════════════════════════════════════════
# COUCHE 2 — Recuperation MITRE
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def search_techniques(query: str, top_k: int = 5) -> list:
    """
    Recherche les techniques MITRE ATT&CK les plus pertinentes pour une requete CTI.
    Retourne les top_k techniques avec ID, nom, tactiques et description courte.
    """
    scored = []
    for tid, obj in TECHNIQUES.items():
        score = _search_text(query, obj)
        if score > 0:
            scored.append((score, tid, obj))

    scored.sort(reverse=True, key=lambda x: x[0])
    results = []
    for score, tid, obj in scored[:top_k]:
        tactics = [p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])]
        desc = obj.get("description", "")[:300]
        results.append({
            "id": tid,
            "name": obj.get("name", ""),
            "tactics": tactics,
            "description": desc,
            "relevance_score": round(score, 2),
        })
    return results


@mcp.tool()
def get_technique(technique_id: str) -> dict:
    """
    Retourne le detail complet d'une technique MITRE par son ID (ex: T1059, T1059.001).
    """
    obj = TECHNIQUES.get(technique_id.upper())
    if not obj:
        return {"error": f"Technique {technique_id} non trouvee"}

    tactics = [p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])]
    platforms = obj.get("x_mitre_platforms", [])
    detection = obj.get("x_mitre_detection", "")

    return {
        "id": technique_id.upper(),
        "name": obj.get("name", ""),
        "tactics": tactics,
        "platforms": platforms,
        "description": obj.get("description", ""),
        "detection": detection[:500] if detection else "",
        "is_subtechnique": "." in technique_id,
    }


@mcp.tool()
def get_techniques_by_tactic(tactic_name: str, top_k: int = 10) -> list:
    """
    Retourne toutes les techniques associees a une tactique MITRE.
    Ex: tactic_name = 'initial-access', 'execution', 'persistence', 'lateral-movement'
    """
    tactic_name = tactic_name.lower().replace(" ", "-")
    results = []
    for tid, obj in TECHNIQUES.items():
        phases = [p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])]
        if tactic_name in phases:
            results.append({
                "id": tid,
                "name": obj.get("name", ""),
                "description": obj.get("description", "")[:200],
            })
    return results[:top_k]


@mcp.tool()
def get_subtechniques(technique_id: str) -> list:
    """
    Retourne toutes les sous-techniques d'une technique parent (ex: T1059 -> T1059.001, T1059.002...).
    """
    parent = technique_id.upper().split(".")[0]
    results = []
    for tid, obj in TECHNIQUES.items():
        if tid.startswith(parent + "."):
            results.append({
                "id": tid,
                "name": obj.get("name", ""),
                "description": obj.get("description", "")[:200],
            })
    return results


@mcp.tool()
def get_group(group_name: str) -> dict:
    """
    Retourne les informations sur un groupe de menace MITRE (APT28, Lazarus, etc.).
    """
    obj = GROUPS.get(group_name.lower())
    if not obj:
        close = [(k, _sim(group_name, k)) for k in GROUPS]
        close.sort(key=lambda x: -x[1])
        if close and close[0][1] > 0.6:
            obj = GROUPS[close[0][0]]
        else:
            return {"error": f"Groupe '{group_name}' non trouve", "suggestions": [k for k, _ in close[:3]]}

    ext = obj.get("external_references", [])
    gid = next((r["external_id"] for r in ext if r.get("source_name") == "mitre-attack"), "")
    return {
        "id": gid,
        "name": obj.get("name", ""),
        "aliases": obj.get("aliases", []),
        "description": obj.get("description", "")[:500],
    }


@mcp.tool()
def get_software(software_name: str) -> dict:
    """
    Retourne les informations sur un logiciel malveillant ou outil (Mimikatz, Cobalt Strike, etc.).
    """
    obj = SOFTWARE.get(software_name.lower())
    if not obj:
        close = [(k, _sim(software_name, k)) for k in SOFTWARE]
        close.sort(key=lambda x: -x[1])
        if close and close[0][1] > 0.6:
            obj = SOFTWARE[close[0][0]]
        else:
            return {"error": f"Logiciel '{software_name}' non trouve", "suggestions": [k for k, _ in close[:3]]}

    ext = obj.get("external_references", [])
    sid = next((r["external_id"] for r in ext if r.get("source_name") == "mitre-attack"), "")
    return {
        "id": sid,
        "name": obj.get("name", ""),
        "type": obj.get("type", ""),
        "aliases": obj.get("x_mitre_aliases", []),
        "description": obj.get("description", "")[:500],
        "platforms": obj.get("x_mitre_platforms", []),
    }


@mcp.tool()
def get_mitigations(technique_id: str) -> list:
    """
    Retourne les mitigations recommandees pour une technique MITRE donnee.
    """
    tid_upper = technique_id.upper()
    obj = TECHNIQUES.get(tid_upper)
    if not obj:
        return [{"error": f"Technique {technique_id} non trouvee"}]

    tech_ref = obj.get("id", "")
    related_mits = []
    for rel in RELATIONSHIPS:
        if rel.get("relationship_type") == "mitigates" and rel.get("target_ref") == tech_ref:
            src = rel.get("source_ref", "")
            for mid, mit in MITIGATIONS.items():
                if mit.get("id") == src:
                    related_mits.append({
                        "id": mid,
                        "name": mit.get("name", ""),
                        "description": mit.get("description", "")[:300],
                    })
    return related_mits if related_mits else [{"info": f"Aucune mitigation trouvee pour {technique_id}"}]


# ══════════════════════════════════════════════════════════════
# COUCHE 3 — Raffinement & Agregation
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def cross_reference(technique_ids: list) -> dict:
    """
    Analyse les relations entre plusieurs techniques MITRE :
    tactiques communes, plateformes communes, co-occurrence.
    """
    details = []
    all_tactics, all_platforms = [], []

    for tid in technique_ids:
        obj = TECHNIQUES.get(tid.upper())
        if obj:
            tactics = [p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])]
            platforms = obj.get("x_mitre_platforms", [])
            all_tactics.extend(tactics)
            all_platforms.extend(platforms)
            details.append({"id": tid, "name": obj.get("name", ""), "tactics": tactics})

    from collections import Counter
    tactic_counts = Counter(all_tactics)
    platform_counts = Counter(all_platforms)

    return {
        "techniques_analyzed": details,
        "common_tactics": [t for t, c in tactic_counts.most_common(3)],
        "common_platforms": [p for p, c in platform_counts.most_common(3)],
        "attack_chain_possible": len(set(all_tactics)) > 1,
    }


@mcp.tool()
def filter_by_platform(technique_ids: list, platform: str) -> list:
    """
    Filtre une liste de techniques par plateforme cible.
    Plateformes : Windows, Linux, macOS, Cloud, Azure AD, Office 365, Android, iOS
    """
    platform_lower = platform.lower()
    results = []
    for tid in technique_ids:
        obj = TECHNIQUES.get(tid.upper())
        if obj:
            platforms = [p.lower() for p in obj.get("x_mitre_platforms", [])]
            if any(platform_lower in p for p in platforms):
                results.append({
                    "id": tid,
                    "name": obj.get("name", ""),
                    "platforms": obj.get("x_mitre_platforms", []),
                })
    return results


@mcp.tool()
def rank_candidates(candidates: list, query: str) -> list:
    """
    Classe une liste de techniques candidates par pertinence par rapport a la requete originale.
    candidates : liste de dicts avec 'id' et optionnellement 'name'
    """
    scored = []
    for item in candidates:
        tid = item.get("id", "") if isinstance(item, dict) else item
        obj = TECHNIQUES.get(tid.upper())
        if obj:
            score = _search_text(query, obj)
            scored.append({
                "id": tid.upper(),
                "name": obj.get("name", ""),
                "relevance_score": round(score, 2),
                "tactics": [p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])],
            })
    scored.sort(key=lambda x: -x["relevance_score"])
    return scored


@mcp.tool()
def validate_technique(technique_id: str, query: str) -> dict:
    """
    Valide si une technique MITRE correspond au comportement decrit dans la requete.
    Retourne un score de confiance et une justification.
    """
    obj = TECHNIQUES.get(technique_id.upper())
    if not obj:
        return {"valid": False, "confidence": 0.0, "reason": f"Technique {technique_id} non trouvee"}

    score = _search_text(query, obj)
    desc = obj.get("description", "").lower()
    q_words = set(query.lower().split())
    desc_words = set(desc.split())
    overlap = q_words & desc_words
    confidence = min(1.0, score / 10.0)

    return {
        "technique_id": technique_id.upper(),
        "technique_name": obj.get("name", ""),
        "valid": confidence > 0.1,
        "confidence": round(confidence, 3),
        "matching_keywords": list(overlap)[:10],
        "reason": f"Score de correspondance : {score:.1f}, mots en commun : {len(overlap)}",
    }


@mcp.tool()
def format_final_answer(technique_ids: list, reasoning: str) -> dict:
    """
    Formate la reponse finale avec les IDs MITRE principaux (sans sous-techniques)
    et le raisonnement de l'agent.
    Format de sortie conforme au dataset CTI-ATTACK.
    """
    main_ids = []
    details = []
    seen = set()

    for tid in technique_ids:
        tid_clean = tid.upper().strip()
        main_id = tid_clean.split(".")[0]
        if main_id not in seen:
            seen.add(main_id)
            main_ids.append(main_id)
            obj = TECHNIQUES.get(tid_clean) or TECHNIQUES.get(main_id)
            if obj:
                details.append({
                    "id": main_id,
                    "name": obj.get("name", ""),
                    "tactics": [p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])],
                })

    final_line = ", ".join(main_ids)
    return {
        "reasoning": reasoning,
        "technique_details": details,
        "final_answer": final_line,
        "technique_count": len(main_ids),
    }


# ─── Point d'entree ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
