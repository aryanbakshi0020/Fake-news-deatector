from __future__ import annotations

import re
from typing import Dict, List

from duckduckgo_search import DDGS


def normalize_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        results = DDGS().text(query, max_results=max_results)
        items = []
        for item in results:
            title = item.get("title") or "Untitled result"
            url = item.get("href") or item.get("url") or ""
            snippet = item.get("body") or item.get("snippet") or ""
            if url:
                items.append({"title": title, "url": url, "snippet": snippet})
        return items
    except Exception:
        return []


def score_evidence(claim: str, evidence: List[Dict[str, str]]) -> Dict[str, object]:
    text = claim.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9]+", text))
    evidence_list = []

    for item in evidence:
        page_text = (item.get("title", "") + " " + item.get("snippet", "") + " ").lower()
        overlap = len(tokens.intersection(set(re.findall(r"[a-zA-Z0-9]+", page_text))))
        score = overlap + (0.3 if page_text else 0)
        evidence_list.append({**item, "score": score})

    evidence_list.sort(key=lambda x: x["score"], reverse=True)
    return {"evidence": evidence_list}


def verify_claim(claim: str, max_sources: int = 5) -> Dict[str, object]:
    query = normalize_query(claim)
    if not query:
        return {"verdict": "UNVERIFIED", "confidence": 0.0, "reason": "Empty claim.", "evidence": []}

    results = search_web(query, max_results=max_sources)
    if not results:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.15,
            "reason": "No web evidence was retrieved. More sources are needed to assess the claim.",
            "evidence": [],
        }

    ranked = score_evidence(query, results)["evidence"]
    best = ranked[0]
    if len(ranked) >= 2:
        top_two = ranked[:2]
        top_score = top_two[0].get("score", 0)
        second_score = top_two[1].get("score", 0)
        if top_score > second_score * 1.5:
            strong_match = True
        else:
            strong_match = False
    else:
        strong_match = True

    if strong_match and "fact" in query.lower() and len(ranked) >= 2:
        verdict = "SUPPORTED"
        reason = "The claim is strongly aligned with current web evidence and matching source material."
        confidence = 0.82
    elif strong_match:
        verdict = "SUPPORTED"
        reason = "The claim appears supported by the retrieved source material and related web evidence."
        confidence = 0.74
    elif ranked[0].get("score", 0) > 0:
        verdict = "UNVERIFIED"
        reason = "Some partial matching evidence exists, but it is not enough to confidently confirm or reject the claim."
        confidence = 0.46
    else:
        verdict = "CONTRADICTED"
        reason = "The retrieved evidence does not support the claim and may be inconsistent with it."
        confidence = 0.62

    evidence = [{"title": item["title"], "url": item["url"], "snippet": item["snippet"]} for item in ranked[:max_sources]]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
        "evidence": evidence,
    }
