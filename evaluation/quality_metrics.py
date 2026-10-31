"""检索质量指标。

不是 RAGAS 的完整实现，而是几个不依赖模型的关键词重合度代理指标，
用于快速判断“报告内容有没有证据支撑”。中文按字符 bigram 切分，英文按空格切分。
"""

from __future__ import annotations

import re
from typing import List, Set


def _tokens(text: str) -> Set[str]:
    text = text.lower()
    tokens: Set[str] = set()
    # 英文单词和数字
    tokens.update(re.findall(r"[a-z0-9]{2,}", text))
    # 中文片段做 bigram
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            tokens.add(seg)
        else:
            tokens.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


def faithfulness(answer: str, contexts: List[str]) -> float:
    """报告用词有多少出现在证据里。"""
    if not contexts:
        return 0.0
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    evidence_tokens: Set[str] = set()
    for ctx in contexts:
        evidence_tokens.update(_tokens(ctx))
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def context_precision(answer: str, contexts: List[str]) -> float:
    """检索到的证据里有多少与报告真正相关（>3 个词重合算相关）。"""
    if not contexts:
        return 0.0
    answer_tokens = _tokens(answer)
    useful = 0
    for ctx in contexts:
        if len(answer_tokens & _tokens(ctx)) >= 3:
            useful += 1
    return useful / len(contexts)


def answer_relevance(question: str, answer: str) -> float:
    q, a = _tokens(question), _tokens(answer)
    if not q or not a:
        return 0.0
    return len(q & a) / len(q | a)


def calculate_all(question: str, answer: str, contexts: List[str]) -> dict:
    return {
        "faithfulness": round(faithfulness(answer, contexts), 3),
        "context_precision": round(context_precision(answer, contexts), 3),
        "answer_relevance": round(answer_relevance(question, answer), 3),
    }
