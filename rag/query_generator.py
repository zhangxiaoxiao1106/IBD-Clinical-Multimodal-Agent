"""基于规则的医学实体识别和 PubMed 查询式扩展。

没有用专门的 NER 模型，就是少量基因/药物/疾病的正则 + 同义词表，
覆盖 IBD 常见场景足够，扩展时直接补词表即可。
"""

from __future__ import annotations

import re
from typing import Dict, List


class MedicalQueryGenerator:
    GENE_PATTERNS = [
        r"\b(TNF|IL6|IL10|IL12B|IL23R|IFNG|NOD2|ATG16L1|IRGM|FCGR3A)\b",
    ]
    DRUG_PATTERNS = [
        r"\b(infliximab|adalimumab|vedolizumab|ustekinumab|golimumab|certolizumab)\b",
        r"(英夫利西单抗|阿达木单抗|维得利珠单抗|乌司奴单抗)",
        r"\b(azathioprine|methotrexate)\b",
    ]
    # 裸缩写 CD/UC 在 PubMed 里检索效果很差，不直接收，
    # 由调用方在拼接临床文本时补成全称
    DISEASE_PATTERNS = [
        r"(Crohn'?s? disease|Crohn disease|ulcerative colitis|inflammatory bowel disease|\bIBD\b)",
        r"(克罗恩病|溃疡性结肠炎|炎症性肠病)",
    ]

    SYNONYMS = {
        "infliximab": ["IFX", "Remicade"],
        "adalimumab": ["ADA", "Humira"],
        "vedolizumab": ["VDZ", "Entyvio"],
        "ustekinumab": ["UST", "Stelara"],
        "IBD": ["inflammatory bowel disease"],
    }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {"genes": [], "drugs": [], "diseases": []}
        for key, patterns in (
            ("genes", self.GENE_PATTERNS),
            ("drugs", self.DRUG_PATTERNS),
            ("diseases", self.DISEASE_PATTERNS),
        ):
            found: List[str] = []
            for pattern in patterns:
                found.extend(re.findall(pattern, text, flags=re.IGNORECASE))
            entities[key] = sorted(set(m.strip() for m in found))
        return entities

    def generate_queries(self, text: str, max_expansions: int = 2) -> List[str]:
        e = self.extract_entities(text)

        parts: List[str] = []
        if e["genes"]:
            parts.append("(" + " OR ".join(e["genes"][:3]) + ")")
        if e["drugs"]:
            parts.append("(" + " OR ".join(e["drugs"][:2]) + ")")
        if e["diseases"]:
            parts.append("(" + " OR ".join(e["diseases"][:2]) + ")")

        queries: List[str] = []
        if parts:
            queries.append(" AND ".join(parts))
        else:
            queries.append("inflammatory bowel disease anti-TNF treatment response biomarker")

        if e["genes"] or e["drugs"]:
            queries.append(queries[0] + " AND (response OR resistance OR prognosis)")

        if e["drugs"]:
            drug = e["drugs"][0].lower()
            syns = self.SYNONYMS.get(drug)
            if syns:
                queries.append(f"({' OR '.join([e['drugs'][0]] + syns)}) AND IBD")

        return queries[: max_expansions + 1]
