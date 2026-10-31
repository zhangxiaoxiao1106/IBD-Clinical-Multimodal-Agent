"""PubMed E-utilities 检索：esearch 拿 PMID，efetch 拉 XML 解析。

只依赖 httpx。带简单的本地缓存和请求间隔控制（NCBI 建议 ≤3 次/秒）。
"""

from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from core.tool_registry import BaseTool, ToolSchema
from tools.base import PubMedSearchInput, PubMedSearchOutput

try:
    import httpx
except ImportError:  # 环境没装 httpx 时整个工具降级
    httpx = None  # type: ignore


LEVEL_RCT = "I级：随机对照试验 (RCT)"
LEVEL_COHORT = "II级：队列研究"
LEVEL_CASE_CONTROL = "III级：病例对照研究"
LEVEL_CASE_REPORT = "IV级：病例报告 / 专家意见"


class PubMedTool(BaseTool):
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    MIN_INTERVAL = 0.34

    def __init__(self) -> None:
        super().__init__(ToolSchema(
            name="pubmed_search",
            description="检索 PubMed，返回带证据等级标注的文献列表",
            input_schema=PubMedSearchInput,
            output_schema=PubMedSearchOutput,
            timeout_seconds=30.0,
            max_retries=1,
            fallback_value={"articles": [], "returned": 0, "total_found": 0,
                            "query_used": "", "search_time_ms": 0.0},
        ))
        self._client = httpx.AsyncClient(timeout=20.0) if httpx is not None else None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_request = 0.0

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError("httpx 未安装，PubMed 工具不可用")

        params = PubMedSearchInput(**inputs)
        cache_key = f"{params.query}|{params.max_results}|{params.date_range}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        elapsed = asyncio.get_event_loop().time() - self._last_request
        if elapsed < self.MIN_INTERVAL:
            await asyncio.sleep(self.MIN_INTERVAL - elapsed)

        start = time.time()
        search_params: Dict[str, Any] = {
            "db": "pubmed",
            "term": params.query,
            "retmax": params.max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if params.date_range and ":" in params.date_range:
            min_year, max_year = params.date_range.split(":", 1)
            search_params["mindate"] = min_year
            search_params["maxdate"] = max_year
            search_params["datetype"] = "pdat"

        search_resp = await self._client.get(f"{self.BASE_URL}/esearch.fcgi", params=search_params)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        result_block = search_data.get("esearchresult", {})
        id_list = result_block.get("idlist", [])
        total = int(result_block.get("count", 0))

        articles: List[Dict[str, Any]] = []
        if id_list:
            fetch_resp = await self._client.get(
                f"{self.BASE_URL}/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"},
            )
            fetch_resp.raise_for_status()
            articles = self._parse_xml(fetch_resp.text)
            for art in articles:
                art["evidence_level"] = self._classify(art)

        self._last_request = asyncio.get_event_loop().time()
        result = {
            "total_found": total,
            "returned": len(articles),
            "articles": articles,
            "query_used": params.query,
            "search_time_ms": (time.time() - start) * 1000,
        }
        self._cache[cache_key] = result
        return result

    def _parse_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        articles: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return articles

        for node in root.findall(".//PubmedArticle"):
            def _text(path: str) -> str:
                el = node.find(path)
                return el.text.strip() if el is not None and el.text else ""

            abstract_parts = [
                el.text.strip()
                for el in node.findall(".//Abstract/AbstractText")
                if el.text
            ]
            year_el = node.find(".//PubDate/Year") or node.find(".//PubDate/MedlineDate")
            year = None
            if year_el is not None and year_el.text:
                digits = "".join(ch for ch in year_el.text[:4] if ch.isdigit())
                year = int(digits) if digits else None

            articles.append({
                "pmid": _text(".//PMID"),
                "title": _text(".//ArticleTitle"),
                "abstract": " ".join(abstract_parts),
                "journal": _text(".//Journal/Title"),
                "year": year,
                "authors": [el.text for el in node.findall(".//Author/LastName") if el.text],
                "mesh_terms": [el.text for el in node.findall(".//MeshHeading/DescriptorName") if el.text],
                "publication_types": [
                    el.text for el in node.findall(".//PublicationType") if el.text
                ],
            })
        return articles

    @staticmethod
    def _classify(article: Dict[str, Any]) -> str:
        """按 PublicationType / MeSH 粗分证据等级，不准的一律按 IV 级处理。"""
        ptypes = " ".join(article.get("publication_types", [])).lower()
        mesh = " ".join(article.get("mesh_terms", [])).lower()

        if "randomized controlled trial" in ptypes or "controlled clinical trial" in ptypes:
            return LEVEL_RCT
        if "cohort studies" in mesh or "cohort" in ptypes or "prospective" in mesh:
            return LEVEL_COHORT
        if "case-control studies" in mesh or "retrospective" in mesh:
            return LEVEL_CASE_CONTROL
        return LEVEL_CASE_REPORT
