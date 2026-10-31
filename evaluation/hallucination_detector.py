"""幻觉检测的规则层（L1）。

NLI 校验（L2）在 Judge Agent 里做，LLM 自评（L3）需要额外一轮模型调用，
这里只做确定性的规则检查，不依赖外部服务：
- 绝对化疗效表述
- 大段没有不确定性措辞的结论
- 带百分比但附近没有引用的数字
"""

from __future__ import annotations

import re
from typing import List, Tuple


ABSOLUTE_PATTERNS = [
    r"(必然|肯定|绝对|一定|毫无疑问)[^。]{0,20}(治愈|根治|完全消除|有效)",
    r"100%\s*(有效|治愈|安全)",
    r"(所有|全部)患者[^。]{0,10}(都|均)[^。]{0,10}(有效|缓解|改善)",
    r"零风险|完全没有副作用",
]

HEDGE_WORDS = ("可能", "或许", "提示", "建议", "考虑", "倾向于", "现有证据", "尚需")


class HallucinationDetector:
    def rule_check(self, text: str) -> Tuple[float, List[str]]:
        """返回 (风险分 0~1, 问题列表)。"""
        issues: List[str] = []
        risk = 0.0

        for pattern in ABSOLUTE_PATTERNS:
            hit = re.findall(pattern, text)
            if hit:
                issues.append(f"出现绝对化表述，临床报告应避免: {hit[0]}")
                risk += 0.25

        hedge_count = sum(text.count(word) for word in HEDGE_WORDS)
        if len(text) > 200 and hedge_count == 0:
            issues.append("结论性段落缺少不确定性措辞（如“提示”“建议”）")
            risk += 0.15

        # 百分比数字附近 100 字内没有 PMID 引用
        for num in re.findall(r"\d+(?:\.\d+)?\s*%", text):
            idx = text.find(num)
            window = text[max(0, idx - 100): idx + 100]
            if "PMID" not in window:
                issues.append(f"数值 {num} 附近未找到引用标注")
                risk += 0.1

        return min(risk, 1.0), issues
