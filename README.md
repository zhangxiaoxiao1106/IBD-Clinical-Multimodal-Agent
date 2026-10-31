# IBD 临床智能体系统

面向炎症性肠病（IBD）抗 TNF-α 治疗预后评估的多 Agent 流程演示。
用状态机驱动 5 个角色协同：分诊 → 文献检索 → 规则风险分析 → 结构化报告 → 质检。

## 流程

```
患者输入
  → Director 分诊（simple / moderate / complex）
  → Retrieval（PubMed 检索 + 可选本地向量库，RRF 融合）
  → Analysis（临床指标规则分层）
  → Report（六段式报告；无 LLM 时用模板生成）
  → Judge（事实一致性 / 引用准确性 / 完整性 / 连贯性）
```

分诊规则：

| 等级 | 条件 | 路径 |
|------|------|------|
| simple | 只有临床表格，无问卷文本 | Analysis → Report |
| moderate | 表格+文本，或无影像的融合输入 | Retrieval → Analysis → Report |
| complex | 含内镜/病理影像 | Retrieval → Analysis → Report → Judge |

## 目录结构

```
├── config/            # YAML 配置与加载器（settings.py）
├── core/              # 状态机、工具注册中心、Agent 基类、记忆、日志
├── models/schemas.py  # Pydantic 模型
├── tools/             # pubmed / vector / llm / nli 四个工具
├── rag/               # 查询式生成（正则 NER）、证据溯源
├── evaluation/        # 幻觉规则检查、检索质量指标
├── agents/            # director / retrieval / analysis / report / judge
├── scripts/run_demo.py
└── tests/
```

## 运行

```bash
pip install -r requirements.txt   # 最小依赖只有 pydantic、pyyaml、httpx
python scripts/run_demo.py
```

结果写入 `outputs/demo_results.json`。

## 各能力的真实状态（请注意，不是论文里的完整系统）

- **PubMed 检索**：真实调用 E-utilities，需要外网；网络不可用时自动降级为无证据。
- **向量检索**：需要 faiss、sentence-transformers 和预先构建好的 `data/faiss_index.bin`；
  缺任何一项都返回空并在证据链 `notes` 里说明，不影响主流程。
- **LLM 报告**：对接 OpenAI 兼容接口（默认 `http://localhost:8000/v1`）。
  服务不可用时报告由模板生成，`generated_by` 字段会标明是 `template` 还是 `llm`。
- **NLI 校验**：需要 torch/transformers（roberta-large-mnli），缺失时质检的事实一致性维度给中性分并标注"未校验"。
- **重排**：RRF 之后按证据等级做启发式加权，**没有** cross-encoder。
- **查询扩展**：正则词表 + 同义词，**没有** NER 模型或 HyDE。
- **风险分层**：基于 CDAI/Mayo/钙卫蛋白/既往生物制剂失败史的确定性规则，不涉及训练模型。

## 测试

```bash
pytest tests/
```

## 许可证

MIT
