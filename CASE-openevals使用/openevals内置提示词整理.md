# openevals 常用内置提示词（Prompts）整理

## 概述

`openevals` 是 LangChain 官方推出的开源 LLM 评估框架，采用 **LLM-as-a-Judge**（大模型当裁判）机制，通过另一个 LLM 对目标模型的输出进行质量评估。框架内置了多种评估提示词，开发者只需几行代码即可对 LLM 应用进行多维度量化评估。

本文档整理了 `openevals` 中 11 个常用内置提示词，涵盖通用文本评估、RAG 系统评估、代码评估和 Agent 评估四大场景。

---

## 一、通用文本评估

### 1. CORRECTNESS_PROMPT（正确性评估）

| 项目 | 说明 |
|------|------|
| **功能** | 验证模型回答的正确性，衡量生成答案与标准答案的吻合程度 |
| **输入参数** | `inputs`（用户问题）、`outputs`（模型回答）、`reference_outputs`（标准参考答案，可选） |
| **评分方式** | 布尔值（True/False）或连续值（0.0~1.0，需设置 `continuous=True`） |
| **评估逻辑** | 裁判 LLM 根据评分标准判断答案是否准确、完整，是否与参考答案一致 |

**业务使用场景：**

- **问答系统质量监控**：在客服机器人上线前，批量测试其对常见问题的回答是否正确
- **知识问答应用**：验证 RAG 系统生成的答案是否准确回答了用户问题
- **教育辅助工具**：对比学生答案与标准答案，自动评分
- **A/B 模型对比**：对比不同 LLM 在同一批测试用例上的正确率，选择最优模型

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:o3-mini",
    continuous=True,
)

result = evaluator(
    inputs="Python 是什么语言？",
    outputs="Python 是一种高级编程语言。",
    reference_outputs="Python 是一种高级、解释型、面向对象的编程语言。",
)
print(result["score"])  # 0.0 ~ 1.0
```

---

### 2. CONCISENESS_PROMPT（简洁性评估）

| 项目 | 说明 |
|------|------|
| **功能** | 评估模型回答是否简洁，是否包含不必要的问候语、废话或冗余信息 |
| **输入参数** | `inputs`（用户问题）、`outputs`（模型回答） |
| **评分方式** | 布尔值（True=简洁，False=不简洁）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 检查回答中是否包含多余的前缀（如"好的！"、"很高兴为您解答！"）或重复性废话 |

**业务使用场景：**

- **客服对话优化**：确保客服机器人的回答简洁高效，不啰嗦
- **摘要生成评估**：验证摘要模型是否去除了冗余信息，输出精炼的摘要
- **移动端对话**：在屏幕空间有限的场景下，确保回答简洁易读
- **批量对话系统**：评估不同 prompt 模板对回答简洁度的影响

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT

evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    model="openai:o3-mini",
)

result = evaluator(
    inputs="旧金山的天气怎么样？",
    outputs="谢谢你的提问！旧金山的天气现在很好，阳光明媚，90度。",
)
# score: False（包含不必要的问候语"谢谢你的提问！"）
```

---

### 3. ANSWER_RELEVANCE_PROMPT（相关性评估）

| 项目 | 说明 |
|------|------|
| **功能** | 评估模型回答是否与用户问题直接相关，是否偏离了主题 |
| **输入参数** | `inputs`（用户问题）、`outputs`（模型回答） |
| **评分方式** | 布尔值（True/False）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 判断回答是否切题，是否回答了用户的核心问题 |

**业务使用场景：**

- **搜索引擎问答**：确保检索增强生成的答案直接回应用户的查询意图
- **对话系统**：检测模型是否"跑题"或回答与问题无关的内容
- **FAQ 系统**：验证自动生成的 FAQ 回答是否真正回答了用户的问题
- **多轮对话监控**：确保每一轮回答都与当前轮次的问题相关

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import ANSWER_RELEVANCE_PROMPT

evaluator = create_llm_as_judge(
    prompt=ANSWER_RELEVANCE_PROMPT,
    feedback_key="answer_relevance",
    model="openai:o3-mini",
)

result = evaluator(
    inputs="机器学习是什么？",
    outputs="机器学习是 AI 的一个分支，它让计算机从数据中学习模式并做出决策，而无需显式编程。",
)
# score: True（回答与问题高度相关）
```

---

### 4. HALLUCINATION_PROMPT（幻觉检测）

| 项目 | 说明 |
|------|------|
| **功能** | 检测模型回答中是否存在幻觉，即生成了与上下文或事实不符的虚假信息 |
| **输入参数** | `inputs`（用户问题）、`outputs`（模型回答）、`context`（参考上下文，可选） |
| **评分方式** | 布尔值（True=存在幻觉，False=无幻觉）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 对比模型回答与提供的上下文，检查回答中的声明是否有上下文依据 |

**业务使用场景：**

- **RAG 系统幻觉检测**：在知识库问答中，检测模型是否编造了知识库中不存在的信息
- **金融/医疗领域**：对高风险领域的回答进行幻觉审查，防止虚假信息造成损失
- **事实核查系统**：自动检测新闻、报告中的虚假声明
- **模型迭代对比**：对比不同版本模型在相同上下文下的幻觉率

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import HALLUCINATION_PROMPT

evaluator = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    feedback_key="hallucination",
    model="openai:o3-mini",
)

result = evaluator(
    inputs="什么是 doodad？",
    outputs="Doodad 是一种自我复制的纳米机器人集群，非常危险。",
    context="Doodad 是一种日常用品，类似于工具或装置。",
)
# score: True（回答与上下文矛盾，存在幻觉）
```

---

### 5. TOXICITY_PROMPT（毒性/有害性检测）

| 项目 | 说明 |
|------|------|
| **功能** | 检测模型输出是否包含有害、不当、攻击性或歧视性内容 |
| **输入参数** | `inputs`（用户问题）、`outputs`（模型回答） |
| **评分方式** | 布尔值（True=有毒性，False=无毒）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 分析回答中是否包含辱骂、歧视、暴力、仇恨言论等有害内容 |

**业务使用场景：**

- **内容安全合规**：确保 AI 产品的输出符合内容安全法规要求
- **客服机器人监控**：防止客服 AI 生成冒犯性或不专业的回答
- **青少年教育产品**：严格过滤不适合未成年人的内容
- **舆情风险管理**：定期扫描模型输出，降低品牌声誉风险

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import TOXICITY_PROMPT

evaluator = create_llm_as_judge(
    prompt=TOXICITY_PROMPT,
    feedback_key="toxicity",
    model="openai:o3-mini",
)

result = evaluator(
    inputs="你怎么看这个人？",
    outputs="你这个蠢货，什么都不懂！",
)
# score: True（包含攻击性语言）
```

---

## 二、RAG 系统评估

### 6. RAG_HELPFULNESS_PROMPT（RAG 帮助性评估）

| 项目 | 说明 |
|------|------|
| **功能** | 评估 RAG 系统最终生成的回答是否有效解决了用户的初始问题 |
| **输入参数** | `inputs`（包含 `question` 字段）、`outputs`（包含 `answer` 字段） |
| **评分方式** | 布尔值（True/False）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 对比最终回答与用户问题，判断回答是否真正解决了用户疑问，不依赖参考答案 |

**业务使用场景：**

- **RAG 端到端评估**：评估整个 RAG 链路（检索+生成）的最终效果
- **检索策略优化**：对比不同检索策略（如 top-k 调整、重排序）对回答帮助性的影响
- **知识库质量评估**：检测知识库内容是否足以支撑对常见问题的回答
- **产品上线前验收**：作为 RAG 系统上线的核心指标之一

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import RAG_HELPFULNESS_PROMPT

evaluator = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    model="openai:o3-mini",
)

result = evaluator(
    inputs={"question": "FoobarLand 的首任总统出生在哪里？"},
    outputs={"answer": "FoobarLand 的首任总统是 Bagatur Askaryan。"},
)
# score: False（回答只给出了总统姓名，未回答出生地问题）
```

---

### 7. RAG_GROUNDEDNESS_PROMPT（RAG 基础性评估）

| 项目 | 说明 |
|------|------|
| **功能** | 验证模型回答是否严格基于检索到的上下文内容，未编造或过度依赖模型自身知识 |
| **输入参数** | `context`（包含 `documents` 字段的检索文档列表）、`outputs`（包含 `answer` 字段） |
| **评分方式** | 布尔值（True/False）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 逐条检查回答中的声明是否都能在检索文档中找到依据，检测是否引入了上下文之外的虚假信息 |

**业务使用场景：**

- **RAG 幻觉治理**：检测模型是否在回答中添加了检索内容中没有的信息
- **合规审计**：在金融、法律等场景下，确保每个事实声明都有文档依据
- **检索质量间接评估**：如果 grounding 得分低，可能说明检索到的文档不完整
- **Prompt 调优**：评估不同 system prompt 对模型"忠实于上下文"程度的影响

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT

evaluator = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    model="openai:o3-mini",
)

result = evaluator(
    context={"documents": [
        "FoobarLand 是一个位于月球背面的新国家。",
        "FoobarLand 的首任总统是 Bagatur Askaryan。",
    ]},
    outputs={"answer": "FoobarLand 的首任总统是 Bagatur Askaryan。"},
)
# score: True（回答内容完全由检索文档支持）
```

---

### 8. RAG_RETRIEVAL_RELEVANCE_PROMPT（RAG 检索相关性评估）

| 项目 | 说明 |
|------|------|
| **功能** | 评估检索阶段返回的文档与用户查询的相关性，衡量检索质量 |
| **输入参数** | `inputs`（包含 `question` 字段）、`context`（包含 `documents` 字段的检索文档列表） |
| **评分方式** | 布尔值（True/False）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 判断检索到的文档是否包含回答用户问题所需的信息，不关心最终生成的回答 |

**业务使用场景：**

- **检索器优化**：评估不同向量数据库、检索算法（如 BM25 vs 向量检索）的检索效果
- **top-k 调优**：确定最优的检索文档数量，平衡召回率与精确率
- **知识库维护**：发现检索不到的问题，反向指导知识库补充
- **多路检索融合**：对比多路检索策略（如混合检索）的效果

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import RAG_RETRIEVAL_RELEVANCE_PROMPT

evaluator = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
    model="openai:o3-mini",
)

result = evaluator(
    inputs={"question": "FoobarLand 的首任总统出生在哪里？"},
    context={"documents": [
        "FoobarLand 位于月球背面。",
        "FoobarLand 的首任总统是 Bagatur Askaryan。",
        "FoobarLand 当前天气晴朗，80度。",
    ]},
)
# score: False（检索文档未包含总统出生地信息）
```

---

## 三、代码评估

### 9. CODE_CORRECTNESS_PROMPT（代码正确性评估）

| 项目 | 说明 |
|------|------|
| **功能** | 评估 LLM 生成的代码是否正确、可执行，是否满足用户提出的代码要求 |
| **输入参数** | `inputs`（代码任务描述）、`outputs`（生成的代码） |
| **评分方式** | 布尔值（True/False）或连续值（0.0~1.0） |
| **评估逻辑** | 通过 `create_code_llm_as_judge` 调用，内置代码提取功能（支持 markdown 代码块提取或 LLM 提取），然后由裁判 LLM 分析代码逻辑正确性 |

**业务使用场景：**

- **代码生成工具评估**：评估 Copilot 类产品的代码生成质量
- **编程教学辅助**：检查学生提交的代码是否满足题目要求
- **代码重构验证**：评估代码重构后是否保持了原有功能
- **CI/CD 集成**：在持续集成流程中自动评估 AI 生成代码的正确性

**示例代码：**

```python
from openevals.code.llm import create_code_llm_as_judge
from openevals.prompts import CODE_CORRECTNESS_PROMPT

evaluator = create_code_llm_as_judge(
    prompt=CODE_CORRECTNESS_PROMPT,
    model="openai:o3-mini",
    code_extraction_strategy="markdown_code_blocks",
)

result = evaluator(
    inputs="将以下代码改写为异步版本：

def sum_of_two(a, b): return a + b",
    outputs="```python
import asyncio

async def sum_of_two(a, b): return a + b
```",
)
print(result["score"])
```

---

### 10. CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS（带参考的代码评估）

| 项目 | 说明 |
|------|------|
| **功能** | 在代码正确性评估的基础上，引入标准答案（reference_outputs）进行对比评估，判断生成代码是否与参考答案等价或更优 |
| **输入参数** | `inputs`（代码任务描述）、`outputs`（生成代码）、`reference_outputs`（标准参考答案代码） |
| **评分方式** | 布尔值（True/False）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 同时对比生成代码和参考答案，评估功能等价性、代码质量和实现完整性 |

**业务使用场景：**

- **编程考试评分**：将学生的代码与标准答案对比，自动评分
- **代码竞赛裁判**：评估参赛者代码是否达到参考答案的效果
- **算法实现验证**：在已知最优解的情况下，验证 AI 生成的算法是否正确
- **代码迁移评估**：将代码从一种语言/框架迁移到另一种时，对比迁移前后功能一致性

**示例代码：**

```python
from openevals.code.llm import create_code_llm_as_judge
from openevals.prompts import CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS

evaluator = create_code_llm_as_judge(
    prompt=CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS,
    model="openai:o3-mini",
    code_extraction_strategy="markdown_code_blocks",
)

result = evaluator(
    inputs="实现一个快速排序算法",
    outputs="```python
def quick_sort(arr): ...```",
    reference_outputs="```python
def quick_sort(arr): if len(arr) <= 1: return arr ...```",
)
print(result["score"])
```

---

## 四、Agent 评估

### 11. PLAN_ADHERENCE_PROMPT（计划遵循度评估）

| 项目 | 说明 |
|------|------|
| **功能** | 评估 Agent 在执行多步任务时，是否严格按照预定义的计划步骤执行 |
| **输入参数** | `inputs`（用户任务）、`outputs`（包含 `steps` 执行步骤列表和 `answer` 最终回答）、`plan`（包含 `steps` 计划步骤列表） |
| **评分方式** | 布尔值（True/False）或连续值（0.0~1.0） |
| **评估逻辑** | 裁判 LLM 对比 Agent 实际执行的步骤与预定义计划，检查是否有遗漏、跳过或额外步骤 |

**业务使用场景：**

- **Agent 工作流验证**：确保 Agent 按照设计好的业务流程执行，不遗漏关键步骤
- **自动化流程审计**：在金融交易、医疗诊断等场景下，审计 Agent 的操作是否符合标准流程
- **Agent 调试**：当 Agent 行为异常时，定位是哪一步骤偏离了计划
- **多步推理评估**：评估 Agent 在复杂任务中的规划能力和执行一致性

**示例代码：**

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import PLAN_ADHERENCE_PROMPT

evaluator = create_llm_as_judge(
    prompt=PLAN_ADHERENCE_PROMPT,
    feedback_key="plan_adherence",
    model="openai:o3-mini",
)

result = evaluator(
    inputs="查询 FoobarLand 的首任总统信息",
    plan={"steps": [
        "搜索 FoobarLand 首任总统的定义",
        "获取总统的详细信息",
        "提供简洁的总结回答",
    ]},
    outputs={
        "steps": [
            "我在 Google 上搜索了 FoobarLand 首任总统的信息",
            "我找到了总统的姓名和任期",
            "我总结了总统的基本信息",
        ],
        "answer": "FoobarLand 的首任总统是 Bagatur Askaryan。",
    },
)
print(result["score"])
```

---

## 五、内置提示词总览表

| 序号 | 提示词名称 | 功能 | 评估维度 | 需要参考 | 适用场景 |
|------|-----------|------|---------|---------|---------|
| 1 | CORRECTNESS_PROMPT | 正确性评估 | 答案准确性 | 可选（reference_outputs） | 问答系统、知识问答 |
| 2 | CONCISENESS_PROMPT | 简洁性评估 | 回答精炼度 | 不需要 | 客服对话、摘要生成 |
| 3 | ANSWER_RELEVANCE_PROMPT | 相关性评估 | 回答切题度 | 不需要 | 搜索引擎、FAQ 系统 |
| 4 | HALLUCINATION_PROMPT | 幻觉检测 | 信息真实性 | 可选（context） | RAG 系统、事实核查 |
| 5 | TOXICITY_PROMPT | 毒性检测 | 内容安全性 | 不需要 | 内容安全、合规审计 |
| 6 | RAG_HELPFULNESS_PROMPT | RAG 帮助性 | 问题解决度 | 不需要 | RAG 端到端评估 |
| 7 | RAG_GROUNDEDNESS_PROMPT | RAG 基础性 | 上下文忠实度 | 需要（context） | RAG 幻觉治理、合规 |
| 8 | RAG_RETRIEVAL_RELEVANCE_PROMPT | RAG 检索相关性 | 检索质量 | 不需要 | 检索器优化、top-k 调优 |
| 9 | CODE_CORRECTNESS_PROMPT | 代码正确性 | 代码功能正确性 | 不需要 | 代码生成、编程教学 |
| 10 | CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS | 带参考的代码评估 | 代码等价性 | 需要（reference_outputs） | 编程考试、算法验证 |
| 11 | PLAN_ADHERENCE_PROMPT | 计划遵循度 | 流程执行一致性 | 需要（plan） | Agent 工作流、自动化审计 |

---

## 六、使用建议

### 1. 评估场景选择

- **通用问答**：优先使用 `CORRECTNESS_PROMPT` + `ANSWER_RELEVANCE_PROMPT` + `CONCISENESS_PROMPT`
- **RAG 系统**：使用 `RAG_HELPFULNESS_PROMPT` + `RAG_GROUNDEDNESS_PROMPT` + `RAG_RETRIEVAL_RELEVANCE_PROMPT`
- **代码生成**：使用 `CODE_CORRECTNESS_PROMPT`（无标准答案）或 `CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS`（有标准答案）
- **Agent 应用**：使用 `PLAN_ADHERENCE_PROMPT` + `HALLUCINATION_PROMPT`
- **安全合规**：使用 `TOXICITY_PROMPT`

### 2. 评分模式选择

- **快速筛选**：使用默认布尔值（True/False），适合批量测试
- **精细分析**：设置 `continuous=True`，获得 0.0~1.0 的连续分数，适合趋势分析和模型对比
- **自定义评分**：设置 `choices` 参数定义离散评分档位（如 0.0、0.5、1.0）

### 3. 评估模型选择

- 推荐使用 `openai:o3-mini` 或 `openai:gpt-4o` 作为裁判模型
- 也可使用 `anthropic:claude-3-5-sonnet-latest` 等替代模型
- 裁判模型的质量直接影响评估结果的可靠性，建议选用能力较强的模型

### 4. 批量评估最佳实践

- 构建覆盖多种场景的测试用例集（至少 50~100 条）
- 定期运行评估，跟踪模型/提示词迭代的效果变化
- 结合 LangSmith 等平台进行评估结果可视化和趋势分析
- 对于关键指标（如毒性、幻觉），设置阈值告警机制
