# LangChain 多文件 RAG 执行流程说明

## 1. 脚本总体目标

这个脚本实现了一个“基于多文件文档的问答系统（RAG）”流程：

- 读取 `docs/` 目录中的多个文本文件
- 将文本切分成小块并做向量化
- 建立 FAISS 向量索引
- 根据用户问题做相似度检索
- 将召回的上下文拼接到提示词里
- 调用大模型生成最终回答

---

## 2. 代码执行主流程

### 第一步：读取环境变量

```python
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
```

作用：

- 从环境变量中读取阿里云百炼（DashScope）访问密钥
- 如果没有配置，程序直接报错退出

---

### 第二步：加载文档并创建/恢复向量索引

#### 入口函数

```python
vector_store = load_documents_and_create_index()
```

#### 执行逻辑

1. 先尝试从 `persist_dir`（默认是 `./langchain_storage`）加载已有 FAISS 索引
2. 如果成功，则直接复用已有索引
3. 如果失败或不存在，则重新读取 `docs/` 下的 `.txt` 文件
4. 对文件内容做分块
5. 用 `DashScopeEmbeddings` 进行向量化
6. 构建 FAISS 索引并保存到本地目录

#### 关键技术点

- `DirectoryLoader`：批量加载目录下的文件
- `TextLoader`：按文本文件方式读取
- `RecursiveCharacterTextSplitter`：递归文本切分器，用于把长文档切成多个 chunk
- `DashScopeEmbeddings`：把文本转成 embedding 向量
- `FAISS.from_documents(...)`：基于 chunk 和 embedding 创建向量库
- `vector_store.save_local(...)`：把索引持久化到本地

---

### 第三步：创建问答链（LCEL）

```python
qa_chain = create_qa_chain(llm)
```

#### 作用

这是一个“提示词模板 + 大模型 + 输出解析器”的组合链路：

```python
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个乐于助人的AI助手。
根据以下上下文内容回答用户的问题。如果上下文中没有相关信息，请如实说明。
你总是用中文回复用户。

上下文内容:
{context}"""),
    ("human", "{question}")
])

qa_chain = qa_prompt | llm | StrOutputParser()
```

#### 这里用到的技术点

- `ChatPromptTemplate`：定义标准聊天提示词模板
- `system` 和 `human` 两个角色：
  - `system`：给模型身份和回答规则
  - `human`：接收用户真正的问题
- `llm`：具体的大模型对象，例如 `ChatTongyi`
- `StrOutputParser`：把模型输出解析为纯字符串结果
- `|`：LangChain LCEL 的链式拼接语法，类似管道

---

### 第四步：用户查询与相似度检索

```python
docs = vector_store.similarity_search(query, k=5)
```

#### 作用

- 使用用户输入的问题 `query`
- 在 FAISS 向量库中找出最相关的前 5 个文本片段
- 这些片段就是 RAG 中的“检索结果”

#### 结果用途

- 先打印出召回的文档片段，观察检索质量
- 再把这些片段拼接成 `context`

```python
context = "\n\n".join(doc.page_content for doc in docs)
```

---

### 第五步：执行 `qa_chain.invoke`

```python
response = qa_chain.invoke({"context": context, "question": query})
```

#### `qa_chain.invoke` 的具体作用

`qa_chain.invoke(...)` 是“执行问答链”的核心调用。

它会把输入字典里的内容送入整个 LCEL 链：

- `context`：相似度检索出来的相关文档片段
- `question`：用户的原始问题

然后链路按顺序执行：

1. `qa_prompt` 根据模板生成最终请求消息
2. `llm` 把这个提示词交给大模型
3. `StrOutputParser` 把模型原始输出转换为字符串结果
4. 返回最终回答给 `response`

换句话说：

`qa_chain.invoke` 就是“把上下文 + 问题喂给大模型并拿回答案”的统一入口。

---

## 3. 这个脚本里最关键的技术点总结

### 3.1 RAG（Retrieval-Augmented Generation）

这是这个项目最核心的架构：

- 检索阶段：用向量库找相关文档片段
- 生成阶段：把相关片段拼到提示词里，让模型基于上下文回答

它的优点是：

- 减少模型“幻觉”
- 让回答更贴近业务知识库
- 支持大规模文档问答

---

### 3.2 向量数据库：FAISS

FAISS 用于把文本片段映射成向量，并执行相似度搜索。

这让系统可以快速找出与用户问题最相关的内容，而不是直接对全文做粗暴匹配。

---

### 3.3 Embedding 嵌入模型

`DashScopeEmbeddings` 会把文本转成高维向量表示。

这一步非常关键，因为：

- 语义相近的文本会在向量空间中靠近
- 查询也会变成向量后再去检索

---

### 3.4 LCEL（LangChain Expression Language）

这个脚本使用的是 `qa_prompt | llm | StrOutputParser()` 这种链式写法。

优点：

- 代码更直观
- 链路清晰，可读性强
- 更适合 LangChain 的新一代写法

---

## 4. 一个最简化的理解版本

可以把整个流程概括为下面这几步：

1. 读文件
2. 切成 chunk
3. 用 embedding 变成向量
4. 建立 FAISS 索引
5. 用户输入问题
6. 从 FAISS 中检索相关文档片段
7. 把这些片段拼成 `context`
8. 调用 `qa_chain.invoke(context, question)`
9. 模型基于上下文返回答案

---

## 5. 运行结果中的关键输出含义

脚本会输出：

- 加载了多少个文档
- 文本被切成多少个 chunk
- 检索到了哪些相关片段
- 最终的 AI 回复

其中：

- `docs = vector_store.similarity_search(query, k=5)` 决定了“检索召回多少段上下文”
- `qa_chain.invoke(...)` 决定了“如何根据这些上下文生成答案”

---

## 6. 结束语

这个脚本体现了一个完整的 LangChain RAG 范式：

- 文档加载
- 文本切分
- 向量化
- 向量检索
- 提示词构造
- 大模型生成

如果你要继续扩展，这个脚本最适合往下做的方向包括：

- 支持 `.docx`、`.pdf` 等文件类型
- 加入聊天记录记忆机制
- 增加多轮问答上下文
- 将检索结果做重排（rerank）
- 使用更强的 LLM 模型做答案汇总
