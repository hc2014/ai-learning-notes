# AGICTO 多文件 RAG 业务与技术梳理

## 1. 业务目标

本脚本实现的是一个“多文件知识库问答（RAG）”流程：

- 读取指定目录下的多个文本文件；
- 对文档进行切片与向量化；
- 通过 FAISS 建立本地向量索引；
- 根据用户问题进行相似度检索；
- 将召回的上下文拼接后交给大模型生成回答。

核心目标是：让模型不是直接回答泛泛问题，而是先依据知识库中的文档内容来回答，从而提高回答的针对性和准确性。

---

## 2. 业务流程

### 2.1 配置环境变量

脚本启动时读取以下环境变量：

- `AGICTO_API_KEY`：必填，作为访问 AGICTO 的认证密钥。
- `AGICTO_BASE_URL`：可选，默认使用 AGICTO 的 OpenAI 兼容地址。
- `AGICTO_MODEL`：可选，默认使用 `qwen-plus`。
- `AGICTO_EMBEDDING_MODEL`：可选，默认使用 `text-embedding-v3`。

这部分配置决定了：

- 大模型是谁来回答；
- 嵌入模型是谁来把文档转成向量；
- 请求的协议地址是什么。

### 2.2 初始化 LLM 与 Embeddings

脚本使用的是 OpenAI 兼容接口封装：

- `ChatOpenAI`：用于大模型问答；
- `OpenAIEmbeddings`：用于把文档切片转成向量。

这意味着它并不是直接走 DashScope 原生 SDK，而是走“兼容 OpenAI 协议”的调用方式。

### 2.3 加载文档并构建索引

函数 `load_documents_and_create_index()` 的职责是：

1. 指定文档目录 `./agent-multi-files/docs`；
2. 如果本地持久化目录 `./langchain_storage_agicto` 已存在，则尝试直接加载已有索引；
3. 如果不存在，则重新扫描目录中的 `.txt` 文件；
4. 使用 `DirectoryLoader + TextLoader` 读入文本；
5. 通过 `RecursiveCharacterTextSplitter` 做分块；
6. 用 embedding 模型将文本块编码成向量；
7. 调用 `FAISS.from_documents()` 创建向量库；
8. 最后用 `save_local()` 持久化到本地目录。

这一步是整个 RAG 的“知识库构建”阶段。

### 2.4 创建问答链

函数 `create_qa_chain()` 负责搭建问答链：

- `ChatPromptTemplate` 组织 system / user 的输入格式；
- `llm` 负责生成回答；
- `StrOutputParser` 将模型返回结果转换成字符串。

实际链路是：

`Prompt -> LLM -> OutputParser`

### 2.5 用户问题检索与回答

在 `main()` 中：

1. 调用 `load_documents_and_create_index()`；
2. 通过 `vector_store.similarity_search(query, k=5)` 去检索最相关的 5 个文本片段；
3. 把这些片段拼接成 `context`；
4. 再把 `context + question` 送入问答链；
5. 输出模型最终答案。

这套逻辑就是典型的 RAG：

- 检索（Retrieval）
- 增强（Augmentation）
- 生成（Generation）

---

## 3. 主要技术栈

### 3.1 LangChain

这是整个流程的核心编排框架：

- `langchain_core.prompts.ChatPromptTemplate`
- `langchain_core.output_parsers.StrOutputParser`
- `langchain_openai.ChatOpenAI`
- `langchain_openai.OpenAIEmbeddings`

LangChain 负责把 prompt、LLM、embedding 和检索流程串起来。

### 3.2 FAISS

`FAISS` 用作本地向量检索引擎：

- `FAISS.from_documents()`：从文本块创建向量库；
- `FAISS.load_local()`：从本地目录加载已有索引；
- `similarity_search()`：根据用户问题进行近似检索。

### 3.3 文档加载与切分

- `DirectoryLoader`：批量加载目录中的文本文件；
- `TextLoader`：按文本文件方式读取内容；
- `RecursiveCharacterTextSplitter`：按长度切分文本块，便于向量化。

### 3.4 AGICTO OpenAI 兼容接口

这份脚本不是直接用 DashScope 原生 SDK，而是通过 OpenAI 接口风格来调用：

- `base_url` 使用 AGICTO 的兼容地址；
- `api_key` 使用 `AGICTO_API_KEY`；
- `model` 走 `qwen-plus` 这样的兼容模型名。

这样做的优点是：

- 兼容 LangChain 中的 `ChatOpenAI` / `OpenAIEmbeddings` 生态；
- 接口调用方式更统一；
- 对已有 OpenAI 风格代码迁移成本较低。

---

## 4. 程序结构说明

### 4.1 模块职责拆分

- `load_documents_and_create_index()`：文档加载、切分、索引构建、索引加载。
- `create_qa_chain()`：构建问答链。
- `main()`：主流程入口，组织“加载索引 -> 检索 -> 生成答案”。

### 4.2 当前目录约定

- 文档目录：`./agent-multi-files/docs`
- 索引目录：`./langchain_storage_agicto`

这意味着脚本默认认为知识库来源在项目目录下的文档目录中。

---

## 5. 已知问题与处理方式

### 5.1 老索引与新 embedding 维度不匹配

这个项目中曾经使用过旧版的向量索引目录 `./langchain_storage`。

如果旧索引目录保留不动，而新脚本又切换了 embedding 模型，可能会出现 FAISS 维度不匹配错误：

- 旧索引是按老 embedding 维度保存的；
- 新 embedding 生成的向量维度不同；
- 于是 FAISS 在检索时触发断言失败。

当前脚本通过使用新的持久化目录 `./langchain_storage_agicto`，并在加载失败时清理旧缓存，避免直接复用旧索引造成冲突。

---

## 6. 总结

这份脚本的核心业务是：

- 用多文件文本知识库构建 RAG 向量检索系统；
- 以 AGICTO 的 OpenAI 兼容接口来调用大模型与 embedding；
- 通过 LangChain 与 FAISS 组织完整的“检索增强生成”流程。

从技术角度看，它体现了一个典型的企业内部知识问答实现：

- 文档知识库
- 嵌入向量化
- 本地 FAISS 检索
- 大模型上下文生成

这也是现在常见的“知识库问答”落地方案之一。
