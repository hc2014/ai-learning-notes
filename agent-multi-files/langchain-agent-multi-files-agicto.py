#!/usr/bin/env python
# coding: utf-8
"""
基于 LangChain 的多文件 RAG 应用（改为使用 AGICTO API Key + OpenAI 兼容接口）
支持加载 docs 文件夹下的多种格式文件进行问答。

环境变量：
- AGICTO_API_KEY：必填
- AGICTO_BASE_URL：可选，默认使用阿里云百炼兼容地址
- AGICTO_MODEL：可选，默认 qwen-plus
- AGICTO_EMBEDDING_MODEL：可选，默认 text-embedding-v3
"""

import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================
# 1. AGICTO 配置
# ==========================
AGICTO_API_KEY = os.getenv("AGICTO_API_KEY")
if not AGICTO_API_KEY:
    raise ValueError("请设置环境变量 AGICTO_API_KEY")

AGICTO_BASE_URL = os.getenv(
    "AGICTO_BASE_URL",
    "https://api.agicto.cn/v1",
)
AGICTO_MODEL = os.getenv("AGICTO_MODEL", "qwen-plus")
AGICTO_EMBEDDING_MODEL = os.getenv("AGICTO_EMBEDDING_MODEL", "text-embedding-v3")


# ==========================
# 2. 初始化模型
# ==========================
llm = ChatOpenAI(
    model=AGICTO_MODEL,
    api_key=AGICTO_API_KEY,
    base_url=AGICTO_BASE_URL,
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model=AGICTO_EMBEDDING_MODEL,
    api_key=AGICTO_API_KEY,
    base_url=AGICTO_BASE_URL,
    check_embedding_ctx_length=False,
    chunk_size=10,
)


# ==========================
# 3. 加载文档并创建索引
# ==========================
def load_documents_and_create_index(
    file_dir: str = './agent-multi-files/docs',
    persist_dir: str = './langchain_storage_agicto',
):
    """加载文档文件夹中的所有文件并创建向量索引"""

    # 如果索引已存在，直接加载
    if os.path.exists(persist_dir):
        try:
            vector_store = FAISS.load_local(
                persist_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print("从存储加载索引成功")
            return vector_store
        except Exception as exc:
            print(f"加载索引失败: {exc}，将清理旧缓存并重新创建索引")
            import shutil
            shutil.rmtree(persist_dir, ignore_errors=True)

    if not os.path.exists(file_dir):
        print(f"文档目录 {file_dir} 不存在")
        return None

    loader = DirectoryLoader(
        file_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")

    if not documents:
        print("没有找到任何文档")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文本被分割成 {len(chunks)} 个块")

    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    vector_store.save_local(persist_dir)
    print(f"索引已保存到 {persist_dir}")

    return vector_store


# ==========================
# 4. 创建问答链
# ==========================
def create_qa_chain():
    """创建 QA 问答链（LCEL 写法）"""

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个乐于助人的 AI 助手。
根据以下上下文内容回答用户的问题。如果上下文中没有相关信息，请如实说明。
你总是用中文回复用户。

上下文内容:
{context}""",
        ),
        ("human", "{question}"),
    ])

    return qa_prompt | llm | StrOutputParser()


# ==========================
# 5. 主函数
# ==========================
def main():
    """主函数"""
    vector_store = load_documents_and_create_index()
    if vector_store is None:
        print("无法创建索引，程序退出")
        return

    qa_chain = create_qa_chain()

    query = "介绍下雇主责任险"
    print(f"\n用户查询: {query}\n")

    docs = vector_store.similarity_search(query, k=5)
    print("===== 召回的文档内容 =====")
    if docs:
        for i, doc in enumerate(docs):
            print(f"\n文档片段 {i + 1}:")
            print(f"内容: {doc.page_content[:200]}...")
            print(f"来源: {doc.metadata.get('source', '未知')}")
    else:
        print("没有召回任何文档内容")
    print("===========================\n")

    context = "\n\n".join(doc.page_content for doc in docs)

    print("===== AI 回复 =====")
    response = qa_chain.invoke({"context": context, "question": query})
    print(response)
    print("===================\n")


if __name__ == "__main__":
    main()
