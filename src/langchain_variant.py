"""
LangChain drop-in variant (REFERENCE ONLY — not wired into the running app).

This module shows how src/chunking.py, src/ingest.py, and the prompt
template in src/llm_backend.py map almost line-for-line onto LangChain
primitives (`RecursiveCharacterTextSplitter`, `DirectoryLoader`,
`PromptTemplate`, `ChatOpenAI`, `FAISS` vectorstore). It is kept here,
uninstalled and unimported by the rest of the app, so the repository stays
runnable with zero heavy dependencies while still demonstrating LangChain
fluency.

To actually run this variant:

    pip install langchain langchain-community langchain-openai faiss-cpu

    export OPENAI_API_KEY=sk-...
    python -m src.langchain_variant
"""
from __future__ import annotations

# --- Document loading & chunking -------------------------------------------
#
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#
# loader = DirectoryLoader(
#     "data/policy_documents", glob="*.txt", loader_cls=TextLoader
# )
# raw_documents = loader.load()
#
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=700,
#     chunk_overlap=120,
#     separators=["\n\n", "\n", ". ", " "],
# )
# documents = splitter.split_documents(raw_documents)
#
# This is functionally identical to src/chunking.py's `chunk_document`,
# which reimplements the same recursive-separator strategy without the
# LangChain dependency.


# --- Embeddings + vector store -----------------------------------------------
#
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
#
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(documents, embeddings)
# vectorstore.save_local("data/index/faiss_langchain")
#
# Equivalent to src/embeddings.py's SentenceTransformerEmbedder +
# src/vector_index.py's FaissIndex, combined into one call by LangChain's
# `FAISS.from_documents`.


# --- Prompt template + retrieval chain ---------------------------------------
#
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
#
# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=(
#         "You are a financial risk and fraud insights assistant for internal "
#         "bank analysts. Answer the question using ONLY the provided context "
#         "excerpts. Always cite the source document for every claim.\n\n"
#         "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER (with citations):"
#     ),
# )
#
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#     chain_type_kwargs={"prompt": prompt},
#     return_source_documents=True,
# )
#
# This mirrors src/rag_pipeline.py's answer_query(): retrieve top-k chunks,
# format them into the same prompt template defined in
# src/llm_backend.py::PROMPT_TEMPLATE, and call an LLM (OpenAIAnswerBackend
# in this project, ChatOpenAI here) to synthesize the final answer.


def run_example(question: str = "What is the policy on counterparty exposure limits?"):
    """Illustrative entry point. Left unimplemented on purpose — see the
    commented code above for the real LangChain call sequence. Uncomment
    and install the langchain packages listed in the module docstring to
    actually execute this path."""
    raise NotImplementedError(
        "This is a reference/demo module. Install langchain, "
        "langchain-community, langchain-openai, and faiss-cpu, uncomment "
        "the code blocks above, and set OPENAI_API_KEY to run this variant. "
        "The default, fully-offline pipeline lives in src/rag_pipeline.py."
    )


if __name__ == "__main__":
    run_example()
