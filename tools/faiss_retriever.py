import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import FAISS_INDEX_PATH
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_or_load_faiss_index(source_dir: str, index_path: str = FAISS_INDEX_PATH):
    embeddings = get_embeddings_model()

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(source_dir):
        raise FileNotFoundError(
            f"'{source_dir}' doesn't exist. Create it and add at least one .txt file "
            f"before building the FAISS index."
        )

    docs = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(source_dir, filename))
            docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No .txt files found in '{source_dir}'.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(index_path)
    return index

def make_faiss_retrieve_tool(index):
    """Returns a faiss_retrieve function bound to a loaded/built FAISS index."""

    def faiss_retrieve(query: str, k: int = 4) -> list[dict]:
        results = index.similarity_search(query, k=k)
        return [
            {"title": doc.metadata.get("source", "local document"), "content": doc.page_content}
            for doc in results
        ]

    return faiss_retrieve


# Testing the retriever

index = build_or_load_faiss_index(source_dir="./data/knowledge_base")
faiss_retrieve = make_faiss_retrieve_tool(index)

results = faiss_retrieve("what did we conclude about RAG evaluation")
for r in results:
    print(f"- {r['title']}")
    print(f"  {r['content'][:120]}...\n")