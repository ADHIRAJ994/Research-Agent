import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_core.documents import Document
from config import FAISS_INDEX_PATH
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


from pathlib import Path
import os

def build_or_load_faiss_index(
    source_dir: str,
    index_path: str = FAISS_INDEX_PATH
):
    embeddings = get_embeddings_model()

    # 1. Load existing FAISS index
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # 2. Resolve knowledge base path
    project_root = Path(__file__).resolve().parents[1]

    source_path = Path(source_dir)

    if not source_path.is_absolute():
        source_path = project_root / source_path

    if not source_path.is_dir():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {source_path}"
        )

    # 3. Find TXT files
    txt_files = sorted(
        file for file in source_path.rglob("*")
        if file.is_file() and file.suffix.lower() == ".txt"
    )

    if not txt_files:
        raise ValueError(
            f"No .txt files found in '{source_path}'."
        )

    # 4. Load documents using Python encoding support
    docs = []

    for file_path in txt_files:

        raw = file_path.read_bytes()

        # Detect UTF-16 BOM
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")

        # Otherwise try UTF-8
        else:
            try:
                text = raw.decode("utf-8-sig")

            except UnicodeDecodeError:
                try:
                    text = raw.decode("cp1252")

                except UnicodeDecodeError:
                    text = raw.decode("latin-1")

        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(file_path)}
                )
            )

    if not docs:
        raise ValueError("No readable text documents found.")

    # 5. Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No chunks generated from documents.")

    # 6. Create FAISS index
    index = FAISS.from_documents(
        chunks,
        embeddings
    )

    # 7. Save index
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


if __name__ == "__main__":

    index = build_or_load_faiss_index(
        "./data/knowledge_base"
    )

    faiss_retrieve = make_faiss_retrieve_tool(index)

    results = faiss_retrieve(
        "what did we conclude about RAG evaluation"
    )

    print(results)