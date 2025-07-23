import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from logger import get_logger

logger = get_logger(__name__)

PDF_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
HASH_FILE = os.path.join(INDEX_DIR, ".doc_hash")

def hash_docs(pdf_paths):
    hash_md5 = hashlib.md5()
    for path in sorted(pdf_paths):
        with open(path, 'rb') as f:
            hash_md5.update(f.read())
    return hash_md5.hexdigest()

def load_documents():
    all_docs = []
    logger.info("Loading PDF documents from '%s'...", PDF_DIR)
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_DIR, filename)
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                logger.info("  - Loaded %s (%d pages)", filename, len(docs))
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load %s: %s", filename, e)
                continue
    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

def build_or_load_vectorstore():
    os.makedirs(INDEX_DIR, exist_ok=True)
    pdf_paths = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    current_hash = hash_docs(pdf_paths)

    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            previous_hash = f.read().strip()
        logger.info("Checking existing FAISS index...")
        if previous_hash == current_hash and os.path.exists(INDEX_DIR):
            logger.info("Loading existing FAISS index from '%s'...", INDEX_DIR)
            try:
                return FAISS.load_local(INDEX_DIR, OllamaEmbeddings(model="nomic-embed-text:latest"), allow_dangerous_deserialization=True)
            except Exception as e:
                logger.error(f"Failed to load FAISS index: %s", e)
                logger.info("Rebuilding FAISS index...")

    logger.info("Rebuilding FAISS index due to PDF changes or loading failure...")
    documents = load_documents()
    if not documents:
        logger.error("No valid PDF documents found in '%s'. Cannot build vector store.", PDF_DIR)
        return None  # Return None to indicate failure
    logger.info(f"Loaded {len(documents)} documents for processing.")
    chunks = split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks.")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    logger.info("Embedding documents")
    try:
        vectorstore = FAISS.from_documents(tqdm(chunks, desc="🔍 Embedding chunks"), embeddings)
    except Exception as e:
        logger.error(f"Embedding failed: %s", e)
        raise
    logger.info("Saving FAISS index")
    try:
        vectorstore.save_local(INDEX_DIR)
        logger.info("Saved vectorstore successfully.")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: %s", e)
        raise
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)
    return vectorstore

def retrieve_context(query, k=10):
    logger.info(f"Retrieving RAG context for query: {query}")
    try:
        vs = build_or_load_vectorstore()
        if vs is None:
            logger.warning("No vector store available due to missing documents.")
            return "No relevant agricultural knowledge found. Please add PDF documents to 'data/docs/'."
        ret = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = ret.invoke(query)
        logger.info(f"Retrieved {len(docs)} relevant chunks.")
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Context retrieval failed: %s", e)
        return "No relevant agricultural knowledge found. Please add PDF documents to 'data/docs/'."