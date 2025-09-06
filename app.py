import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# Load HuggingFace model
# -------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

llm = load_llm()

# -------------------------------
# Build FAISS index from PDFs/DOCX
# -------------------------------
@st.cache_resource
def build_vectorstore():
    documents = []

    # Load all PDFs
    pdf_loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader, recursive=True)
    documents.extend(pdf_loader.load())

    # Load all DOCX
    docx_loader = DirectoryLoader("data", glob="*.docx", loader_cls=Docx2txtLoader, recursive=True)
    documents.extend(docx_loader.load())

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    # Create embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

retriever = build_vectorstore()

# -------------------------------
# RetrievalQA
# -------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="📚 RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot for My Documents")

query = st.text_input("❓ Ask a question about your data:")

if query:
    with st.spinner("Thinking..."):
        result = qa(query)

    st.markdown("### 💡 Answer:")
    st.write(result["result"])

    st.markdown("### 📂 Sources:")
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "unknown_file")
        page = doc.metadata.get("page", "?")
        st.write(f"{i}. **{source}** (page {page})")
