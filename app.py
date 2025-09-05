import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# Load HuggingFace model
# -------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=hf_pipeline)

llm = load_llm()

# -------------------------------
# Build FAISS index from repo docs
# -------------------------------
@st.cache_resource
def build_vectorstore():
    loader = DirectoryLoader("data", glob="**/*.*", loader_cls=UnstructuredFileLoader, recursive=True)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

retriever = build_vectorstore()

# -------------------------------
# Build RetrievalQA chain
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
st.title("📚 RAG Chatbot")

query = st.text_input("❓ Ask a question:")

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
