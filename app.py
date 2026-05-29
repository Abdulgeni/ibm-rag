import streamlit as st
import tempfile
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

st.set_page_config(page_title="IBM Guided RAG", page_icon="🔵")
st.title("🔵 IBM-Style Guided RAG with LangChain Patterns")
st.markdown("Production-grade RAG with retrieval pipeline and response grounding.")

with st.sidebar:
    st.header("🔵 Production RAG Architecture")
    st.markdown("""
    **IBM Enterprise Pattern:**
    1. Document ingestion
    2. Intelligent chunking
    3. Embedding pipeline
    4. Retrieval with grounding
    5. Response validation
    
    ---
    🆓 100% Free
    """)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

uploaded_files = st.file_uploader("Upload PDFs:", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    chunk_metadata = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        pdf_reader = PyPDF2.PdfReader(tmp_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Production-style chunking with overlap
        words = text.split()
        chunk_size = 200
        overlap = 50
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 100:
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'source': uploaded_file.name,
                    'chunk_index': len(all_chunks),
                    'word_count': len(chunk.split())
                })
    
    st.success(f"📦 {len(all_chunks)} chunks with metadata")
    
    # ChromaDB
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("ibm_rag")
    except:
        pass
    collection = chroma_client.create_collection("ibm_rag")
    
    with st.spinner("🔢 Creating embeddings..."):
        embeddings = model.encode(all_chunks)
        for i, emb in enumerate(embeddings):
            collection.add(
                embeddings=[emb.tolist()],
                documents=[all_chunks[i]],
                metadatas=[chunk_metadata[i]],
                ids=[str(i)]
            )
    
    st.success("✅ Indexed in ChromaDB vector store")
    
    question = st.text_input("Ask a question:")
    
    if question:
        with st.spinner("🔍 Retrieving with grounding..."):
            q_embedding = model.encode([question])[0]
            
            # Retrieve
            results = collection.query(
                query_embeddings=[q_embedding.tolist()],
                n_results=3
            )
            
            # Grounding check
            st.subheader("📋 Grounded Response")
            st.markdown("**Retrieved Context (with sources):**")
            
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                with st.expander(f"Source {i+1}: {meta['source']} (Score: {results['distances'][0][i]:.2f})"):
                    st.write(doc[:500])
            
            # Validation
            avg_distance = np.mean(results['distances'][0])
            grounding = "STRONG" if avg_distance < 1.0 else "MODERATE" if avg_distance < 1.5 else "WEAK"
            
            st.metric("Grounding Quality", grounding)