import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Updated for better table handling
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import RetrievalQA
# --- STEP 1: CREATE SAMPLE PDF WITH A REAL TABLE ---
def create_sample_pdf(filename="table_test.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("Financial Report 2026", styles['Title']))
    
    # This data structure is what we want the AI to read correctly
    data = [
        ['Quarter', 'Revenue', 'Profit'],
        ['Q1', '$10,000', '$2,000'],
        ['Q2', '$12,500', '$3,100'],
        ['Q3', '$11,000', '$2,800'],
        ['Q4', '$15,000', '$4,500']
    ]
    
    t = Table(data)
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke)
    ]))
    
    elements.append(t)
    doc.build(elements)
    print(f"Generated {filename}")

# --- STEP 2: TABLE-AWARE RAG PIPELINE ---
def start_table_rag(filename="table_test.pdf"):
    # 1. Use PDFPlumber for better layout/table retention
    loader = PDFPlumberLoader(filename)
    pages = loader.load()

    # 2. Chunking (Keep chunks larger to ensure whole tables stay together)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    # 3. Embeddings (Local via Ollama)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Vector Store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_table_storage"
    )

    # 5. LLM (Ensure Ollama is running)
    llm = ChatOllama(model="llama3", temperature=0)

    # 6. Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3})
    )

    # 7. Targeted Query
    query = "Look at the table. What was the Profit in Q3?"
    response = qa_chain.invoke(query)

    print(f"\nQUERY: {query}")
    print(f"ANSWER: {response['result']}")

if __name__ == "__main__":
    # Ensure you run: pip install pdfplumber langchain-ollama langchain-chroma
    create_sample_pdf()
    start_table_rag()