**Document Exemplar Search Engine**

**Overview**
The Document Exemplar Search Engine is a semantic academic document retrieval system that identifies research papers conceptually similar to a given document. The system uses transformer-based embeddings and a persistent vector database to perform semantic similarity search across both arXiv-sourced papers and user-uploaded PDFs.

**Features**
- Automatic corpus initialization using arXiv papers across multiple academic domains
- Upload and index custom PDF documents with categorization
- Semantic similarity search using all-MiniLM-L6-v2 embeddings
- Persistent vector storage using ChromaDB
- Advanced filtering by domain, document type, and publication date
- Repository management with search, view, delete, and bulk download options

**Tech Stack**
- Python
- Streamlit
- Sentence Transformers (all-MiniLM-L6-v2)
- ChromaDB (persistent storage)
- arXiv API

**Project Structure**
search_engine.py      # Main application
chroma_db/            # Persistent vector database
uploaded_pdfs/        # User-uploaded documents
requirements.txt      # Dependencies

**Run Instructions**
pip install -r requirements.txt
streamlit run search_engine.py

**Use Cases**
Academic literature review, research paper discovery, semantic document comparison, and intelligent repository management.