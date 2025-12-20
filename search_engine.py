import streamlit as st
from sentence_transformers import SentenceTransformer
import PyPDF2
import io
import tempfile
from typing import List, Dict, Tuple
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import zipfile
from collections import Counter
import re
import os
from dotenv import load_dotenv
from supabase import create_client, Client

ARXIV_CATEGORIES = {
    'cs.AI': 'Artificial Intelligence',
    'cs.LG': 'Machine Learning',
    'cs.CV': 'Computer Vision',
    'cs.CL': 'Computation and Language',
    'cs.NE': 'Neural and Evolutionary Computing',
    'math.CO': 'Combinatorics',
    'physics.comp-ph': 'Computational Physics',
    'q-bio.QM': 'Quantitative Methods',
    'stat.ML': 'Machine Learning (Statistics)',
    'econ.EM': 'Econometrics'
}

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# @st.cache_resource
# def init_chromadb():
#     client = chromadb.Client(Settings(
#         anonymized_telemetry=False,
#         is_persistent=True,
#         persist_directory="./chroma_db"
#     ))
#     return client

load_dotenv()

@st.cache_resource
def init_supabase():
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
    if url and key:
        return create_client(url, key)
    return None

def store_arxiv_pdf_supabase(pdf_bytes: bytes, doc_id: str):

    try:
        supabase = init_supabase()
        if not supabase:
            return False

        file_path = f"{doc_id}.pdf"

        supabase.storage.from_("pdfs").upload(
            file_path,
            pdf_bytes,
            {
                "content-type": "application/pdf",
                "upsert": "true"
            }
        )
        return True

    except Exception as e:
        st.error(f"Error storing PDF in Supabase: {str(e)}")
        return False


def fetch_arxiv_papers(category: str, max_results: int = 20) -> List[Dict]:
    base_url = 'http://export.arxiv.org/api/query?'
    query = f'search_query=cat:{category}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    
    try:
        response = requests.get(base_url + query)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            paper = {
                'id': entry.find('atom:id', ns).text.split('/abs/')[-1],
                'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                'summary': entry.find('atom:summary', ns).text.strip().replace('\n', ' '),
                'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                'published': entry.find('atom:published', ns).text,
                'pdf_url': entry.find('atom:id', ns).text.replace('/abs/', '/pdf/') + '.pdf',
                'category': category
            }
            papers.append(paper)
        
        return papers
    except Exception as e:
        st.error(f"Error fetching papers from {category}: {str(e)}")
        return []

def download_arxiv_pdf(pdf_url: str) -> bytes:
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error downloading PDF: {str(e)}")
        return None

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        max_pages = min(10, len(pdf_reader.pages))
        for i in range(max_pages):
            text += pdf_reader.pages[i].extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return ""

def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        max_pages = min(10, len(pdf_reader.pages))
        for i in range(max_pages):
            text += pdf_reader.pages[i].extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks


def generate_summary_from_abstract(abstract: str) -> str:
    
    if not abstract:
        return "No abstract available."

    abstract = re.sub(r'\s+', ' ', abstract).strip()

    sentences = re.split(r'(?<=[.!?])\s+', abstract)

    summary = " ".join(sentences[:2])

    if len(summary) > 320:
        summary = summary[:320].rsplit(' ', 1)[0] + "..."

    return summary


def add_document(_collection, model, doc_text: str, doc_id: str, metadata: Dict):
    success, message = store_embeddings_supabase(
        model=model,
        doc_id=doc_id,
        text=doc_text,
        metadata=metadata
    )
    return success, message

    
def store_uploaded_pdf_supabase(pdf_file, doc_id: str):
    try:
        supabase = init_supabase()
        if not supabase:
            return False, "Supabase not initialized"
        
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        
        file_path = f"{doc_id}.pdf"
        supabase.storage.from_("pdfs").upload(
            file_path,
            pdf_bytes,
            {"content-type": "application/pdf", "upsert": "true"}
        )
        
        return True, file_path
    except Exception as e:
        return False, str(e)

def get_uploaded_pdf_supabase(doc_id: str):
    try:
        supabase = init_supabase()
        if not supabase:
            return None
        
        file_path = f"{doc_id}.pdf"
        response = supabase.storage.from_("pdfs").download(file_path)
        return response
    except Exception as e:
        return None

def store_embeddings_supabase(model, doc_id: str, text: str, metadata: dict):

    supabase = init_supabase()
    if not supabase:
        return False, "Supabase not initialized"

    chunks = chunk_text(text)

    if not chunks:
        return False, "No text chunks found"

    embeddings = model.encode(chunks).tolist()

    rows = []
    for i, chunk in enumerate(chunks):
        rows.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "content": chunk,
            "metadata": metadata,
            "embedding": embeddings[i]
        })

    try:
        supabase.table("document_chunks").insert(rows).execute()
        return True, "Embeddings stored in Supabase"
    except Exception as e:
        return False, str(e)

def search_embeddings_supabase(model, query_text: str, top_k: int = 5):

    supabase = init_supabase()
    if not supabase:
        return []

    query_embedding = model.encode([query_text])[0].tolist()

    try:
        response = supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": top_k * 5
            }
        ).execute()

        rows = response.data if response.data else []

        doc_scores = {}
        doc_metadata = {}

        for row in rows:
            doc_id = row["doc_id"]
            similarity = 1 - row["distance"]

            if doc_id not in doc_scores:
                doc_scores[doc_id] = []
                doc_metadata[doc_id] = row.get("metadata", {})

            doc_scores[doc_id].append(similarity)

        results = []
        for doc_id, scores in doc_scores.items():
            results.append({
                "doc_id": doc_id,
                "similarity": sum(scores) / len(scores),
                "metadata": doc_metadata[doc_id]
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    except Exception as e:
        st.error(f"Vector search error: {str(e)}")
        return []


def search_similar_documents(_collection, model, query_text: str, top_k: int = 5) -> List[Dict]:
    return search_embeddings_supabase(
        model=model,
        query_text=query_text,
        top_k=top_k
    )

def delete_document_supabase(doc_id: str):

    supabase = init_supabase()
    if not supabase:
        return False, "Supabase not initialized"

    try:
        supabase.table("document_chunks") \
            .delete() \
            .eq("doc_id", doc_id) \
            .execute()

        return True, "Deleted document successfully"

    except Exception as e:
        return False, f"Error deleting document: {str(e)}"


def list_documents_supabase():

    supabase = init_supabase()
    if not supabase:
        return [], {}

    try:
        response = supabase.table("document_chunks") \
            .select("doc_id, metadata") \
            .execute()

        rows = response.data if response.data else []

        doc_ids = set()
        doc_info = {}

        for row in rows:
            doc_id = row.get("doc_id")
            metadata = row.get("metadata", {})

            if doc_id:
                doc_ids.add(doc_id)
                if doc_id not in doc_info:
                    doc_info[doc_id] = metadata

        return list(doc_ids), doc_info

    except Exception as e:
        st.error(f"Error listing documents from Supabase: {str(e)}")
        return [], {}


def initialize_arxiv_corpus(_collection, model):
    doc_ids, _ = list_documents_supabase()

    if len(doc_ids) > 0:
        st.info(f"Corpus already initialized with {len(doc_ids)} documents")
        return True, f"Corpus already initialized with {len(doc_ids)} documents"

    
    st.info("🔄 Initializing corpus with arXiv papers. This will take several minutes...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_categories = len(ARXIV_CATEGORIES)
    papers_per_category = 5
    total_papers = 0
    successful_papers = 0
    
    for idx, (category, category_name) in enumerate(ARXIV_CATEGORIES.items()):
        status_text.text(f"Fetching papers from {category_name}...")
        
        papers = fetch_arxiv_papers(category, papers_per_category)
        
        for paper_idx, paper in enumerate(papers):
            total_papers += 1
            
            try:
                pdf_bytes = download_arxiv_pdf(paper['pdf_url'])
                
                if pdf_bytes:
                    text = extract_text_from_pdf_bytes(pdf_bytes)
                    
                    if text and len(text) > 100:
                        doc_id = paper['id'].replace('/', '_').replace('.', '_')
                        
                        metadata = {
                            'name': paper['title'][:200],
                            'source': f"arXiv:{paper['id']}",
                            'category': category,
                            'category_name': category_name,
                            'authors': ', '.join(paper['authors'][:3]),
                            'published': paper['published'][:10],
                            'type': 'arxiv',
                            'abstract': paper['summary']
                        }
                        
                        success, message = add_document(None, model, text, doc_id, metadata)
   
                        if success:
                            successful_papers += 1
                            if pdf_bytes:
                                store_arxiv_pdf_supabase(pdf_bytes, doc_id)
                
                time.sleep(0.5)
                
            except Exception as e:
                st.warning(f"Failed to process paper {paper['id']}: {str(e)}")
                continue
            
        
        progress = (idx + 1) / total_categories
        progress_bar.progress(progress)
        status_text.text(f"Processed {category_name}: {successful_papers}/{total_papers} papers indexed")
    
    progress_bar.progress(1.0)
    return True, f"Successfully initialized corpus with {successful_papers} papers from {total_categories} domains"

def main():
    st.set_page_config(page_title="Document Exemplar Search Engine", layout="wide")
    
    st.markdown(
        "<h1 style='text-align: center; padding: 20px 0;'>Document Exemplar Search Engine</h1>",
        unsafe_allow_html=True
    )

    st.divider()
    
    model = load_model()
    
    st.sidebar.title("Navigation")
    
    if 'page' not in st.session_state:
        st.session_state.page = "Corpus Status"
    
    page = st.sidebar.radio("Select Operation", 
                           ["Corpus Status", "Add Documents", "Search Documents", "Manage Repository"],
                           index=["Corpus Status", "Add Documents", "Search Documents", "Manage Repository"].index(st.session_state.page))
    
    st.session_state.page = page
    
    # Corpus Status Page
    if page == "Corpus Status":
        st.header("Corpus Status")
        
        doc_ids, doc_info = list_documents_supabase()

        
        if len(doc_ids) == 0:
            st.warning("⚠️ Corpus is empty. Initialize with arXiv papers to get started.")
            
            if st.button("Initialize Corpus with arXiv Papers", type="primary"):
                success, message = initialize_arxiv_corpus(None, model)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.info(message)
        else:
            st.success(f"Corpus contains **{len(doc_ids)}** documents")
            
            st.subheader("Documents by Domain")
            
            category_counts = {}
            for doc_id in doc_ids:
                info = doc_info.get(doc_id, {})
                cat_name = info.get('category_name', 'Other')
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            
            cols = st.columns(3)
            for idx, (cat_name, count) in enumerate(sorted(category_counts.items())):
                with cols[idx % 3]:
                    st.metric(cat_name, count)
            
            st.divider()
            st.subheader("📥 Download More Papers")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_categories = st.multiselect(
                    "Select domains to download papers from:",
                    options=list(ARXIV_CATEGORIES.keys()),
                    format_func=lambda x: f"{ARXIV_CATEGORIES[x]} ({x})",
                    default=[]
                )
            
            with col2:
                papers_to_fetch = st.number_input(
                    "Papers per domain:",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1
                )
            
            if selected_categories:
                if st.button(f"Download {papers_to_fetch} papers from {len(selected_categories)} domain(s)", type="primary"):
                    st.info(f"📄 Fetching new papers from selected domains...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_indexed = 0
                    
                    for idx, category in enumerate(selected_categories):
                        category_name = ARXIV_CATEGORIES[category]
                        status_text.text(f"Fetching papers from {category_name}...")
                        
                        existing_docs, _ = list_documents_supabase()

                        
                        papers_fetched = 0
                        papers_to_fetch_actual = papers_to_fetch
                        max_fetch_attempts = papers_to_fetch * 3 
                        
                        papers = fetch_arxiv_papers(category, max_fetch_attempts)
                        
                        for paper in papers:
                            if papers_fetched >= papers_to_fetch:
                                break
                                
                            try:
                                doc_id = paper['id'].replace('/', '_').replace('.', '_')
                                
                                if doc_id in existing_docs:
                                    continue  
                                
                                pdf_bytes = download_arxiv_pdf(paper['pdf_url'])
                                
                                if pdf_bytes:
                                    text = extract_text_from_pdf_bytes(pdf_bytes)
                                    
                                    if text and len(text) > 100:
                                        metadata = {
                                            'name': paper['title'][:200],
                                            'source': f"arXiv:{paper['id']}",
                                            'category': category,
                                            'category_name': category_name,
                                            'authors': ', '.join(paper['authors'][:3]),
                                            'published': paper['published'][:10],
                                            'type': 'arxiv',
                                            'abstract': paper.get('summary', '')
                                        }
                                        
                                        success, _ = add_document(None, model, text, doc_id, metadata)
   
                                        if success:
                                            total_indexed += 1
                                            papers_fetched += 1
                                            existing_docs.append(doc_id)
                                            if pdf_bytes:
                                                store_arxiv_pdf_supabase(pdf_bytes, doc_id)
                                
                                time.sleep(0.5)
                                
                            except Exception as e:
                                st.warning(f"Failed to process paper {paper['id']}: {str(e)}")
                                continue
                        
                        progress = (idx + 1) / len(selected_categories)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {category_name}: {papers_fetched} new papers indexed")
                    
                    progress_bar.progress(1.0)
                    
                    if total_indexed > 0:
                        st.success(f"Successfully indexed {total_indexed} new papers!")
                    else:
                        st.warning(f"⚠️ No new papers found. All recent papers from selected domains are already in your repository.")
                    
                    time.sleep(2)
                    st.rerun()
            else:
                st.info("👆 Select one or more domains to download additional papers")
            
            st.divider()
            st.subheader("Sample Documents")
            sample_docs = list(doc_ids)[:5]
            for doc_id in sample_docs:
                info = doc_info.get(doc_id, {})
                with st.expander(f"📄 {info.get('name', doc_id)[:100]}..."):
                    st.write("**Source:**", info.get('source', 'N/A'))
                    st.write("**Category:**", info.get('category_name', 'N/A'))
                    st.write("**Authors:**", info.get('authors', 'N/A'))
                    st.write("**Published:**", info.get('published', 'N/A'))
    
    # Add Documents Page
    elif page == "Add Documents":
        st.header("📄 Add Documents to Repository")
        
        st.info("💡 Add your own documents in addition to the arXiv corpus")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents (first 10 pages will be indexed)",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"Loaded {len(uploaded_files)} file(s)")
            
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 {uploaded_file.name}"):
                    doc_name = st.text_input(
                        "Document Name",
                        value=uploaded_file.name.replace('.pdf', ''),
                        key=f"name_{uploaded_file.name}"
                    )
                    doc_source = st.text_input(
                        "Source",
                        key=f"source_{uploaded_file.name}"
                    )
                    
                    _, doc_info = list_documents_supabase()

                    existing_categories = set([info.get('category_name', '') for info in doc_info.values() if info.get('category_name')])
                    existing_categories = sorted(list(existing_categories))
                    
                    category_options = existing_categories + ["Custom", "Enter New Domain"]
                    
                    selected_category = st.selectbox(
                        "Select Category/Domain",
                        options=category_options,
                        key=f"category_select_{uploaded_file.name}"
                    )
                    
                    if selected_category == "Enter New Domain":
                        new_domain = st.text_input(
                            "Enter new domain name",
                            key=f"new_domain_{uploaded_file.name}",
                            placeholder="e.g., Biology, Chemistry, Finance"
                        )
                        final_category = new_domain if new_domain else "Custom"
                    else:
                        final_category = selected_category
                    
                    if st.button(f"Add to Repository", key=f"add_{uploaded_file.name}"):
                        with st.spinner("Processing document..."):
                            text = extract_text_from_pdf(uploaded_file)
                            
                            if text:
                                doc_id = doc_name.replace(' ', '_').replace('/', '_')
                                
                                pdf_stored, pdf_path = store_uploaded_pdf_supabase(uploaded_file, doc_id)
                                
                                metadata = {
                                    "name": doc_name,
                                    "source": doc_source,
                                    "filename": uploaded_file.name,
                                    "category_name": final_category,
                                    "type": "uploaded",
                                    "pdf_stored": pdf_stored
                                }
                                
                                success, message = add_document(None, model, text, doc_id, metadata)
                                
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                            else:
                                st.error("Failed to extract text from PDF")
    
    # Search Documents Page
    elif page == "Search Documents":
        st.header("🔍 Search for Similar Documents")
        
        doc_ids, _ = list_documents_supabase()

        
        if len(doc_ids) == 0:
            st.warning("⚠️ No documents in repository. Please initialize the corpus first.")
            if st.button("Go to Corpus Status"):
                st.rerun()
        else:
            query_file = st.file_uploader("Upload a test document (PDF)", type=['pdf'])
            
            if 'search_results' not in st.session_state:
                st.session_state.search_results = None
            if 'query_text' not in st.session_state:
                st.session_state.query_text = None
            
            if query_file:
                if st.button("Search for Similar Documents", type="primary"):
                    with st.spinner("Searching..."):
                        query_text = extract_text_from_pdf(query_file)
                        
                        if query_text:
                            results = search_similar_documents(None, model, query_text, top_k=100)
                            st.session_state.search_results = results
                            st.session_state.query_text = query_text
                        else:
                            st.error("Failed to extract text from PDF")
            
            if st.session_state.search_results:
                results = st.session_state.search_results
                
                st.success(f"Found {len(results)} similar documents")
                st.divider()
                
                dates = []
                for r in results:
                    pub_date = r['metadata'].get('published', '')
                    if pub_date:
                        try:
                            dates.append(datetime.strptime(pub_date, '%Y-%m-%d').date())
                        except:
                            pass
                
                min_date = min(dates) if dates else datetime.now().date()
                max_date = max(dates) if dates else datetime.now().date()
                
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col2:
                    _, doc_info = list_documents_supabase()

                    categories = set([info.get('category_name', 'Other') for info in doc_info.values()])
                    categories_list = ['All Categories'] + sorted(list(categories))
                    selected_category = st.selectbox("Select category", categories_list)
                
                with col3:
                    date_range = st.date_input(
                        "Publication date range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                
                filtered_results = results
                
                if selected_category != 'All Categories':
                    filtered_results = [r for r in filtered_results if r['metadata'].get('category_name') == selected_category]
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_results = [
                        r for r in filtered_results 
                        if r['metadata'].get('published', '') and 
                        start_date <= datetime.strptime(r['metadata']['published'], '%Y-%m-%d').date() <= end_date
                    ]
                
                with col1:
                    max_results = len(filtered_results)
                    if max_results > 1:
                        top_k = st.slider(
                            f"Show top (1 to {max_results})", 
                            min_value=1, 
                            max_value=max_results, 
                            value=min(5, max_results),
                            key="top_k_slider"
                        )
                    else:
                        top_k = 1

                
                display_results = filtered_results[:top_k]
                
                if selected_category != 'All Categories':
                    st.info(f"Showing top {len(display_results)} of {len(filtered_results)} documents in category '{selected_category}'")
                else:
                    st.info(f"Showing top {len(display_results)} of {len(filtered_results)} similar documents")
                
                st.divider()
                
                if len(display_results) > 0:
                    for i, result in enumerate(display_results):
                        similarity_pct = result['similarity'] * 100
                        metadata = result['metadata']
                        doc_id = result['doc_id']
                        
                        keywords = []
                        abstract = metadata.get("abstract", "")
                        summary = generate_summary_from_abstract(abstract)

                        
                        card_html = f"""
                        <div style='background-color: #33363D; padding: 10px; border-radius: 10px; margin-bottom: 15px; position: relative;'>
                            <div style='position: absolute; top: 15px; right: 15px; background-color: #28a745; color: white; padding: 8px 15px; border-radius: 8px; font-weight: bold; font-size: 16px;'>
                                {similarity_pct:.1f}%
                            </div>
                            <h4 style='margin-top: 0; padding-right: 100px;'>#{i+1} - {metadata.get('name', 'Unknown')[:120]}</h4>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        with st.container():
                            detail_col1, detail_col2 = st.columns(2)
                            
                            with detail_col1:
                                st.write("**Authors:**", metadata.get('authors', 'N/A'))
                                st.write("**Category:**", metadata.get('category_name', 'N/A'))
                            
                            with detail_col2:
                                st.write("**Published:**", metadata.get('published', 'N/A'))
                                st.write("**Type:**", metadata.get('type', 'N/A'))
                            
                            if keywords:
                                keyword_html = "**Top Keywords:** " + " ".join([
                                    f"<span style='background-color: #FF0000; padding: 3px 8px; border-radius: 5px; margin-right: 5px;'>{kw}</span>"
                                    for kw in keywords
                                ])
                                st.markdown(keyword_html, unsafe_allow_html=True)
                            
                            st.write("")
                            st.write(f"**Summary:** {summary}")
                            
                            btn_col1, btn_col2, btn_col3 = st.columns(3)
                            
                            with btn_col1:
                                if metadata.get('type') == 'arxiv' and metadata.get('source'):
                                    arxiv_id = metadata.get('source', '').replace('arXiv:', '')
                                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                                    
                                    try:
                                        pdf_bytes = download_arxiv_pdf(pdf_url)
                                        if pdf_bytes:
                                            st.download_button(
                                                label="📥 Download PDF",
                                                data=pdf_bytes,
                                                file_name=f"{arxiv_id.replace('/', '_')}.pdf",
                                                mime="application/pdf",
                                                key=f"download_{doc_id}_{i}",
                                                use_container_width=True
                                            )
                                    except:
                                        pass
                                    
                            st.divider()
                            
                            with btn_col2:
                                if metadata.get('type') == 'arxiv' and metadata.get('source'):
                                    arxiv_id = metadata.get('source', '').replace('arXiv:', '')
                                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                                    st.link_button("📄 PDF Preview", pdf_url, use_container_width=True)
                            
                            with btn_col3:
                                if metadata.get('type') == 'arxiv' and metadata.get('source'):
                                    arxiv_id = metadata.get('source', '').replace('arXiv:', '')
                                    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                                    st.link_button("🔗 View on arXiv", arxiv_url, use_container_width=True)
                        
                        st.write("")
                else:
                    st.warning("No documents match the selected filters")
    
    # Manage Repository Page
    elif page == "Manage Repository":
        st.header("🗂️ Manage Document Repository")
        
        doc_ids, doc_info = list_documents_supabase()
        
        if doc_ids:
            st.subheader(f"Repository contains {len(doc_ids)} documents")
            
            search_query = st.text_input("🔍 Search documents by title, author, or source:", 
                                        placeholder="e.g., neural networks, Smith, arXiv:2301.12345")
            
            col1, col2 = st.columns(2)

            with col1:
                filter_type = st.selectbox("Filter by type", ["All", "arXiv papers", "Uploaded documents"])         
            
            with col2:
                all_categories = set([info.get('category_name', 'Other') for info in doc_info.values()])
                all_categories = sorted(list(all_categories))
                filter_category = st.selectbox("Filter by domain", ["All Domains"] + all_categories)
            
            if st.button("Search Repository", type="primary"):
                st.session_state.show_manage_results = True
            
            if 'show_manage_results' not in st.session_state:
                st.session_state.show_manage_results = False
            
            if st.session_state.show_manage_results:
                filtered_docs = doc_ids
                
                if filter_type == "arXiv papers":
                    filtered_docs = [d for d in filtered_docs if doc_info.get(d, {}).get('type') == 'arxiv']
                elif filter_type == "Uploaded documents":
                    filtered_docs = [d for d in filtered_docs if doc_info.get(d, {}).get('type') == 'uploaded']
                
                if filter_category != "All Domains":
                    filtered_docs = [d for d in filtered_docs if doc_info.get(d, {}).get('category_name') == filter_category]
                
                if search_query:
                    search_lower = search_query.lower()
                    filtered_docs = [
                        d for d in filtered_docs 
                        if search_lower in doc_info.get(d, {}).get('name', '').lower() 
                        or search_lower in doc_info.get(d, {}).get('source', '').lower()
                        or search_lower in doc_info.get(d, {}).get('authors', '').lower()
                    ]
                
                st.write(f":green[Showing {len(filtered_docs)} documents]")
                
                # Bulk actions
                if len(filtered_docs) > 0:
                    st.divider()
                    col_bulk1, col_bulk2 = st.columns([3, 1])
                    with col_bulk1:
                        st.write("**Bulk Actions:**")
                    with col_bulk2:
                        zip_buffer = io.BytesIO()

                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for doc_id in filtered_docs:
                                info = doc_info.get(doc_id, {})
                                
                                if info.get('type') == 'arxiv' and info.get('source'):
                                    arxiv_id = info.get('source', '').replace('arXiv:', '')
                                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                                    try:
                                        pdf_bytes = download_arxiv_pdf(pdf_url)
                                        if pdf_bytes:
                                            zip_file.writestr(
                                                f"{arxiv_id.replace('/', '_')}.pdf",
                                                pdf_bytes
                                            )
                                    except:
                                        pass
                                
                                elif info.get('type') == 'uploaded':
                                    pdf_bytes = get_uploaded_pdf_supabase(doc_id)
                                    if pdf_bytes:
                                        filename = info.get('filename', f"{doc_id}.pdf")
                                        zip_file.writestr(filename, pdf_bytes)

                        zip_buffer.seek(0)

                        st.download_button(
                            label="📥 Download Zip",
                            data=zip_buffer,
                            file_name=f"papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            key="bulk_download"
                        )

                
                st.divider()
                
                
                # Display documents
                display_limit = 50
                for doc_id in filtered_docs[:display_limit]:
                    info = doc_info.get(doc_id, {})
                    
                    with st.expander(f"📄 {info.get('name', doc_id)[:100]}..."):
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.write("**Document ID:**", doc_id)
                            st.write("**Source:**", info.get('source', 'N/A'))
                            st.write("**Category:**", info.get('category_name', 'N/A'))
                        
                        with info_col2:
                            st.write("**Type:**", info.get('type', 'N/A'))
                            if info.get('authors'):
                                st.write("**Authors:**", info.get('authors', 'N/A'))
                            if info.get('published'):
                                st.write("**Published:**", info.get('published', 'N/A'))
                        
                        st.divider()
                        action_col1, action_col2, action_col3 = st.columns(3)
                        
                        with action_col1:
                            if info.get('type') == 'arxiv' and info.get('source'):
                                arxiv_id = info.get('source', '').replace('arXiv:', '')
                                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                                
                                try:
                                    pdf_bytes = download_arxiv_pdf(pdf_url)
                                    if pdf_bytes:
                                        st.download_button(
                                            label="📥 Download PDF",
                                            data=pdf_bytes,
                                            file_name=f"{arxiv_id.replace('/', '_')}.pdf",
                                            mime="application/pdf",
                                            key=f"download_{doc_id}"
                                        )
                                except:
                                    st.button("📥 Download PDF (Error)", disabled=True, key=f"download_{doc_id}")
                            
                            elif info.get('type') == 'uploaded':
                                pdf_bytes = get_uploaded_pdf_supabase(doc_id)
                                if pdf_bytes:
                                    st.download_button(
                                        label="📥 Download PDF",
                                        data=pdf_bytes,
                                        file_name=info.get('filename', f"{doc_id}.pdf"),
                                        mime="application/pdf",
                                        key=f"download_{doc_id}"
                                    )
                                else:
                                    st.info("PDF not available")
                            else:
                                st.info("Download not available")
                        
                        with action_col2:
                            if info.get('type') == 'arxiv' and info.get('source'):
                                arxiv_id = info.get('source', '').replace('arXiv:', '')
                                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                                st.link_button("🔗 View on arXiv", arxiv_url)
                            else:
                                st.write("")
                        
                        with action_col3:
                            if st.button("🗑️ Delete", key=f"del_{doc_id}", type="secondary"):
                                success, message = delete_document_supabase(doc_id)
                                if success:
                                    st.success(message)
                                    time.sleep(1)
                                    st.session_state.show_manage_results = False
                                    st.rerun()
                                else:
                                    st.error(message)
                
                if len(filtered_docs) > display_limit:
                    st.info(f"Showing first {display_limit} of {len(filtered_docs)} documents. Use filters or search to narrow results.")
                    
            #Clear Repository
            st.divider()
            st.subheader("⚠️ Danger Zone")
            
            confirm_clear = st.checkbox("I understand this will delete all documents permanently")
            
            if st.button("🗑️ Clear Entire Repository", 
                        type="secondary",
                        disabled=not confirm_clear):
                if confirm_clear:
                    try:
                        supabase = init_supabase()
                        if not supabase:
                            st.error("Supabase not initialized")
                        else:
                            supabase.table("document_chunks").delete().neq("doc_id", "").execute()
                            st.success("✅ Repository cleared successfully!")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing repository: {str(e)}")

        
        else:
            st.info("📭 No documents in repository. Please initialize the corpus from the 'Corpus Status' page to get started.")
            if st.button("Go to Corpus Status", type="primary"):
                st.session_state.page = "Corpus Status"
                st.session_state.show_manage_results = False
                st.rerun()  
                 
if __name__ == "__main__":
    main()