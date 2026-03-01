import logging
import time
from typing import List, Dict
from Bio import Entrez
from datetime import datetime
import os
import sys
import ssl

# Bypass SSL verification for legacy environments
ssl._create_default_https_context = ssl._create_unverified_context

# Add parent directory to path to import backend modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Beta_5.5 root
chatbot_root = os.path.join(project_root, 'health_advisory_chatbot')
backend_root = os.path.join(chatbot_root, 'backend')

# Add both to path to handle different import styles
sys.path.insert(0, chatbot_root)
sys.path.insert(0, backend_root)

from backend.chatbot.rag.vector_store import get_vector_store, COLLECTION_CLINICAL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Entrez Config (Required by NCBI)
Entrez.email = "health_chatbot_dev@example.com"  # Replace with valid email in prod
Entrez.tool = "ElderlyCareAdvisory"

# Priority Search Queries
SEARCH_QUERIES = {
    "dementia_management": "dementia[Title/Abstract] AND (management[Title/Abstract] OR care[Title/Abstract]) AND elderly[Title/Abstract] AND \"last 5 years\"[dp]",
    "alzheimers_prevention": "alzheimer's[Title/Abstract] AND prevention[Title/Abstract] AND lifestyle[Title/Abstract] AND \"last 5 years\"[dp]",
    "diabetes_elderly": "type 2 diabetes[Title/Abstract] AND elderly[Title/Abstract] AND management[Title/Abstract] AND \"last 5 years\"[dp]",
    "sleep_apnea_elderly": "sleep apnea[Title/Abstract] AND elderly[Title/Abstract] AND risks[Title/Abstract] AND \"last 5 years\"[dp]",
    "insomnia_treatment": "insomnia[Title/Abstract] AND elderly[Title/Abstract] AND treatment[Title/Abstract] AND \"last 5 years\"[dp]",
    "fall_prevention": "accidental falls[Title/Abstract] AND prevention[Title/Abstract] AND elderly[Title/Abstract] AND \"last 5 years\"[dp]"
}

def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    """Search PubMed for PMIDs."""
    logger.info(f"Searching PubMed for: {query}")
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

def fetch_details(id_list: List[str]) -> List[Dict]:
    """Fetch paper details for PMIDs."""
    if not id_list:
        return []
        
    ids = ",".join(id_list)
    logger.info(f"Fetching details for {len(id_list)} papers...")
    
    try:
        handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        papers = []
        for article in records['PubmedArticle']:
            try:
                medline = article['MedlineCitation']['Article']
                title = medline.get('ArticleTitle', 'No Title')
                abstract = ""
                if 'Abstract' in medline and 'AbstractText' in medline['Abstract']:
                    # AbstractText can be a list or finding
                    abs_text = medline['Abstract']['AbstractText']
                    if isinstance(abs_text, list):
                        abstract = " ".join(abs_text)
                    else:
                        abstract = str(abs_text)
                        
                if not abstract:
                    continue
                    
                journal = medline.get('Journal', {}).get('Title', 'Unknown Journal')
                year = medline.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', 'N/A')
                
                papers.append({
                    "pmid": str(article['MedlineCitation']['PMID']),
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "source": "PubMed",
                    "full_text": f"Title: {title}\nAbstract: {abstract}\nJournal: {journal} ({year})"
                })
            except Exception as e:
                continue
                
        return papers
    except Exception as e:
        logger.error(f"Fetch details failed: {e}")
        return []

def ingest_data():
    """Main ingestion flow."""
    vector_store = get_vector_store()
    total_indexed = 0
    
    for topic, query in SEARCH_QUERIES.items():
        logger.info(f"Processing topic: {topic}")
        
        # 1. Search
        pmids = search_pubmed(query, max_results=30)
        time.sleep(1) # Rate limit courtesy
        
        # 2. Fetch
        papers = fetch_details(pmids)
        
        if not papers:
            continue
            
        # 3. Index
        documents = [p['full_text'] for p in papers]
        metadatas = [{
            "source": "PubMed",
            "pmid": p['pmid'],
            "title": p['title'],
            "year": p['year'],
            "category": topic
        } for p in papers]
        ids = [f"pubmed_{p['pmid']}" for p in papers]
        
        try:
            vector_store.add_documents(
                collection_name=COLLECTION_CLINICAL,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            total_indexed += len(documents)
            logger.info(f"Indexed {len(documents)} papers for {topic}")
        except Exception as e:
            logger.error(f"Indexing failed for {topic}: {e}")
            
    logger.info(f"Ingestion complete. Total documents indexed: {total_indexed}")

if __name__ == "__main__":
    ingest_data()
