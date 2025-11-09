# ingest.py
import os
import re
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
DATA_DIR = "data"
DB_DIR = "db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# This is a good default chunk size.
# The 'source' tags in your docs help, but we still need to split.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- Utility Function to Extract Metadata from Filename ---
def extract_metadata_from_filename(filename):
    """
    Extracts policy number and title from filenames like:
    'POL-COR-008_Code_of_Conduct_Ethics.txt'
    """
    # Remove the .txt extension
    base_name = os.path.splitext(filename)[0]
    
    parts = base_name.split('_', 1)
    if len(parts) == 2:
        policy_number = parts[0]
        # Replace underscores with spaces for a clean title
        policy_title = parts[1].replace('_', ' ')
        return policy_number, policy_title
    else:
        # Fallback for unexpected filenames
        return base_name, base_name

# --- Main Ingestion Process ---
def main():
    print("Starting ingestion process...")
    
    all_docs = []
    
    # 1. Load Documents and Add Metadata
    print(f"Loading documents from '{DATA_DIR}'...")
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                # Load the document
                loader = TextLoader(filepath, encoding='utf-8')
                doc = loader.load()[0] # TextLoader loads one doc per file
                
                # Extract metadata from the filename
                policy_number, policy_title = extract_metadata_from_filename(filename)
                
                # Add all metadata to the document
                # This metadata will be inherited by all chunks
                doc.metadata = {
                    "policy_number": policy_number,
                    "policy_title": policy_title,
                    "source_file": filename
                }
                
                all_docs.append(doc)
                print(f"  - Loaded: {filename}")
            except Exception as e:
                print(f"  - ERROR loading {filename}: {e}")
                
    if not all_docs:
        print("No documents found. Aborting.")
        return

    # 2. Split Documents into Chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"Total documents split into {len(splits)} chunks.")

    # 3. Create Embedding Function
    # This will use the 'all-MiniLM-L6-v2' model
    # The first time you run this, it will download the model (approx. 90MB)
    print("\nLoading embedding model (this may take a moment)...")
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL,
        # Use 'mps' for Apple Silicon (M-series chips) for hardware acceleration
        model_kwargs={'device': 'mps'} 
    )

    # 4. Create and Persist Vector Store
    # This will create the 'db' folder and store the vectors
    print("\nCreating and persisting vector store...")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print("\n--- Ingestion Complete! ---")
    print(f"Vector store saved to '{DB_DIR}' directory.")


if __name__ == "__main__":
    main()