from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from tqdm import tqdm
import os
import shutil
import pickle

def load_cluster_data():
    """Load all cluster data files from the cluster_data directory."""
    print("Loading cluster data files...")
    
    combined_text = ""
    cluster_data_dir = "cluster_data"
    
    # Get list of files and create progress bar
    files = [f for f in os.listdir(cluster_data_dir) if f.startswith("cubes_") and f.endswith(".txt")]
    for filename in tqdm(files, desc="Loading cluster files"):
        filepath = os.path.join(cluster_data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            combined_text += f.read() + "\n\n"
    
    return combined_text

def create_vector_store(text_content):
    """Create Chroma vector store from the text content."""
    print("Creating vector store...")
    
    persist_directory = "chroma_db"
    embeddings_directory = "embeddings_cache"
    
    # Create embeddings cache directory
    os.makedirs(embeddings_directory, exist_ok=True)
    
    # Delete existing vector store to force recreation
    if os.path.exists(persist_directory):
        print("Removing old vector store...")
        shutil.rmtree(persist_directory)
    
    print("Creating new vector store...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_text(text_content)
    print(f"Split into {len(texts)} chunks")
    
    # Create embeddings using Ollama
    try:
        print("Testing Ollama connection...")
        embeddings = OllamaEmbeddings(model="llama2")
        test_embedding = embeddings.embed_query("test")
        print("Ollama connection successful")
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        raise
    
    try:
        # Generate and save embeddings in batches
        print("Generating embeddings...")
        batch_size = 1000
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_num in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Generate embeddings for this batch
            batch_embeddings = []
            for text in tqdm(batch_texts, desc=f"Batch {batch_num+1}/{total_batches}", leave=False):
                embedding = embeddings.embed_query(text)
                batch_embeddings.append(embedding)
            
            # Save this batch
            batch_data = {
                'texts': batch_texts,
                'embeddings': batch_embeddings
            }
            batch_file = os.path.join(embeddings_directory, f"batch_{batch_num}.pkl")
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f)
            
            print(f"\nSaved batch {batch_num+1}/{total_batches}")
        
        print("\nCreating Chroma vector store from saved embeddings...")
        vector_store = None
        
        # Load batches and create vector store
        for batch_num in tqdm(range(total_batches), desc="Creating vector store"):
            batch_file = os.path.join(embeddings_directory, f"batch_{batch_num}.pkl")
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            if vector_store is None:
                vector_store = Chroma.from_texts(
                    texts=batch_data['texts'],
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name="token_clusters"
                )
            else:
                vector_store.add_texts(batch_data['texts'])
            
            vector_store.persist()
            print(f"\nProcessed batch {batch_num+1}/{total_batches}")
        
        print("Vector store creation complete")
        return vector_store
        
    except Exception as e:
        print(f"Error in vector store creation: {str(e)}")
        import traceback
        print("Full error:")
        print(traceback.format_exc())
        raise

def setup_qa_chain(vector_store):
    """Set up the QA chain with the vector store."""
    print("Setting up QA chain...")
    
    # Initialize Ollama LLM
    llm = Ollama(model="llama2")
    
    # Create and return the QA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

def main():
    try:
        # Load cluster data
        content = load_cluster_data()
        
        # Create or load vector store
        vector_store = create_vector_store(content)
        
        # Setup QA chain
        qa_chain = setup_qa_chain(vector_store)
        
        print("\nToken Cluster Analysis QA System Ready!")
        print("Enter your questions about the token clusters (or 'quit' to exit)")
        print("Example questions:")
        print("- What tokens are in cluster cube0_cluster0?")
        print("- What patterns do you see in the connected nodes?")
        print("- Describe the tokens in hypercube 5")
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() == 'quit':
                break
            
            if question:
                try:
                    result = qa_chain({"query": question})
                    print("\nAnswer:", result["result"])
                    print("\nSources used:")
                    for i, doc in enumerate(result["source_documents"], 1):
                        print(f"\nSource {i}:")
                        print("-" * 40)
                        print(doc.page_content[:300] + "...")
                except Exception as e:
                    print(f"\nError processing question: {str(e)}")
                    print("Please try rephrasing your question.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure Ollama is running and the llama2 model is installed.")

if __name__ == "__main__":
    main() 