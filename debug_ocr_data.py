#!/usr/bin/env python3

import sys
from pathlib import Path
from rag.indexing import MultiModalIndex
from rag.models import ClipMultiModal
from rag.retrieval import Retriever

def debug_index_data():
    print("🔍 DEBUGGING INDEX DATA")
    print("=" * 50)
    
    try:
        clip = ClipMultiModal()
        index_dir = Path("index")
        
        if not index_dir.exists():
            print("❌ No index directory found!")
            return
        
        index = MultiModalIndex(persist_dir=index_dir, persistent=True)
        retriever = Retriever(index=index, clip=clip)
        
        # Check if we can access documents
        print("🗂️ Index Overview:")
        
        # Check different ways to access the index
        print(f"Index directory: {index_dir}")
        print(f"Index exists: {index_dir.exists()}")
        
        # List files in index directory
        if index_dir.exists():
            index_files = list(index_dir.glob("*"))
            print(f"Index files: {[f.name for f in index_files]}")
        
        # Test a simple search to get some results
        print("\n📋 Sample Search Results:")
        test_queries = ["email", "code", "algorithm", "screenshot"]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.search_text(query, k=3)
            
            if results:
                for i, result in enumerate(results):
                    print(f"  Result {i+1}:")
                    print(f"    Score: {result.score:.4f}")
                    print(f"    Metadata keys: {list(result.metadata.keys())}")
                    
                    # Check OCR text
                    ocr_text = result.metadata.get('ocr_text', '')
                    if ocr_text:
                        print(f"    OCR text (first 100 chars): {ocr_text[:100]}...")
                    else:
                        print(f"    OCR text: EMPTY or MISSING")
                    
                    # Check source info
                    source_name = result.metadata.get('source_name', 'Unknown')
                    doc_type = result.metadata.get('type', 'Unknown')
                    print(f"    Source: {source_name}")
                    print(f"    Type: {doc_type}")
                    
                    print(f"    All metadata: {result.metadata}")
                    print()
            else:
                print("  No results found")
        
        # Try to check if the collection has any documents
        try:
            # Try different approaches to get collection info
            print("\n🗄️ Collection Information:")
            collection = index.collection
            if hasattr(collection, 'count'):
                doc_count = collection.count()
                print(f"Document count: {doc_count}")
            
            if hasattr(collection, 'get'):
                # Try to get some documents
                all_docs = collection.get()
                if all_docs and 'metadatas' in all_docs:
                    print(f"First few document metadata:")
                    for i, metadata in enumerate(all_docs['metadatas'][:3]):
                        print(f"  Doc {i}: {metadata}")
                        
        except Exception as e:
            print(f"Could not get collection info: {e}")
        
    except Exception as e:
        print(f"❌ Error debugging index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_index_data()