#!/usr/bin/env python3

import sys
from pathlib import Path
from rag.indexing import MultiModalIndex
from rag.models import ClipMultiModal
from rag.retrieval import Retriever
import numpy as np

def test_search_accuracy(query, expected_type="any"):
    """Test search accuracy for a given query"""
    print(f"\n=== TESTING QUERY: '{query}' ===")
    print(f"Expected type: {expected_type}")
    
    try:
        # Initialize components
        clip = ClipMultiModal()
        INDEX_DIR = Path("index")
        
        if INDEX_DIR.exists():
            index = MultiModalIndex(persist_dir=INDEX_DIR, persistent=True)
        else:
            print("No index found!")
            return
        
        retriever = Retriever(index=index, clip=clip)
        
        # Test different k values to see ranking
        for k in [3, 5, 10]:
            print(f"\n--- Top {k} Results ---")
            results = retriever.search_text(query, k=k)
            
            if not results:
                print(f"No results found for k={k}")
                continue
                
            for i, result in enumerate(results):
                score = result.score
                modality = result.metadata.get('modality', 'unknown')
                result_type = result.metadata.get('type', 'unknown')
                source = result.metadata.get('source_name', 'unknown')
                
                # Show content preview
                if modality == 'image' or result_type in ['image', 'image_text']:
                    ocr_text = result.metadata.get('ocr_text', result.metadata.get('text', ''))[:100]
                    content = f"OCR: {ocr_text}..." if ocr_text else "No OCR text"
                else:
                    text = result.metadata.get('text', '')[:100]
                    content = f"Text: {text}..." if text else "No text"
                
                # Relevance indicator
                is_relevant = "✅" if (
                    (expected_type == "image" and (modality == 'image' or result_type in ['image', 'image_text'])) or
                    (expected_type == "text" and modality == 'text' and result_type == 'text') or
                    expected_type == "any"
                ) else "❌"
                
                print(f"{i+1}. {is_relevant} {source} (Score: {score:.4f})")
                print(f"   Type: {modality}/{result_type}")
                print(f"   Content: {content}")
                print()
                
    except Exception as e:
        print(f"Error testing query: {e}")
        import traceback
        traceback.print_exc()

def analyze_embedding_quality():
    """Analyze the quality of embeddings for different content types"""
    print("\n=== EMBEDDING QUALITY ANALYSIS ===")
    
    try:
        clip = ClipMultiModal()
        
        # Test queries
        test_queries = [
            "email screenshot",
            "binary search code", 
            "algorithm implementation",
            "user interface",
            "programming code"
        ]
        
        print("Testing embedding similarity between queries...")
        embeddings = []
        
        for query in test_queries:
            emb = clip.embed_text([query])[0]
            embeddings.append(emb)
            print(f"✅ Embedded: '{query}'")
        
        # Calculate similarity between queries
        print(f"\nQuery Similarity Matrix:")
        print(f"{'Query':<20} ", end="")
        for i, q in enumerate(test_queries):
            print(f"{i+1:<8}", end="")
        print()
        
        for i, (q1, emb1) in enumerate(zip(test_queries, embeddings)):
            print(f"{q1:<20} ", end="")
            for j, emb2 in enumerate(embeddings):
                # Cosine similarity
                similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                print(f"{similarity:.3f}   ", end="")
            print()
            
    except Exception as e:
        print(f"Error in embedding analysis: {e}")

def suggest_improvements():
    """Suggest specific improvements based on current setup"""
    print(f"\n=== ACCURACY IMPROVEMENT SUGGESTIONS ===")
    
    suggestions = [
        {
            "issue": "Poor OCR Quality",
            "solutions": [
                "Try different OCR models (Tesseract vs EasyOCR)",
                "Preprocess images (contrast, denoising)",
                "Use multiple OCR engines and combine results",
                "Manual text correction for key documents"
            ]
        },
        {
            "issue": "Embedding Mismatch", 
            "solutions": [
                "Use domain-specific embedding models",
                "Fine-tune CLIP on your specific content",
                "Add keyword-based fallback search",
                "Use multiple embedding strategies"
            ]
        },
        {
            "issue": "Over-aggressive Image Boosting",
            "solutions": [
                "Reduce boost multiplier (1.5x → 1.2x)",
                "Use conditional boosting based on query type",
                "Implement relevance thresholding",
                "Add semantic similarity checks"
            ]
        },
        {
            "issue": "Poor Content Enhancement",
            "solutions": [
                "Improve OCR text preprocessing", 
                "Add more descriptive context",
                "Use better keyword detection",
                "Include image descriptions from vision models"
            ]
        }
    ]
    
    for suggestion in suggestions:
        print(f"\n🔧 {suggestion['issue']}:")
        for i, solution in enumerate(suggestion['solutions'], 1):
            print(f"   {i}. {solution}")

if __name__ == "__main__":
    print("🎯 MULTIMODAL RAG ACCURACY ANALYSIS")
    print("=" * 50)
    
    # Test specific queries
    test_search_accuracy("email screenshot", "image")
    test_search_accuracy("binary search code", "image") 
    test_search_accuracy("algorithm", "any")
    test_search_accuracy("programming", "any")
    
    # Analyze embeddings
    analyze_embedding_quality()
    
    # Show suggestions
    suggest_improvements()
    
    print(f"\n✅ Analysis complete! Review results above for accuracy insights.")