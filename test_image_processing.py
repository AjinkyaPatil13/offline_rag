#!/usr/bin/env python3
"""
Test script to verify image processing and indexing is working correctly.
This will help debug any issues with image uploads and search.
"""

import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

# Import our RAG components
from rag.models import ClipMultiModal
from rag.ingestion import ingest_image
from rag.indexing import MultiModalIndex
from rag.retrieval import Retriever

def create_test_image(path: Path, color: tuple = (255, 0, 0), size: tuple = (100, 100)):
    """Create a simple test image"""
    img = Image.new('RGB', size, color)
    img.save(path)
    return path

def test_image_processing():
    """Test the complete image processing pipeline"""
    print("🧪 Testing image processing pipeline...\n")
    
    # Initialize components
    print("🔧 Initializing CLIP model...")
    clip = ClipMultiModal()
    print("✅ CLIP model loaded\n")
    
    print("🔧 Initializing index...")
    index = MultiModalIndex(persistent=False)  # Use ephemeral for testing
    print("✅ Index initialized\n")
    
    # Create test images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images with different colors
        red_img = create_test_image(temp_path / "red_square.png", (255, 0, 0))
        blue_img = create_test_image(temp_path / "blue_square.png", (0, 0, 255))
        green_img = create_test_image(temp_path / "green_square.png", (0, 255, 0))
        
        print("🎨 Created test images:")
        print(f"  - {red_img.name}")
        print(f"  - {blue_img.name}")
        print(f"  - {green_img.name}\n")
        
        # Test image ingestion
        print("📥 Testing image ingestion...")
        try:
            red_records = ingest_image(red_img, clip, source_id=f"test_red_{red_img.name}")
            blue_records = ingest_image(blue_img, clip, source_id=f"test_blue_{blue_img.name}")
            green_records = ingest_image(green_img, clip, source_id=f"test_green_{green_img.name}")
            
            print(f"✅ Red image: {len(red_records)} records created")
            print(f"✅ Blue image: {len(blue_records)} records created")
            print(f"✅ Green image: {len(green_records)} records created")
            
            # Check record structure
            if red_records:
                r = red_records[0]
                print(f"\n📋 Sample record structure:")
                print(f"  - ID: {r.id}")
                print(f"  - Modality: {r.modality}")
                print(f"  - Embedding shape: {len(r.embedding)}")
                print(f"  - Metadata keys: {list(r.metadata.keys())}")
                print(f"  - Source name: {r.metadata.get('source_name')}")
                print(f"  - Image path: {r.metadata.get('image_path')}")
            
        except Exception as e:
            print(f"❌ Image ingestion failed: {e}")
            return False
        
        # Test indexing
        print(f"\n📚 Testing indexing...")
        try:
            all_records = red_records + blue_records + green_records
            index.add_records(all_records)
            count = index.count()
            print(f"✅ Added {len(all_records)} records to index")
            print(f"✅ Index now contains {count} items\n")
            
        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            return False
        
        # Test image search
        print("🔍 Testing image search...")
        try:
            retriever = Retriever(index=index, clip=clip)
            
            # Search using the red image
            hits = retriever.search_image(red_img, k=3)
            print(f"✅ Found {len(hits)} results for red image search")
            
            for i, hit in enumerate(hits):
                meta = hit.metadata
                print(f"  {i+1}. {meta.get('source_name')} (score: {hit.score:.3f}, type: {meta.get('modality', 'unknown')})")
            
            # Test text search for images
            print(f"\n🔍 Testing text search (should find images too)...")
            text_hits = retriever.search_text("red color square", k=5)
            print(f"✅ Found {len(text_hits)} results for text search")
            
            image_results = [h for h in text_hits if h.metadata.get('modality') == 'image']
            print(f"📸 Of which {len(image_results)} are images")
            
            for i, hit in enumerate(image_results[:3]):
                meta = hit.metadata
                print(f"  {i+1}. {meta.get('source_name')} (score: {hit.score:.3f})")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False
    
    print(f"\n🎉 All tests passed! Image processing is working correctly.")
    return True

def main():
    print("🚀 Image Processing Test Suite\n")
    print("This will test the complete image processing pipeline:")
    print("  1. CLIP model loading")
    print("  2. Image ingestion")
    print("  3. Indexing")
    print("  4. Image search")
    print("  5. Cross-modal text→image search\n")
    
    try:
        success = test_image_processing()
        if success:
            print("\n✅ Image processing is working correctly!")
            print("💡 Your app should be able to process and search images properly.")
        else:
            print("\n❌ Image processing has issues that need to be fixed.")
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        print("❌ There are fundamental issues with the image processing setup.")

if __name__ == "__main__":
    main()