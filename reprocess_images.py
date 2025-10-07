#!/usr/bin/env python3

import sys
from pathlib import Path
from rag.indexing import MultiModalIndex
from rag.models import ClipMultiModal
from rag.ingestion import ingest_image
import shutil

def reprocess_existing_images():
    """Reprocess existing images with improved OCR"""
    print("🔄 REPROCESSING IMAGES WITH IMPROVED OCR")
    print("=" * 50)
    
    try:
        # Initialize components
        clip = ClipMultiModal()
        INDEX_DIR = Path("index")
        
        if not INDEX_DIR.exists():
            print("No index found!")
            return
            
        index = MultiModalIndex(persist_dir=INDEX_DIR, persistent=True)
        
        # Check uploads directory
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            print("No uploads directory found!")
            return
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(uploads_dir.rglob(f'*{ext}'))
        
        print(f"Found {len(image_files)} image files to reprocess")
        
        if not image_files:
            print("No images found to reprocess!")
            return
        
        # Backup current index
        backup_dir = Path("index_backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(INDEX_DIR, backup_dir)
        print(f"✅ Backed up current index to {backup_dir}")
        
        # Clear current index
        print("🗑️ Clearing current index...")
        index.reset()
        
        # Reprocess all images
        all_records = []
        for i, img_path in enumerate(image_files):
            print(f"📷 Processing {i+1}/{len(image_files)}: {img_path.name}")
            
            try:
                # Create source ID
                sid = f"{img_path.name}:{img_path.stat().st_size}:{int(img_path.stat().st_mtime)}"
                
                # Ingest with improved OCR
                records = ingest_image(img_path, clip, source_id=sid)
                all_records.extend(records)
                
                print(f"   ✅ Created {len(records)} records")
                
                # Show what OCR text was extracted
                for record in records:
                    if record.metadata.get('ocr_text'):
                        ocr_preview = record.metadata['ocr_text'][:100] + "..." if len(record.metadata['ocr_text']) > 100 else record.metadata['ocr_text']
                        print(f"   📝 OCR: {ocr_preview}")
                        
            except Exception as e:
                print(f"   ❌ Failed to process {img_path.name}: {e}")
        
        # Add all records back to index
        if all_records:
            print(f"\n📊 Adding {len(all_records)} total records to index...")
            index.add_records(all_records)
            print(f"✅ Reprocessing complete! Index now has {index.count()} items")
        else:
            print("❌ No records created - restoring backup")
            shutil.rmtree(INDEX_DIR)
            shutil.move(backup_dir, INDEX_DIR)
        
        print(f"\n🎯 Reprocessing complete! Test your searches now.")
        
    except Exception as e:
        print(f"❌ Error during reprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("This will reprocess all images with improved OCR and rebuild the index.")
    response = input("Do you want to continue? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        reprocess_existing_images()
    else:
        print("Operation cancelled.")