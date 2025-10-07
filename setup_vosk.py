#!/usr/bin/env python3
"""
Setup script to download and configure Vosk model for offline speech recognition.
"""

import os
import sys
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Model configuration
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / MODEL_NAME

def download_file(url: str, filepath: Path) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def extract_model(zip_path: Path, extract_dir: Path) -> bool:
    """Extract the downloaded model"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🎙️ Setting up Vosk offline speech recognition...")
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"✅ Model {MODEL_NAME} already exists at {MODEL_PATH}")
        return True
    
    print(f"📥 Downloading {MODEL_NAME} model...")
    print(f"📍 URL: {MODEL_URL}")
    print(f"📁 Target: {MODEL_PATH}")
    
    # Download model
    zip_path = MODELS_DIR / f"{MODEL_NAME}.zip"
    if not download_file(MODEL_URL, zip_path):
        print("❌ Download failed!")
        return False
    
    print(f"📦 Extracting model to {MODELS_DIR}...")
    if not extract_model(zip_path, MODELS_DIR):
        print("❌ Extraction failed!")
        return False
    
    # Clean up zip file
    try:
        zip_path.unlink()
        print("🧹 Cleaned up zip file")
    except:
        pass
    
    # Verify model structure
    if MODEL_PATH.exists():
        print(f"✅ Model successfully set up at: {MODEL_PATH}")
        
        # List model contents
        print("\n📋 Model contents:")
        for item in MODEL_PATH.iterdir():
            print(f"   - {item.name}")
        
        print(f"\n🎯 Model size: ~40MB")
        print(f"🎯 Accuracy: ~90% (LibriSpeech test-clean)")
        print(f"🎯 Language: English (US)")
        print(f"🎯 Usage: Lightweight model for real-time recognition")
        return True
    else:
        print("❌ Model setup verification failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    print("\n🚀 Ready to use offline speech recognition!")