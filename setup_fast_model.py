#!/usr/bin/env python3
"""
Setup script to download the fastest Ollama model for optimal RAG performance.
This will try to install the smallest, fastest models available.
"""

import subprocess
import sys
import time

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_ollama():
    """Check if Ollama is installed and running"""
    print("🔍 Checking Ollama installation...")
    success, stdout, stderr = run_command("ollama --version")
    if not success:
        print("❌ Ollama not found. Please install Ollama first:")
        print("   Visit: https://ollama.ai/download")
        return False
    
    print(f"✅ Ollama found: {stdout.strip()}")
    return True

def download_fastest_model():
    """Download the fastest available model"""
    # List of models in order of speed (fastest first)
    models = [
        ("qwen2:0.5b", "Qwen2 0.5B - Ultra fast, good quality"),
        ("llama3.2:1b", "Llama 3.2 1B - Fast and reliable"),
        ("tinyllama", "TinyLlama - Extremely fast, basic quality"),
    ]
    
    for model, description in models:
        print(f"\n🚀 Trying to install {model} ({description})...")
        
        # Check if model is already installed
        success, stdout, stderr = run_command(f"ollama list | grep {model.split(':')[0]}")
        if success:
            print(f"✅ {model} is already installed!")
            return model
        
        # Try to download the model
        print(f"📥 Downloading {model}... (this may take a few minutes)")
        success, stdout, stderr = run_command(f"ollama pull {model}")
        
        if success:
            print(f"✅ Successfully installed {model}!")
            return model
        else:
            print(f"❌ Failed to install {model}: {stderr}")
    
    print("❌ Could not install any fast models. Using default.")
    return None

def test_model(model):
    """Test the model with a simple query"""
    if not model:
        return False
        
    print(f"\n🧪 Testing {model}...")
    test_cmd = f'ollama run {model} "What is 2+2?" --verbose'
    
    start_time = time.time()
    success, stdout, stderr = run_command(test_cmd)
    end_time = time.time()
    
    if success:
        response_time = end_time - start_time
        print(f"✅ {model} responded in {response_time:.2f} seconds")
        print(f"📝 Response preview: {stdout[:100]}...")
        return True
    else:
        print(f"❌ {model} test failed: {stderr}")
        return False

def main():
    print("🚀 Setting up fastest Ollama model for RAG performance...\n")
    
    # Check if Ollama is available
    if not check_ollama():
        return False
    
    # Download fastest model
    fastest_model = download_fastest_model()
    
    if fastest_model:
        # Test the model
        if test_model(fastest_model):
            print(f"\n🎉 Setup complete! Your app will use: {fastest_model}")
            print(f"\n💡 To use this model, make sure your app.py has:")
            print(f'   ollama_model = "{fastest_model}"')
            print(f"\n🔧 Or set environment variable:")
            print(f'   export OLLAMA_MODEL="{fastest_model}"')
            return True
    
    print(f"\n⚠️ Setup completed with issues. The app should still work with default settings.")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)