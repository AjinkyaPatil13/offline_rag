#!/usr/bin/env python3
"""
Test script for Vosk offline speech recognition
"""

import json
import vosk
import sounddevice as sd
import numpy as np
from pathlib import Path

def test_vosk():
    """Test Vosk offline speech recognition"""
    print("🎙️ Testing Vosk offline speech recognition...")
    
    # Check model exists
    model_path = Path("models/vosk-model-small-en-us-0.15")
    if not model_path.exists():
        print("❌ Vosk model not found! Run setup_vosk.py first.")
        return False
    
    try:
        # Load model
        print("📦 Loading Vosk model...")
        vosk.SetLogLevel(-1)  # Reduce verbosity
        model = vosk.Model(str(model_path))
        print("✅ Model loaded successfully!")
        
        # Setup recognizer
        sample_rate = 16000
        rec = vosk.KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)
        
        # Test microphone
        print("🎤 Testing microphone access...")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if not input_devices:
            print("❌ No input devices found!")
            return False
        
        print(f"✅ Found {len(input_devices)} input device(s)")
        for i, device in enumerate(input_devices):
            print(f"   {i}: {device['name']}")
        
        # Record test audio
        duration = 3  # seconds
        print(f"\n🎤 Recording for {duration} seconds... Say something!")
        
        audio_data = sd.rec(
            frames=int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        print("🔄 Processing audio...")
        
        # Process with Vosk
        audio_bytes = audio_data.tobytes()
        
        # Process in chunks
        chunk_size = 4000
        results = []
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                if result.get('text', '').strip():
                    results.append(result['text'].strip())
        
        # Final result
        final_result = json.loads(rec.FinalResult())
        if final_result.get('text', '').strip():
            results.append(final_result['text'].strip())
        
        # Combine results
        recognized_text = ' '.join(results).strip()
        
        if recognized_text:
            print(f"✅ Speech recognized: '{recognized_text}'")
            print("🎯 Vosk offline speech recognition is working!")
            return True
        else:
            print("⚠️ No speech detected. Try speaking louder or closer to the microphone.")
            print("🎯 Vosk is working, but didn't detect clear speech.")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_vosk()
    if success:
        print("\n🚀 Ready for offline speech recognition in Streamlit!")
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")