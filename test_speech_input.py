#!/usr/bin/env python3
"""
Simple test app to verify Vosk speech-to-text input functionality
"""

import streamlit as st
import json
import vosk
import sounddevice as sd
import numpy as np
import time
from pathlib import Path

st.set_page_config(page_title="Speech-to-Text Test", layout="wide")

@st.cache_resource(show_spinner=False)
def get_vosk_model():
    """Load Vosk model for offline speech recognition"""
    model_path = Path("models/vosk-model-small-en-us-0.15")
    if not model_path.exists():
        st.error(f"❌ Vosk model not found at {model_path}. Please run setup_vosk.py first.")
        return None
    
    try:
        vosk.SetLogLevel(-1)
        model = vosk.Model(str(model_path))
        return model
    except Exception as e:
        st.error(f"❌ Failed to load Vosk model: {e}")
        return None

def record_speech():
    """Record and transcribe speech using Vosk"""
    model = get_vosk_model()
    if model is None:
        return None
    
    # Create status placeholder
    status_placeholder = st.empty()
    
    try:
        # Vosk setup
        sample_rate = 16000
        rec = vosk.KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)
        
        duration = 5
        
        status_placeholder.info(f"🎤 Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            frames=int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        status_placeholder.info("🔄 Processing speech...")
        
        # Process with Vosk
        audio_bytes = audio_data.tobytes()
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
        
        recognized_text = ' '.join(results).strip()
        
        if recognized_text:
            status_placeholder.success(f"✅ Recognized: '{recognized_text}'")
            time.sleep(1)
            status_placeholder.empty()
            return recognized_text
        else:
            status_placeholder.warning("🤔 No speech detected")
            time.sleep(2)
            status_placeholder.empty()
            return None
            
    except Exception as e:
        status_placeholder.error(f"❌ Error: {e}")
        time.sleep(2)
        status_placeholder.empty()
        return None

def main():
    st.title("🎙️ Speech-to-Text Input Test")
    st.markdown("### Test Vosk offline speech recognition in text input")
    
    # Initialize session state
    if 'speech_text' not in st.session_state:
        st.session_state.speech_text = ""
    
    # Create columns for input and mic button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Text input that gets updated with speech
        user_input = st.text_input(
            "Your message:",
            value=st.session_state.speech_text,
            placeholder="Type or click mic to speak...",
            key="text_input"
        )
        
        # Keep session state in sync
        if user_input != st.session_state.speech_text:
            st.session_state.speech_text = user_input
    
    with col2:
        # Microphone button
        if st.button("🎤", help="Click to speak (5 seconds)"):
            recognized_text = record_speech()
            
            if recognized_text:
                # Update session state and rerun to show in input
                st.session_state.speech_text = recognized_text
                st.rerun()
    
    # Display current input
    if st.session_state.speech_text:
        st.write(f"**Current input:** {st.session_state.speech_text}")
        
        if st.button("🚀 Process"):
            st.success(f"Processing: '{st.session_state.speech_text}'")
    
    # Instructions
    st.markdown("""
    ---
    ### Instructions:
    1. Click the 🎤 button
    2. Speak clearly for 5 seconds
    3. The recognized text should appear in the input field above
    4. Edit if needed and click Process
    """)

if __name__ == "__main__":
    main()