#!/usr/bin/env python3
"""
Working speech-to-text with proper Streamlit input handling
"""

import streamlit as st
import json
import vosk
import sounddevice as sd
import numpy as np
import time
from pathlib import Path

st.set_page_config(page_title="🎤 Working Speech Input", layout="wide")

@st.cache_resource(show_spinner=False)
def get_vosk_model():
    """Load Vosk model"""
    model_path = Path("models/vosk-model-small-en-us-0.15")
    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}")
        return None
    
    try:
        vosk.SetLogLevel(-1)
        return vosk.Model(str(model_path))
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

def record_speech():
    """Record and recognize speech"""
    model = get_vosk_model()
    if not model:
        return None
    
    status = st.empty()
    
    try:
        rec = vosk.KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        
        status.info("🎤 Recording 5 seconds... Speak now!")
        
        audio_data = sd.rec(frames=80000, samplerate=16000, channels=1, dtype=np.int16)
        sd.wait()
        
        status.info("🔄 Processing...")
        
        # Process audio
        audio_bytes = audio_data.tobytes()
        results = []
        
        for i in range(0, len(audio_bytes), 4000):
            chunk = audio_bytes[i:i + 4000]
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                if result.get('text', '').strip():
                    results.append(result['text'].strip())
        
        final = json.loads(rec.FinalResult())
        if final.get('text', '').strip():
            results.append(final['text'].strip())
        
        text = ' '.join(results).strip()
        
        if text:
            status.success(f"✅ Recognized: '{text}'")
            time.sleep(1)
            status.empty()
            return text
        else:
            status.warning("🤔 No speech detected")
            time.sleep(2)
            status.empty()
            return None
            
    except Exception as e:
        status.error(f"❌ Error: {e}")
        time.sleep(2)
        status.empty()
        return None

def main():
    st.title("🎙️ Working Speech-to-Text Input")
    
    # Initialize session state
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'speech_result' not in st.session_state:
        st.session_state.speech_result = ""
    if 'update_input' not in st.session_state:
        st.session_state.update_input = False
    
    col1, col2 = st.columns([4, 1])
    
    with col2:
        if st.button("🎤 Speak", key="mic_button"):
            with st.spinner("Processing speech..."):
                recognized = record_speech()
                if recognized:
                    # Store the result and flag for update
                    st.session_state.speech_result = recognized
                    st.session_state.update_input = True
                    st.rerun()
    
    with col1:
        # If we have a speech result and need to update, use it as value
        input_value = ""
        if st.session_state.update_input and st.session_state.speech_result:
            input_value = st.session_state.speech_result
            st.session_state.update_input = False  # Reset flag
        
        # Text input
        user_input = st.text_input(
            "Your message:",
            value=input_value,
            placeholder="Type your message or click mic to speak...",
            key="text_input"
        )
        
        # Update session state
        st.session_state.current_text = user_input
    
    # Show current input
    if st.session_state.current_text:
        st.write(f"**Current:** {st.session_state.current_text}")
        
        if st.button("🚀 Process"):
            st.success(f"Processing: '{st.session_state.current_text}'")
    
    # Instructions
    st.markdown("""
    ---
    ### How to use:
    1. Click 🎤 Speak button
    2. Speak clearly for 5 seconds
    3. Text should appear in input field
    4. Edit if needed and process
    
    ### Technical approach:
    - Uses session state to manage text updates
    - Reruns the app when speech is recognized
    - Sets input value before widget creation
    """)

if __name__ == "__main__":
    main()