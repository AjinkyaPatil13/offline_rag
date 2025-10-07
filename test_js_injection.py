#!/usr/bin/env python3
"""
Test app using JavaScript injection for Vosk speech-to-text
"""

import streamlit as st
import json
import vosk
import sounddevice as sd
import numpy as np
from pathlib import Path
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="🎤 JS Injection Test", layout="wide")

@st.cache_resource(show_spinner=False)
def get_vosk_model():
    """Load Vosk model for offline speech recognition"""
    model_path = Path("models/vosk-model-small-en-us-0.15")
    if not model_path.exists():
        st.error(f"❌ Vosk model not found at {model_path}")
        return None
    
    try:
        vosk.SetLogLevel(-1)
        model = vosk.Model(str(model_path))
        return model
    except Exception as e:
        st.error(f"❌ Failed to load Vosk model: {e}")
        return None

def inject_text_to_input(text: str):
    """Inject recognized text into the text input using JavaScript"""
    # Escape quotes for JavaScript
    safe_text = text.replace('"', '\\"')
    
    # Simpler JavaScript approach
    js_code = f"""
    (function() {{
        const inputs = window.parent.document.querySelectorAll('input[type="text"]');
        if (inputs.length > 0) {{
            const lastInput = inputs[inputs.length - 1];
            lastInput.value = "{safe_text}";
            lastInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
            return 'success';
        }}
        return 'failed';
    }})();
    """
    
    # Execute JavaScript
    try:
        result = streamlit_js_eval(js_code, key=f"inject_{hash(text)%1000}")
        return result == 'success'
    except Exception as e:
        st.error(f"JS injection error: {str(e)[:100]}...")
        return False

def recognize_offline(duration=5):
    """Record and recognize speech using Vosk"""
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
            return recognized_text
        else:
            status_placeholder.warning("🤔 No speech detected")
            return None
            
    except Exception as e:
        status_placeholder.error(f"❌ Error: {e}")
        return None

def main():
    st.title("🎙️ JavaScript Injection Test")
    st.markdown("### Test Vosk + JavaScript injection for text input")
    
    # Create columns for input and mic button
    col1, col2 = st.columns([4, 1])
    
    with col2:
        # Voice button
        if st.button("🎤", help="Click to speak (5 seconds)"):
            text = recognize_offline()
            if text:
                st.success(f"Recognized: {text}")
                # Inject text using JavaScript
                success = inject_text_to_input(text)
                if success:
                    st.info("✅ Text injected successfully!")
                else:
                    st.warning("⚠️ JavaScript injection may have failed")
            else:
                st.warning("Didn't catch that. Try again.")
    
    with col1:
        # Text input that will receive the injected text
        user_query = st.text_input(
            "Your message:",
            placeholder="Type your message or click mic to speak...",
            key="main_input"
        )
    
    # Display current input
    if user_query:
        st.write(f"**Current input:** {user_query}")
        
        if st.button("🚀 Process"):
            st.success(f"Processing: '{user_query}'")
    
    # Instructions
    st.markdown("""
    ---
    ### Instructions:
    1. Click the 🎤 button
    2. Speak clearly for 5 seconds  
    3. The text should appear in the input field automatically
    4. Edit if needed and click Process
    
    ### How it works:
    - Uses Vosk for offline speech recognition
    - Uses JavaScript injection via streamlit-js-eval
    - Directly modifies the DOM input element
    """)

if __name__ == "__main__":
    main()