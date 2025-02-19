import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import hashlib
import os
from skimage.transform import resize

def add_custom_css():
    st.markdown("""
        <style>
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes rainbow {
            0% { color: #ff0000; }
            17% { color: #ff8800; }
            33% { color: #ffff00; }
            50% { color: #00ff00; }
            67% { color: #0000ff; }
            83% { color: #8800ff; }
            100% { color: #ff0000; }
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-20px); }
            60% { transform: translateY(-10px); }
        }

        .main {
            padding-bottom: 100px;
        }
        
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            color: white;
        }
        
        .title-text {
            font-size: 3.5em;
            font-weight: bold;
            text-align: center;
            animation: rainbow 8s linear infinite;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
        }

        .genre-box {
            background: rgba(255,255,255,0.15);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            margin: 15px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            transition: transform 0.3s ease;
        }
        
        .genre-box:hover {
            transform: translateY(-5px);
        }
        
        .confidence-bar {
            height: 25px;
            background: linear-gradient(90deg, #00ff87 0%, #60efff 100%);
            border-radius: 12px;
            transition: width 1s ease-in-out;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .genre-icon {
            font-size: 50px;
            margin-bottom: 15px;
            animation: bounce 2s infinite;
        }
        
        .progress-label {
            position: absolute;
            right: 10px;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            line-height: 25px;
            padding-right: 10px;
        }
        
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: rgba(0,0,0,0.9);
            backdrop-filter: blur(10px);
            color: white;
            padding: 20px;
            text-align: center;
            z-index: 999;
            box-shadow: 0 -5px 25px rgba(0,0,0,0.3);
        }
        </style>
    """, unsafe_allow_html=True)

# Genre icons with animations
genre_icons = {
    'blues': '🎺', 'classical': '🎻', 'country': '🤠',
    'disco': '🕺', 'hiphop': '🎤', 'jazz': '🎷',
    'metal': '🤘', 'pop': '🎵', 'reggae': '🌴', 'rock': '🎸'
}

# Disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@st.cache_resource
def load_model():
    """Load the model once and cache it"""
    try:
        model = tf.keras.models.load_model("iotamodel1.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_file_hash(file_content):
    """Generate a unique hash for the file content"""
    return hashlib.md5(file_content).hexdigest()

def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily and return the path"""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def predict_genre(audio_path, target_shape=(100, 100)):
    """predict music genre from audio file"""
    model = load_model()
    if model is None:
        return None
        
    # Load audio file
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    
    # Compute spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    
    # Resizing spectrogram
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    
    # Preparing the input
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

    # Make predictions
    prediction = model.predict(mel_spectrogram)
    
    return prediction

def main():
    # Add CSS
    add_custom_css()
    
    st.markdown("""
        <div class="title-text">
           🎵 Music Genre Classifier
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="upload-zone">
            <div class="upload-content">
                <center class="upload-text">Drop your audio file here</center>
                <center class="upload-subtext">or click to browse</center>
                <center class="upload-subtext">(Supports MP3 & WAV)</center>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["mp3", "wav"], key="music_uploader", label_visibility="collapsed")

    if uploaded_file is not None:
        # Get file hash for uniqueness
        file_content = uploaded_file.getvalue()
        file_hash = get_file_hash(file_content)
        
        # Use file hash in session state
        if 'last_processed_file' not in st.session_state:
            st.session_state.last_processed_file = None
        
        # Check if this is a new file
        if st.session_state.last_processed_file != file_hash:
            st.session_state.last_processed_file = file_hash
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="genre-box">
                        <h3>🎧 Now Playing</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.audio(uploaded_file, format="audio/wav")

            with col2:
                with st.spinner("🎼 Analyzing your track..."):
                    # Save uploaded file temporarily
                    temp_file_path = save_uploaded_file(uploaded_file)
                    
                    try:
                        # Get predictions using predict_genre function
                        predictions = predict_genre(temp_file_path)
                        if predictions is None:
                            st.error("Could not process the audio file. Please check if the model is loaded correctly.")
                            return
                            
                        genre_labels = ['blues', 'classical', 'country', 'disco', 
                                      'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                        
                        # Get the predicted genre and probabilities
                        genre_index = np.argmax(predictions[0])
                        probabilities = predictions[0]
                        
                        predicted_genre = genre_labels[genre_index]
                        confidence = probabilities[genre_index] * 100

                        # Display predicted genre
                        st.markdown(f"""
                            <div class="genre-box">
                                <div class="genre-icon">{genre_icons.get(predicted_genre, '🎵')}</div>
                                <h2 style="animation: rainbow 8s linear infinite;">
                                    {predicted_genre.upper()}
                                </h2>
                                <div style="position: relative; background: rgba(255,255,255,0.1); border-radius: 12px; margin: 10px 0;">
                                    <div class="confidence-bar" style="width: {confidence}%;">
                                        <span class="progress-label">{confidence:.1f}%</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        # Display all genre probabilities
                        st.markdown("### Genre Probabilities")
                        for label, prob in zip(genre_labels, probabilities):
                            prob_percentage = prob * 100
                            st.markdown(f"""
                                <div class="genre-box" style="padding: 10px; margin: 5px 0;">
                                    <div style="display: flex; align-items: center;">
                                        <div style="width: 100px;">{genre_icons.get(label, '🎵')} {label}</div>
                                        <div style="flex-grow: 1; background: rgba(255,255,255,0.1); border-radius: 10px; margin-left: 10px;">
                                            <div class="confidence-bar" style="width: {prob_percentage}%;">
                                                <span class="progress-label">{prob_percentage:.1f}%</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

    st.markdown(f"""
        <div class="footer">
            <p>🎵 Music Genre Classifier | Made with ❤️ </p>
            <p style="animation: rainbow 8s linear infinite;">Powered by VEDANT KASAR✨</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Configure page
    st.set_page_config(
        page_title="🎶 Music Genre Classifier",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Clear tensorflow session at startup
    tf.keras.backend.clear_session()
    
    # Run main app
    main()