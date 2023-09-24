# Importing Libraries
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

# Utility Functions
def load_lottie(filepath):
    """
    Load and return the contents of a Lottie file.
    
    Args:
        filepath (str): Path to the Lottie file.
    
    Returns:
        dict: JSON contents of the Lottie file.
    """
    with open(filepath, 'r') as f:
        return json.load(f)
    
# Initial configuration
st.set_page_config(
    page_title="😃😐😞SentimentaX: Sentiment Analyzer",
    initial_sidebar_state="expanded",
    layout='wide',
) 
# Loading model and tokenizer
model = tf.keras.models.load_model("Models/LSTM.h5")
with open("Models/tokenizer.pkl", 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
    
# Sentiment Mappings
class_mapping={
0:'Negative',
1:'Neutral',
2:'Positive'}

# Setting max length
max_sequence_length = 100 

# Streamlit Application
st.title("😃😐😞SentimentaX: Sentiment Analyzer")
st.markdown("#### It is an advanced machine learning application designed to accurately identify and classify various types of texts either positive, negative or neutral. It is trained on reddit dataset and uses LSTM model at its backend.")
left, right = st.columns(2)
with left:
    user_input = st.text_area("Enter Text")
    button = st.button("Analyze")
    if button:
        if user_input=="":
            st.error("Enter the text")
        else:
            # User input preprocessing
            user_sequences = tokenizer.texts_to_sequences([user_input])
            user_padded = tf.keras.preprocessing.sequence.pad_sequences(user_sequences, maxlen=max_sequence_length)

            # Making classification
            user_predictions = model.predict(user_padded)[0]
            user_pred_classes = np.argmax(user_predictions)
            predicted_prob = user_predictions[user_pred_classes]
            st.markdown(f'#### Predicted Class: {class_mapping[user_pred_classes]}')
            st.write('#### Predicted Probability: ', predicted_prob)
            st.write('### WordCloud')
            wordcloud = WordCloud(width=800, height=400, mode="RGBA", background_color=None).generate(user_input)
            image = wordcloud.to_image()
            data = np.array(image)
            data[data[:, :, 3] == 0] = [255, 255, 255, 0]
            st.image(data)
            
with right:
    animation1 = load_lottie("Vectors/1.json")
    st_lottie(animation1, loop=True)