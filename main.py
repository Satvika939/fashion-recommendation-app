import streamlit as st
import os
import uuid
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide"
)

os.makedirs('uploads', exist_ok=True)

@st.cache_data
def load_recommendation_data():
    with open('embeddings.pkl', 'rb') as embeddings_file:
        feature_list = np.array(pickle.load(embeddings_file))
    with open('filenames.pkl', 'rb') as filenames_file:
        filenames = pickle.load(filenames_file)
    return feature_list, filenames

@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tensorflow.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

feature_list, filenames = load_recommendation_data()
model = load_model()

st.title('Fashion Recommender')
st.write('Upload a clothing image and discover visually similar pieces from the collection.')

def save_uploaded_file(uploaded_file):
    try:
        safe_name = f'{uuid.uuid4().hex}_{os.path.basename(uploaded_file.name)}'
        file_path = os.path.join('uploads', safe_name)
        with open(file_path, 'wb') as uploaded_image:
            uploaded_image.write(uploaded_file.getbuffer())
        return file_path
    except OSError:
        return None

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=min(6, len(feature_list)), algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

with st.sidebar:
    st.header('How to use')
    st.write('Upload a clear clothing or accessory photo to find similar items.')
    st.caption('Supported formats: JPG, JPEG, PNG, and WEBP')

uploaded_file = st.file_uploader(
    'Choose an image',
    type=['jpg', 'jpeg', 'png', 'webp'],
    help='Upload one clothing image to receive recommendations.'
)

if uploaded_file is not None:
    try:
        display_image = Image.open(uploaded_file)
        display_image.verify()
        uploaded_file.seek(0)
        display_image = Image.open(uploaded_file).convert('RGB')
    except (OSError, ValueError):
        st.error('Please upload a valid image file.')
    else:
        st.subheader('Your Image')
        st.image(display_image, width=320)

        saved_path = save_uploaded_file(uploaded_file)
        if saved_path is None:
            st.error('The image could not be saved. Please try again.')
        else:
            with st.spinner('Finding similar fashion items...'):
                features = feature_extraction(saved_path, model)
                indices = recommend(features, feature_list)

            st.subheader('Recommended Items')
            recommendation_columns = st.columns(5)
            for position, column in enumerate(recommendation_columns):
                if position >= len(indices[0]):
                    break
                with column:
                    st.image(
                        filenames[indices[0][position]],
                        use_container_width=True,
                        caption=f'Recommendation {position + 1}'
                    )

