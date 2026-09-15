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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        :root {
            --ink: #18324b;
            --canvas: #f4f7f8;
            --panel: #ffffff;
            --teal: #087f8c;
            --coral: #d96c4f;
            --line: #d9e2e8;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(244, 247, 248, 0.92); }
        [data-testid="stSidebar"] { background: var(--ink); }
        [data-testid="stSidebar"] * { color: #f7fbfc; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #c9d7df; }
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            background: #18324b !important;
            border-radius: 0 8px 8px 0;
            z-index: 1000;
        }
        [data-testid="stSidebarCollapsedControl"] button {
            color: #ffffff !important;
            background: #087f8c !important;
            border: 0 !important;
            min-width: 36px;
            min-height: 36px;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
        h1 { font-weight: 750; }
        [data-testid="stFileUploader"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.35rem;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #eef7f7;
            border: 1px dashed #83bfc4;
        }
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] section * {
            color: var(--ink) !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            color: #ffffff !important;
            background: var(--teal) !important;
            border-color: var(--teal) !important;
        }
        [data-testid="stFileUploaderDropzone"] svg { color: var(--teal) !important; }
        [data-testid="stImage"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.35rem;
        }
        [data-testid="stImage"] img { aspect-ratio: 1 / 1; object-fit: contain; }
        [data-testid="stImage"] + div { color: var(--teal); font-weight: 650; }
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: var(--coral);
            border-color: var(--coral);
        }
    </style>
    """,
    unsafe_allow_html=True
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

def recommend(features, feature_list, result_count):
    neighbors = NearestNeighbors(
        n_neighbors=min(result_count, len(feature_list)),
        algorithm='brute',
        metric='euclidean'
    )
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return distances[0], indices[0]

with st.sidebar:
    st.header('How to use')
    st.write('Upload a clear clothing or accessory photo to find similar items.')
    st.caption('Supported formats: JPG, JPEG, PNG, and WEBP')
    result_count = st.slider('Number of results', min_value=1, max_value=20, value=5)

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
                distances, indices = recommend(features, feature_list, result_count)

            st.subheader(f'Similar Items ({len(indices)})')
            for row_start in range(0, len(indices), 5):
                row_indices = indices[row_start:row_start + 5]
                recommendation_columns = st.columns(5)
                for offset, column in enumerate(recommendation_columns):
                    with column:
                        if offset < len(row_indices):
                            position = row_start + offset
                            match_score = max(0, min(100, round((1 - distances[position] / 2) * 100)))
                            st.image(
                                filenames[indices[position]],
                                use_container_width=True,
                                caption=f'{match_score}% visual match'
                            )

