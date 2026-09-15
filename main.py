import streamlit as st
import streamlit.components.v1 as components
import os
import uuid
import html
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

def render_image_gallery(image_urls, match_scores):
    gallery_items = []
    for image_url, match_score in zip(image_urls, match_scores):
        safe_url = html.escape(str(image_url), quote=True)
        gallery_items.append(
            f'''<button class="gallery-item" type="button" onclick="openViewer(this)"
                data-image="{safe_url}" aria-label="Open image showing a {match_score}% visual match">
                <img src="{safe_url}" alt="{match_score}% visual match">
                <span>{match_score}% visual match</span>
            </button>'''
        )

    gallery_html = f'''
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: sans-serif; background: transparent; }}
        .gallery {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 16px; }}
        .gallery-item {{ min-width: 0; border: 1px solid #d9e2e8; border-radius: 8px;
            padding: 6px; background: #ffffff; cursor: zoom-in; text-align: center; }}
        .gallery-item:hover {{ border-color: #087f8c; box-shadow: 0 5px 16px rgba(24, 50, 75, .14); }}
        .gallery-item img {{ display: block; width: 100%; aspect-ratio: 1 / 1;
            object-fit: contain; border-radius: 5px; }}
        .gallery-item span {{ display: block; padding: 8px 2px 4px; color: #087f8c;
            font-size: 0.82rem; font-weight: 650; }}
        .viewer {{ display: none; position: fixed; inset: 0; z-index: 10; padding: 24px;
            background: rgba(10, 23, 35, .88); align-items: center; justify-content: center; }}
        .viewer.open {{ display: flex; }}
        .viewer img {{ width: min(78vw, 760px); height: min(78vh, 760px); object-fit: cover;
            border-radius: 8px; background: #ffffff; box-shadow: 0 14px 50px rgba(0, 0, 0, .35); }}
        .close {{ position: absolute; top: 14px; right: 20px; border: 0; background: transparent;
            color: #ffffff; font-size: 2rem; line-height: 1; cursor: pointer; }}
        @media (max-width: 800px) {{ .gallery {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    </style>
    <div class="gallery">{''.join(gallery_items)}</div>
    <div class="viewer" id="viewer" onclick="closeOnBackdrop(event)">
        <button class="close" type="button" onclick="closeViewer()" aria-label="Close image">&times;</button>
        <img id="large-image" alt="Enlarged recommendation">
    </div>
    <script>
        const viewer = document.getElementById('viewer');
        const largeImage = document.getElementById('large-image');
        function openViewer(item) {{ largeImage.src = item.dataset.image; viewer.classList.add('open'); }}
        function closeViewer() {{ viewer.classList.remove('open'); largeImage.src = ''; }}
        function closeOnBackdrop(event) {{ if (event.target === viewer) closeViewer(); }}
        document.addEventListener('keydown', event => {{ if (event.key === 'Escape') closeViewer(); }});
    </script>'''
    components.html(gallery_html, height=360 if len(image_urls) <= 5 else 360 * ((len(image_urls) + 4) // 5), scrolling=False)

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
                row_urls = [filenames[index] for index in row_indices]
                row_scores = [
                    max(0, min(100, round((1 - distances[row_start + offset] / 2) * 100)))
                    for offset in range(len(row_indices))
                ]
                render_image_gallery(row_urls, row_scores)

