from __future__ import annotations

from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image

from annotated_images.app_support import (
    MODEL_LABELS,
    discover_latest_checkpoints,
    draw_uploaded_prediction,
    load_class_names,
    load_inference_model,
    predict_uploaded_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
PREPARED_ROOT = PROJECT_ROOT / "artifacts" / "prepared"
OOD_DIR = PROJECT_ROOT / "OOD_Test_Files"
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]


st.set_page_config(page_title="Annotated Images Demo", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 75, 75, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 163, 108, 0.16), transparent 24%),
            linear-gradient(180deg, #fffdf7 0%, #f5f7f2 100%);
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.3rem 0;
        font-size: 2.2rem;
    }
    .hero p {
        margin: 0;
        color: #425466;
        font-size: 1rem;
    }
    .metric-card {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    </style>
    <div class="hero">
        <h1>Insect Annotation Demo</h1>
        <p>Upload an image, pick any trained detector with available weights, and inspect the predicted boxes, class names, and confidence scores.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


available_checkpoints = discover_latest_checkpoints(RUNS_DIR)
if not available_checkpoints:
    st.error(f"No trained checkpoints found under {RUNS_DIR}.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_model_bundle(model_type: str, checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    class_names = load_class_names(PREPARED_ROOT, checkpoint, model_type)
    model = load_inference_model(model_type, checkpoint, class_names, device="cpu")
    return model, class_names


model_options = {
    f"{MODEL_LABELS[model_type]}": model_type
    for model_type in ("yolo", "rtdetr", "fasterrcnn")
    if model_type in available_checkpoints
}
ood_images = sorted(
    path.name for path in OOD_DIR.iterdir() if path.is_file() and path.suffix.lower().lstrip(".") in SUPPORTED_TYPES
) if OOD_DIR.exists() else []

left_col, right_col = st.columns([0.9, 1.4], gap="large")

with left_col:
    selected_label = st.selectbox("Model", options=list(model_options.keys()))
    selected_model_type = model_options[selected_label]
    selected_checkpoint = available_checkpoints[selected_model_type]
    threshold = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    source_mode = st.radio("Image source", options=["Upload", "OOD_Test_Files"], horizontal=True)

    uploaded_file = None
    selected_ood_image = None
    if source_mode == "Upload":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=SUPPORTED_TYPES,
            help="Supported: PNG, JPG, JPEG, WEBP, BMP, TIF, TIFF",
        )
    else:
        if ood_images:
            selected_ood_image = st.selectbox("OOD image", options=ood_images)
        else:
            st.info(f"No supported images found under {OOD_DIR}.")

    st.markdown(
        f"""
        <div class="metric-card">
            <strong>Latest checkpoint</strong><br>
            <code>{selected_checkpoint.relative_to(PROJECT_ROOT)}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    if source_mode == "Upload" and uploaded_file is None:
        st.info("Upload an image to run inference.")
        st.stop()
    if source_mode == "OOD_Test_Files" and selected_ood_image is None:
        st.info("Select an image from OOD_Test_Files to run inference.")
        st.stop()

    if source_mode == "Upload":
        image = Image.open(uploaded_file).convert("RGB")
        original_caption = "Original upload"
    else:
        image = Image.open(OOD_DIR / selected_ood_image).convert("RGB")
        original_caption = f"OOD sample: {selected_ood_image}"

    with st.spinner(f"Running {selected_label} on CPU..."):
        model, class_names = get_model_bundle(selected_model_type, str(selected_checkpoint.resolve()))
        predictions = predict_uploaded_image(
            model_type=selected_model_type,
            model=model,
            image=image,
            class_names=class_names,
            score_threshold=threshold,
            device="cpu",
        )
        rendered = draw_uploaded_prediction(image, predictions)

    per_class_counts = Counter(str(item["label"]) for item in predictions)

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Detections", len(predictions))
    stat_col2.metric("Classes Found", len({item["label"] for item in predictions}))
    stat_col3.metric("Total Count", sum(per_class_counts.values()))

    image_col1, image_col2 = st.columns(2, gap="large")
    image_col1.image(image, caption=original_caption, use_container_width=True)
    image_col2.image(rendered, caption="Predicted annotations", use_container_width=True)

    if per_class_counts:
        st.subheader("Per-Class Counts")
        st.dataframe(
            [
                {"class_name": class_name, "count": count}
                for class_name, count in sorted(per_class_counts.items())
            ],
            use_container_width=True,
            hide_index=True,
        )

    if predictions:
        st.subheader("Detections")
        st.dataframe(
            [
                {
                    "class_name": item["label"],
                    "confidence": round(float(item["score"]), 4),
                    "xmin": round(float(item["box"][0]), 1),
                    "ymin": round(float(item["box"][1]), 1),
                    "xmax": round(float(item["box"][2]), 1),
                    "ymax": round(float(item["box"][3]), 1),
                }
                for item in predictions
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No detections passed the current confidence threshold.")
