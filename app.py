import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import shap
import time
import io
import json
import cv2
import traceback
from dotenv import load_dotenv
import os
import globals as gl
from explanier import *

load_dotenv()
device = os.getenv("DEVICE")

# Set page configuration
st.set_page_config(
    page_title="Image Classification Explainer", page_icon="🔍", layout="wide"
)

# App title and description
st.title("Image Classification Explainer")
st.markdown(
    """
This app uses SHAP (SHapley Additive exPlanations) to explain the decisions made by an image classification model.
You can either use a pre-trained model or upload your own custom PyTorch model.
"""
)

# Sidebar for model settings
st.sidebar.header("Model Settings")
model_source = st.sidebar.radio(
    "Model Source:", ["Use Pre-trained Model", "Upload Custom Model"]
)


# Function to load a custom PyTorch model

# Main function
# def main():
# Model selection/upload logic
class_names = load_imagenet_labels()
custom_transform = None
input_shape = (128, 128, 3)

if model_source == "Use Pre-trained Model":
    model_name = st.sidebar.selectbox(
        "Select a pre-trained model:", ["ResNet18", "MobileNetV2", "VGG16"]
    )

    # Load the selected model
    with st.spinner("Loading pre-trained model..."):
        gl.model = load_pretrained_model(model_name)

else:  # Upload Custom Model
    st.sidebar.markdown("### Upload Your Custom Model")

    model_file = st.sidebar.file_uploader(
        "Upload Complete model file (.pt or .pth)", type=["pt", "pth"]
    )
    if model_file:
        model_bytes = io.BytesIO(model_file.read())
        with st.spinner("Loading custom model..."):
            gl.model, error = load_custom_model(model_bytes)
            if error:
                st.sidebar.error(error)

    # Class names for custom model
    st.sidebar.markdown("### Class Names")
    class_names_option = st.sidebar.radio(
        "Class names:",
        ["Use default (ImageNet)", "Upload JSON file", "Enter manually"],
    )

    if class_names_option == "Upload JSON file":
        class_file = st.sidebar.file_uploader(
            "Upload class names (JSON format)", type=["json"]
        )
        if class_file:
            try:
                class_names = json.load(class_file)
                st.sidebar.success(f"Loaded {len(class_names)} class names")
            except:
                st.sidebar.error("Error loading class names. Using defaults.")

    elif class_names_option == "Enter manually":
        class_input = st.sidebar.text_area("Enter class names (one per line)")
        if class_input:
            class_names = [
                line.strip() for line in class_input.split("\n") if line.strip()
            ]
            st.sidebar.success(f"Loaded {len(class_names)} class names")

    # Custom preprocessing
    st.sidebar.markdown("### Custom Preprocessing")
    use_custom_preproc = st.sidebar.checkbox("Use custom preprocessing parameters")

    if use_custom_preproc:
        img_size = st.sidebar.number_input(
            "Image size", min_value=32, max_value=512, value=224
        )
        mean_r = st.sidebar.slider(
            "Mean (R)", min_value=0.0, max_value=1.0, value=0.485
        )
        mean_g = st.sidebar.slider(
            "Mean (G)", min_value=0.0, max_value=1.0, value=0.456
        )
        mean_b = st.sidebar.slider(
            "Mean (B)", min_value=0.0, max_value=1.0, value=0.406
        )
        std_r = st.sidebar.slider("Std (R)", min_value=0.01, max_value=1.0, value=0.229)
        std_g = st.sidebar.slider("Std (G)", min_value=0.01, max_value=1.0, value=0.224)
        std_b = st.sidebar.slider("Std (B)", min_value=0.01, max_value=1.0, value=0.225)

        custom_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[mean_r, mean_g, mean_b], std=[std_r, std_g, std_b]
                ),
            ]
        )

        input_shape = (3, img_size, img_size)

# Image upload section
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if we have both a model and an image
if gl.model is not None and uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image

    # Make prediction
    with st.spinner("Making prediction..."):
        try:
            # class_idx, probabilities = predict(model, img_tensor)

            # Convert tensor to numpy array for visualization
            gl.model = gl.model.to(gl.device)
            x = preprocess(image, input_shape[:2])
            class_idx, probabilities = predict(x)

            # Display prediction results
            with col2:
                st.subheader("Prediction Results")
                top_k = min(5, len(probabilities))
                top_probs, top_classes = torch.topk(probabilities, top_k)

                # Display top predictions
                for i, (prob, class_id) in enumerate(zip(top_probs, top_classes)):
                    class_name = (
                        class_names[class_id]
                        if class_id < len(class_names)
                        else f"Class {class_id}"
                    )
                    st.write(f"{i+1}. {class_name}: {prob.item()*100:.2f}%")

                # Highlight top prediction
                top_class_name = (
                    class_names[top_classes[0]]
                    if top_classes[0] < len(class_names)
                    else f"Class {top_classes[0]}"
                )
                st.success(
                    f"Top prediction: **{top_class_name}** with {top_probs[0].item()*100:.2f}% confidence"
                )

            # # Generate SHAP explanation if user wants it
            if st.button("Explain this prediction with SHAP"):
                with st.spinner("Generating SHAP values (this may take a minute)..."):
                    start_time = time.time()

                    # Generate SHAP values for the top predicted class

                    shap_values = generate_shap_values(image, input_shape, class_names)
                    end_time = time.time()
                    st.info(
                        f"SHAP explanation generated in {end_time - start_time:.2f} seconds"
                    )

                #     # Plot SHAP explanations
                st.subheader(f"SHAP Explanation for '{top_class_name}'")

                #     # Create SHAP visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                #     shap_img = shap.image_plot(
                #         shap_values.values[0],
                #         np.transpose(img_array, (0, 2, 3, 1))[0],
                #         show=False,
                #     )
                shap.image_plot(
                    shap_values=shap_values.values,
                    pixel_values=shap_values.data,
                    labels=shap_values.output_names,
                    true_labels=[class_names[class_idx]],
                    show=False,
                )
                st.pyplot(plt)

                #     # Explanation of the visualization
                st.markdown(
                    """
                ### How to interpret the visualization:
                
                - **Original image** is shown on the left
                - **SHAP values** are shown on the right
                - **Red areas** indicate features that pushed the prediction toward the class
                - **Blue areas** indicate features that pushed the prediction away from the class
                - **Intensity** of the color indicates the strength of the effect
                
                This helps you understand which parts of the image were most important for the model's decision.
                """
                )

        except Exception as e:
            st.error(
                f"Error during prediction or explanation: {str(traceback.format_exc())}"
            )
            st.info(
                "This might be due to model compatibility issues. Make sure your custom model can process the image format and size."
            )

elif gl.model is None and uploaded_file is not None:
    st.warning("Please select or upload a model to analyze the image.")

elif (
    uploaded_file is None
    and model_source == "Upload Custom Model"
    and gl.model is not None
):
    st.success("Custom model loaded successfully! Now upload an image to analyze.")


# if __name__ == "__main__":
#     main()
