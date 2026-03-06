# ==========================================================
# 🌾 Crop Disease & Soil Analysis Platform
# Models: MobileNetV2, EfficientNet-B0, Soil Model
# ==========================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- DEVICE SETUP ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="🌾 Crop Disease & Soil Analysis", page_icon="🌿", layout="wide")
st.title("🌾 Crop Disease & Soil Analysis Platform")

# ---------------- TABS ---------------- #
tab1, tab2 = st.tabs(["🍅 Tomato Leaf Disease Detection", "🌱 Soil Type Detection & Crop Recommendation"])

# ==========================================================
# TAB 1: TOMATO LEAF DISEASE DETECTION
# ==========================================================
with tab1:
    st.header("🍅 Tomato Leaf Disease Detection System")

    disease_classes = [
        "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
        "Septoria Leaf Spot", "Spider Mites", "Target Spot",
        "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
    ]
    num_classes = len(disease_classes)

    # ---------------- Model Loader ---------------- #
    @st.cache_resource
    def load_model(model_name, num_classes):
        model_file = f"{model_name}_tomato_leaf_disease.pth"

        if model_name == "MobileNetV2":
            model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == "EfficientNetB0":
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            st.error("❌ Unknown model name.")
            return None

        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            st.success(f"✅ {model_name} loaded successfully!")
            return model
        else:
            st.warning(f"⚠️ Model file '{model_file}' not found.")
            return None

    mobilenet = load_model("MobileNetV2", num_classes)
    efficientnet = load_model("EfficientNetB0", num_classes)

    # ---------------- Upload Image ---------------- #
    if mobilenet and efficientnet:
        uploaded_file = st.file_uploader("📸 Upload a Tomato Leaf Image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Tomato Leaf", width=240)
            st.markdown("</div>", unsafe_allow_html=True)

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            img_tensor = preprocess(image).unsqueeze(0).to(device)

            # ---------------- Prediction ---------------- #
            with torch.no_grad():
                output_m = torch.softmax(mobilenet(img_tensor), dim=1)
                output_e = torch.softmax(efficientnet(img_tensor), dim=1)

            conf_m, pred_m = torch.max(output_m, 1)
            conf_e, pred_e = torch.max(output_e, 1)

            pred_m_label = disease_classes[pred_m.item()]
            pred_e_label = disease_classes[pred_e.item()]

            if conf_m.item() < 0.5:
                pred_m_label = "Healthy"
            if conf_e.item() < 0.5:
                pred_e_label = "Healthy"

            col1, col2 = st.columns(2)
            col1.metric("MobileNetV2 Prediction", pred_m_label, f"{conf_m.item()*100:.2f}%")
            col2.metric("EfficientNetB0 Prediction", pred_e_label, f"{conf_e.item()*100:.2f}%")

            # ---------------- One-Line Suggested Care ---------------- #
            cure_dict = {
                "Bacterial Spot": "Apply copper sprays and remove infected leaves to control spread.",
                "Early Blight": "Use mancozeb or chlorothalonil and remove infected lower leaves.",
                "Late Blight": "Remove infected plants immediately and apply metalaxyl fungicide.",
                "Leaf Mold": "Improve ventilation and apply copper-based fungicides.",
                "Septoria Leaf Spot": "Remove infected leaves and use preventive fungicides.",
                "Spider Mites": "Increase humidity and apply neem oil or miticides.",
                "Target Spot": "Avoid overhead watering and use broad-spectrum fungicides.",
                "Yellow Leaf Curl Virus": "Remove infected plants and control whiteflies with neem oil.",
                "Mosaic Virus": "Use virus-free seeds and control aphids with biological methods.",
                "Healthy": "No disease detected — maintain regular monitoring and hygiene."
            }

            st.markdown("### 🌿💊 Suggested Care")
            st.info(f"🩺 **{pred_m_label}:** {cure_dict.get(pred_m_label, 'Follow integrated pest management (IPM) practices.')}")
            if pred_m_label != pred_e_label:
                st.info(f"🩺 **{pred_e_label}:** {cure_dict.get(pred_e_label, 'Follow integrated pest management (IPM) practices.')}")

            # ---------------- Bright Accuracy Chart ---------------- #
            st.markdown("### 📈 Model Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=200)

            accuracies = [conf_m.item() * 100, conf_e.item() * 100]
            model_names = ["MobileNetV2", "EfficientNetB0"]

            # Brighter blue and green shades
            bars = ax.bar(model_names, accuracies, color=["#00c853", "#2979ff"], width=0.4)

            for bar, acc in zip(bars, accuracies):
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 2,
                    f"{acc:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=6,
                    fontweight='bold',
                    color="#000000"
                )

            ax.set_ylabel("Confidence (%)", fontsize=7)
            ax.set_ylim(0, 115)
            ax.set_title("Model Accuracy", fontsize=8, fontweight="bold", pad=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            plt.tight_layout(pad=0.6)

            st.pyplot(fig, use_container_width=False)
            plt.savefig("accuracy_bar.png", bbox_inches="tight", dpi=300)

            # ---------------- PDF Report ---------------- #
            st.markdown("### 📄 Generate & Download Report")
            if st.button("Generate PDF Report"):
                pdf_file = "Tomato_Leaf_Disease_Report.pdf"
                styles = getSampleStyleSheet()
                doc = SimpleDocTemplate(pdf_file, pagesize=A4)
                story = []

                story.append(Paragraph("<b>🍅 Tomato Leaf Disease Detection Report</b>", styles["Title"]))
                story.append(Spacer(1, 12))
                story.append(Paragraph(
                    f"<b>MobileNetV2:</b> {pred_m_label} ({conf_m.item()*100:.2f}%)<br/>"
                    f"<b>EfficientNetB0:</b> {pred_e_label} ({conf_e.item()*100:.2f}%)",
                    styles["Normal"]
                ))
                story.append(Spacer(1, 10))
                story.append(Paragraph("<b>🌿 Suggested Care:</b>", styles["Heading3"]))
                story.append(Paragraph(f"• {pred_m_label}: {cure_dict.get(pred_m_label)}", styles["Normal"]))
                if pred_m_label != pred_e_label:
                    story.append(Paragraph(f"• {pred_e_label}: {cure_dict.get(pred_e_label)}", styles["Normal"]))
                story.append(Spacer(1, 10))
                story.append(RLImage("accuracy_bar.png", width=160, height=100))
                story.append(Spacer(1, 12))
                story.append(Paragraph("📄 Report generated using Streamlit AI Platform.", styles["Italic"]))
                doc.build(story)

                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=f,
                        file_name=pdf_file,
                        mime="application/pdf"
                    )

# ==========================================================
# TAB 2: SOIL TYPE DETECTION & CROP RECOMMENDATION
# ==========================================================
with tab2:
    st.header("🌱 Soil Type Detection & Crop Recommendation")
    soil_file = st.file_uploader("📸 Upload a Soil Image", type=["jpg", "jpeg", "png"])

    if soil_file:
        img = Image.open(soil_file).convert("RGB")
        st.image(img, caption="Uploaded Soil Image", use_container_width=True)

        model_path = "soil_model.pth"

        if not os.path.exists(model_path):
            st.error("❌ Soil model not found. Please place 'soil_model.pth' in this directory.")
        else:
            soil_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            state_dict = torch.load(model_path, map_location=device)
            classifier_weight_key = [k for k in state_dict.keys() if "classifier.1.weight" in k][0]
            num_classes = state_dict[classifier_weight_key].shape[0]
            soil_model.classifier[1] = nn.Linear(soil_model.classifier[1].in_features, num_classes)
            soil_model.load_state_dict(state_dict, strict=False)
            soil_model = soil_model.to(device)
            soil_model.eval()

            default_classes = ["Clay Soil", "Loamy Soil", "Sandy Soil", "Black Soil", "Red Soil", "Laterite Soil"]
            class_names = default_classes[:num_classes]

            transform_soil = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            soil_input = transform_soil(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = soil_model(soil_input)
                pred_idx = torch.argmax(outputs, dim=1).item()

            soil_pred = class_names[pred_idx]
            st.success(f"🧱 Predicted Soil Type: **{soil_pred}**")

            crop_recommendations = {
                "Clay": ["Rice", "Wheat", "Sugarcane"],
                "Loamy": ["Maize", "Cotton", "Tomato"],
                "Sandy": ["Millet", "Groundnut", "Carrot"],
                "Black": ["Cotton", "Soybean", "Sunflower"],
                "Red": ["Groundnut", "Pulses", "Tobacco"],
                "Laterite": ["Tea", "Cashew", "Rubber"]
            }

            matched_key = next((k for k in crop_recommendations.keys() if k.lower() in soil_pred.lower()), None)
            recommended_crops = crop_recommendations.get(matched_key, ["N/A"])

            st.subheader("🌾 Recommended Crops:")
            for crop in recommended_crops:
                st.write(f"- ✅ {crop}")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("<center>© 2025 Crop Disease & Soil Analysis Platform | Powered by MobileNetV2 & EfficientNetB0</center>", unsafe_allow_html=True)
