# Tomato-plant-disease-dectection
This is a web app that detects plant diseases by analyzing images of plant leaves. It helps identify diseases early so they can be prevented and treated.

To run this app on your computer, download and install the following requirements.

For Soil Images Dataset: "/content/drive/MyDrive/Tomatodataset.zip"

For Tomato Leaf Dataset : "https://drive.google.com/file/d/1WIELZz9_86ENWg0TvzcLEoCay41XsG8U/view?usp=drive_link"

MobileNetV2_tomato_leaf_disease.pth, EfficientNetB0_tomato_leaf_disease.pth are Pre trained Models on Collab

Tomatopred.py is the Main Python File

$ requirements2.txt

Command to Install all Libraries : pip install "streamlit>=1.31.0" "torch>=2.2.0" "torchvision>=0.17.0" "Pillow>=10.2.0" "matplotlib>=3.8.0" "reportlab>=4.1.0"

🍅Main Command to Run ----$: streamlit run Tomatopred.py
