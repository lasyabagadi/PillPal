import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import os

# Load class labels
CLASS_NAMES = sorted(os.listdir("data/processed/c3pi224/train"))

# Define transforms (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load trained model
@st.cache_resource
def load_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/pill_classifier.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("💊 Pill Recognition App")
st.write("Upload a pill image to identify its class.")

uploaded_file = st.file_uploader("Choose a pill image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_class = torch.max(probs, dim=0)

    st.write(f"**Predicted:** {CLASS_NAMES[top_class]} ({top_prob*100:.2f}%)")
    st.bar_chart(probs.numpy())