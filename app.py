import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
import streamlit as st

from model import RESNET18
from model import transform_image


model_name = "resnet18_model.pth"
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'call', 'del', 'space', 'thumbsup']
image = None

model = torch.load(model_name, map_location=torch.device('cpu'))
model.eval()

st.set_page_config(page_title="ASL", page_icon=":wave:", layout="wide")

def classify(dataset):
    dataloader = DataLoader(dataset)

    for data in dataloader:
        predictions = model(data)
        probabilities = F.softmax(predictions, dim=1)
        k = 3
        prob, pred = torch.topk(probabilities, k=k, dim=1)
        st.write("Top 3 Predictions (% probability): {} - {:.1f}%, {} - {:.1f}%, {} - {:.1f}%.".format(
            letters[pred[0][0].item()], prob[0][0].item()*100,
            letters[pred[0][1].item()], prob[0][1].item()*100,
            letters[pred[0][2].item()], prob[0][2].item()*100
        ))


def home():
    st.title("Home")
    st.header("Welcome to Team 10's ASL Alphabet Classifier!")
    st.write("Use the sidebar to nagivate.")

def instructions():
    st.title("Instructions")
    st.header("ASL Alphabet List")
    st.write("If you are not familiar with the ASL alphabet, here is a list of potential hand gestures.")
    st.image("asl_chart.jpg", caption="ASL Alphabet List", use_column_width=True)
    st.header("Use Camera")
    st.write("Press Take Photo and it will predict the hand gesture.")
    st.header("Upload Image")
    st.write("Upload an image and it will predict the hand gesture.")

def camera():
    dataset = []
    image = st.camera_input("Take a picture")
    if image:
        image = Image.open(image)
        transform = transform_image()
        image = transform(image)
        dataset.append(image)

        classify(dataset)

def upload():
    st.title("Upload Image")

    dataset = []
    image = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    for img in image:
        st.write("Name of file: {}".format(img.name))
        img = Image.open(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        transform = transform_image()
        img = transform(img)

        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(img)
        st.image(pil_image)

        dataset.append(img)
        classify(dataset)
        dataset = []

page_options = {
    "Home": home,
    "Use Camera": camera,
    "Upload Image": upload,
    "Instructions": instructions
}

selected_page = st.sidebar.radio("Navigation", list(page_options.keys()))

page_options[selected_page]()
