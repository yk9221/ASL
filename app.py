import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2
from PIL import Image
import streamlit as st

from model import RESNET50
from model import resize_and_pad_image
from model import transform_image

model_name = "full_resnet50.pth"
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'call', 'del', 'space', 'thumbsup']
dataset = []
image = None

model = torch.load(model_name, map_location=torch.device('cpu'))
model.eval()

transform = transform_image()

st.set_page_config(page_title="ASL", page_icon=":wave:", layout="wide")
camera_on = st.sidebar.checkbox("Use Camera", False)
upload_on = st.sidebar.checkbox("Upload Image", True)

if camera_on:
    image = st.camera_input("Take a picture")

elif upload_on:
    image = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if image:
    if camera_on:
        image = Image.open(image)
        image = transform(image)
        dataset.append(image)
    else:
        for img in image:
            img = Image.open(img)
            img = transform(img)
            dataset.append(img)

    dataloader = DataLoader(dataset)

    for data in dataloader:
        predictions = model(data)
        probabilities = F.softmax(predictions, dim=1)
        prob, pred = torch.max(F.softmax(predictions, dim=1), 1)

        st.write("{} with {:.1f}%.".format(letters[pred.item()], prob.item()*100))