import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
import streamlit as st

from model import RESNET18
from model import transform_image
from model import transform_image_normalize


model_name = "dropout_full_resnet18_2.pth"
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'call', 'del', 'space', 'thumbsup']
dataset = []
image = None

model = torch.load(model_name, map_location=torch.device('cpu'))
model.eval()

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
        # image_np = np.array(image)
        # image_np = np.fliplr(image_np)
        # image = Image.fromarray(image_np)
        transform = transform_image()
        image = transform(image)
        dataset.append(image)
        # from torchvision import transforms
        # to_pil = transforms.ToPILImage()
        # pil_image = to_pil(image)
        # st.image(pil_image)
    else:
        for img in image:
            img = Image.open(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            transform = transform_image_normalize()
            img = transform(img)
            dataset.append(img)

    dataloader = DataLoader(dataset)

    for data in dataloader:
        predictions = model(data)
        probabilities = F.softmax(predictions, dim=1)
        k = 3
        prob, pred = torch.topk(probabilities, k=k, dim=1)
        st.write("Our model predicts:")
        for i in range(k):
            st.write("{} with {:.1f}%.".format(letters[pred[0][i].item()], prob[0][i].item()*100))