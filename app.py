import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import streamlit as st
import PIL.Image as Image
import pandas as pd



@st.cache
def get_model(path):
    model = torchvision.models.resnext101_32x8d(pretrained=True)
    saved_model_parameters = torch.load(path, map_location=torch.device('cpu'))

    num_features = model.fc.in_features
    num_classes = 2

#    Replacing model.fc layer
    model.fc = torch.nn.Sequential(
           nn.Linear(num_features, 1024),
           nn.Linear(1024, num_classes)
    )

    model.load_state_dict(saved_model_parameters)

    # Placing model in evaluation mode
    model.eval()
    return model
@st.cache
def get_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            labels.append(line.replace('\n',''))
    return labels
def classify_image(img, labels):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
    probablilities = (torch.nn.functional.softmax(output, dim = 1).flatten()) * 100
    return {labels[i]:[probablilities[i].item()] for i in range(len(labels))}
model_path = "C:\\Users\\zahin\\OneDrive\\Desktop\\GitHub Repositories\\pneumonia_classification\\Model\\chest_xray_scan_model.pt"
model = get_model(model_path)
labels_path = "C:\\Users\\zahin\\OneDrive\\Desktop\\GitHub Repositories\\pneumonia_classification\\scan_labels.txt"
labels = get_labels(labels_path)
st.title('Pneumonia Classification')
st.write("This program allows users to upload images of lung scans and determine if the person whose lungs were scanned has pneumonia or not.")
st.write("Here is a link to my code for those who want to see it: " + "https://github.com/nzahin04/pneumonia_classification")

uploaded_file = st.file_uploader('Upload file')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded scan")
    probabilities = classify_image(image, labels)
    df = pd.DataFrame.from_dict(probabilities)
    st.bar_chart(df)

