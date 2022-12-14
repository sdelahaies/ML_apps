# class_app_2.py

import streamlit as st
from PIL import Image
import io
import torch
from torchvision import models
import torchvision.transforms as transforms
from urllib.request import urlopen
import base64
import time
import style
from planets import *
import gc
from datetime import datetime
import psutil


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "jpg"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
                }}
         </style>
         """,
        unsafe_allow_html=True
    )


set_bg_hack('assets/background_light.jpg')


im = Image.open(r"assets/logo_1_k.png")
with st.sidebar:
    st.image(im, width=100)
    st.title("Machine Learning apps")
    st.text('memory load:')
    mem_info = st.empty()
    mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    mem_info.text('{:5.2f}'.format(mem))


model = False

tab1, tab2, tab3, tab4 = st.tabs(
    ["Classification", "Detection", "Style transfer", "Language processing"])

with tab1:
    if not model:
        model = models.densenet121(pretrained=True)
        model.eval()
    else:
        del model
        gc.collect()
        model = models.densenet121(pretrained=True)
        model.eval()
    class_labels_url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    class_labels = urlopen(class_labels_url).read().decode("utf-8").split("\n")

    mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    mem_info.text('{:5.2f}%'.format(mem))

    # Define the transofrmation of the input image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    image_file = st.file_uploader("Choose an image")
    if image_file is not None:
        # To read file as bytes:
        image_bytes = image_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, width=300)

    class_click = st.button('Classify')
    if class_click:
        image_tensor = transform(image).unsqueeze(0)
        # Pass the image through the model
        with torch.no_grad():
            output = model(image_tensor)
        # Select the class with the higherst probability
        class_id = torch.argmax(output).item()
        class_name = class_labels[class_id]
        st.title(class_name)
        st.balloons()

with tab2:
    st.experimental_memo.clear()
    if not model:
        model = init_model(im_size=144)
    else:
        del model
        gc.collect()
        model = init_model(im_size=144)
    mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    mem_info.text('{:5.2f}%'.format(mem))

    clicked_gen = st.button('Generate shot and detect')
    if clicked_gen:
        shot = gen_shot(data_generator())
        im0 = detect(shot, model)
        st.image(im0, width=250)

with tab3:

    # img = st.sidebar.selectbox(
    img = st.selectbox(
        'Select Image',
        ('amber.jpg', 'cat.png', 'agaric.jpg')
    )

  #  style_name = st.sidebar.selectbox(
    style_name = st.selectbox(
        'Select Style',
        ('candy', 'mosaic', 'rain_princess', 'udnie')
    )

    model_name = "saved_models/" + style_name + ".pth"
    input_image = "images/content-images/" + img
    output_image = "images/output-images/" + style_name + "-" + img

    st.write('### Source image:')
    image = Image.open(input_image)
    st.image(image, width=400)  # image: numpy array

    if not model:
        model = style.load_model(model_name)
        model.eval()
    else:
        del model
        gc.collect()
        model = style.load_model(model_name)
        model.eval()

    mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    mem_info.text('{:5.2f}%'.format(mem))

    clicked = st.button('Stylize')
    if clicked:
        style.stylize(model, input_image, output_image)

        st.write('### Output image:')
        image = Image.open(output_image)
        st.image(image, width=400)

with tab4:
    # the callback function for the button will add 1 to the
    # slider value up to 10
    def plus_one():
        if st.session_state["slider"] < 10:
            st.session_state.slider += 1
        else:
            pass
        return

    # when creating the button, assign the name of your callback
    # function to the on_click parameter
    add_one = st.button("Add one to the slider",
                        on_click=plus_one, key="add_one")

    # create the slider
    slide_val = st.slider("Pick a number", 0, 10, key="slider")
