
import streamlit as st
from PIL import Image
from predict import ImageClassifier

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Gucci Mat demo by Tidrael")
st.write("")

# Load model
model_name = "GucciMat"

# file_up = st.file_uploader("Upload an image", type="jpg")
uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # image_byte = uploaded_file.getvalue()
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.spinner("Loading model")
    model = ImageClassifier(model_name)
    st.spinner("Predict...")
    prediction = model.get_prediction(image=image)
    st.success(f"The result is: {prediction}")

    # print out the top 5 prediction labels with scores
    # for i in labels:
    #     st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])