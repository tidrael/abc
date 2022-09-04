import json
from utilities import get_model, transform_image

class ImageClassifier():
    def __init__(self, model_name):
        self.model = get_model(model_name)
        self.model.eval()
        self.class_labels = json.load(open(f"./model/{model_name}/label_names.json"))

    def get_prediction(self, image):
        tensor=transform_image(image=image)
        outputs=self.model.forward(tensor)
        _, y_hat=outputs.max(1)
        predicted_idx=str(y_hat.item())
        return self.class_labels[predicted_idx]

# with open("./test/0.png", "rb") as file:
#     image_bytes = file.read()
#     model = ImageClassifier("GucciMat")
#     prediction = model.get_prediction(image_bytes=image_bytes)
#     print(prediction)


