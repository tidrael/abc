from torchvision import models
import torchvision.transforms as transforms
# import io
# from PIL import Image
import json
import torch
from torch import nn


def get_model(name: str):
    if name == "GucciMat":
        label_path = f"./model/{name}/label_names.json"
        with open (label_path, "rb") as f:
            label_names = json.load(f)
        num_class = len(label_names)
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
        in_fc = model.classifier.fc.in_features
        model.classifier.fc = nn.Linear(in_fc, num_class)
        # model = torch.load('./model/GucciMat/model.pt', map_location=torch.device('cpu'))
        # Load params
        model.load_state_dict(torch.load(f"./model/{name}/params", map_location=torch.device('cpu')))
        return model
    if name == "DenseNet121":
        return models.densenet121(pretrained=True)

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.5, 0.5, 0.5],
                                            [0.5, 0.5, 0.5])])
    # image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

        