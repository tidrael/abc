import requests

def predict_image(file):
    url = "https://demo-dense.herokuapp.com/"
    resp = requests.post(url=url, files={'file': file})
    return resp.json()['prediction']

# file = open('./static/images/lady-img.jpg', 'rb')
# response = predict_image(file)
# print(response)
