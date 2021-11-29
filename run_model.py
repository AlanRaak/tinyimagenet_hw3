import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
from PIL import Image

### FILL THOSE
path_to_project_folder = "C:/Users/user/Desktop/Loengud/Neural_Networks/HW3/tinyimagenet_hw3"
your_model_name = "best_model_10_val_accuracy=0.5567.pt"
### FILL THOSE

states_dict = torch.load(f"{path_to_project_folder}/trained_models/{your_model_name}", map_location=torch.device('cpu'))

# Create correct loader
loader = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])

def image_loader(image):
    """load image, returns cuda tensor"""
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image

def create_dict_for_val(filename):
    d = {}
    with open(filename) as f:
        for line in f:
            (key, val, _, _, _, _) = line.split('\t')
            d[key] = val
    return d

def create_dict_for_words(filename):
    d = {}
    with open(filename) as f:
        for line in f:
            (key, val) = line.split('\t')
            d[key] = val
    return d

with torch.no_grad():

    # Loading the saved model
    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=200)
    model.load_state_dict(states_dict)
    model.eval()

    validation_ = create_dict_for_val(f"{path_to_project_folder}/val_annotations.txt")
    words_ = create_dict_for_words(f"{path_to_project_folder}/words.txt")
    
    # Generate prediction
    enter_file_name = ""
    while enter_file_name != ".JPEG":
        
        print("Enter filename (ex. val_5606)")
        enter_file_name = f"{input()}.JPEG"
        # Show result
        try:
            raw_image = Image.open(f"{path_to_project_folder}/test_images/{enter_file_name}")
            image = image_loader(raw_image)
            acthual = words_[validation_[enter_file_name]]

            # Disable grad

            prediction = model(image)

            # Predicted class value using argmax
            predicted_class = np.argmax(prediction)
        except:
            print("Error; perhaps you entered a wrong file name")
            continue

        plt.imshow(raw_image)
        plt.title(f'Prediction: {predicted_class} - Actual: {acthual}')
        plt.show()

model.eval()