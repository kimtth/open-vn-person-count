import PIL.Image as Image
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import os
import torch
from torch_Model import CSRNet


def get_model_a():
    model = CSRNet()
    checkpoint = torch.load('.\\model\\PartAmodel_best.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_model_b():
    model = CSRNet()
    checkpoint = torch.load('.\\model\\PartBmodel_best.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


# Access commons
model = get_model_a()
# Standard RGB transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])


def get_prediction(file):
    img = transform(Image.open(file).convert('RGB'))
    img = img.cpu()
    output = model(img.unsqueeze(0))
    prediction = int(output.detach().cpu().sum().numpy())
    x = random.randint(1, 100000)
    density = '.\\csr_rtn\\density_map' + str(x) + '.jpg'
    plt.imsave(density, output.detach().cpu().numpy()[0][0])
    return prediction, density


def torch_to_onnx(type):
    # Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
    type_lower = str(type).lower()
    if type_lower == 'b':
        model = get_model_b()
        # Evaluate the model to switch some operations from training mode to inference.
        model.eval()
        # Create dummy input for the model. It will be used to run the model inside export function.
        dummy_input = torch.randn(1, 3, 224, 224)
        # Call the export function
        torch.onnx.export(model, (dummy_input,), '.\\model_b.onnx')
    elif type_lower == 'a':
        model = get_model_a()
        # Evaluate the model to switch some operations from training mode to inference.
        model.eval()
        # Create dummy input for the model. It will be used to run the model inside export function.
        dummy_input = torch.randn(1, 3, 224, 224)
        # Call the export function
        torch.onnx.export(model, (dummy_input,), '.\\model_a.onnx')


if __name__ == '__main__':
    torch_to_onnx('a')
    torch_to_onnx('b')

    '''
    files_dir = '.\\csr_rtn\\images_a\\'
    files_rtn_dir = '.\\csr_rtn\\'
    files_in_dir = os.listdir(files_dir)
    filtered_files = [file for file in files_in_dir if file.endswith(".jpg") or file.endswith(".jpeg")]
    for file in filtered_files:
        path = os.path.join(files_dir, file)
        # file_obj = open(path)
        file_path = os.path.abspath(path)
        prediction, density = get_prediction(file_path)
        print(path, prediction, density)
    '''
