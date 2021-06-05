import cv2
import numpy as np
# Import OpenVINO Inference Engine
from openvino.inference_engine import IECore
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])


def run_nn(model_xml_path, file_path):
    ie = IECore()
    net = ie.read_network(model=model_xml_path)
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # # Get Output Layer Information
    # print("Output Layer: ", out_blob)
    # # Get Input Shape of Model
    print("Input Shape: ", net.input_info[input_blob].input_data.shape)
    # # Get Output Shape of Model
    print("Output Shape: ", net.outputs[out_blob].shape)
    # # Load IECore Object
    # print("Available Devices: ", ie.available_devices)
    # # ('Starting inference in synchronous mode')
    # print('Starting inference in synchronous mode')
    exec_net = ie.load_network(network=net, device_name=ie.available_devices[0])

    # Following sample Without Image Reshape will make a broadcast error.
    # rgb = Image.open(file_path).convert('RGB')
    # img = transform(rgb)
    # torch.reshape(img, (1, 3, 224, 224))
    # output = exec_net.infer(inputs={input_blob: img.unsqueeze(0)})

    # Reshape Input Size
    # https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/ie_bridges/python/sample/object_detection_sample_ssd/object_detection_sample_ssd.py
    original_image = cv2.imread(file_path)
    image = original_image.copy()
    _, _, net_h, net_w = net.input_info[input_blob].input_data.shape
    if image.shape[:-1] != (net_h, net_w):
        image = cv2.resize(image, (net_w, net_h))

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    output = exec_net.infer(inputs={input_blob: image})
    prediction_arr = next(iter(output.values()))
    prediction = int(np.sum(prediction_arr, dtype=np.float32))
    print('Approx. Num of peoples: ', prediction)


if __name__ == '__main__':
    file_path = 'C:\\Users\\Kim&Suzuki\\Desktop\\open-vn-person-count\\csr_rtn\\images_a\\IMG_100.jpg'

    model_xml_path = '.\\model\\csrnet-crowd-counting-caffe\\shanghaia.xml'
    run_nn(model_xml_path, file_path)
    # Not sure why it makes diffrent result with same model,
    # this model is converted through processes from pytorch to onnx then openvino.
    # model_xml_path = '.\\model\\csrnet-crowd-counting-pytorch\\model_a.xml'
    # run_nn(model_xml_path, file_path)
