import torch
import torchvision
import argparse
import shap
from PIL import Image
import numpy as np
from utils import *
import globals as gl


def load_imagenet_labels():
    with open("./assets/imagenet_classes.txt") as f:
        return [line.strip() for line in f.readlines()]


def predict(x):
    device = gl.device
    if isinstance(x, np.ndarray):
        x = nhwc_to_nchw(torch.Tensor(x)).to(device)
        with torch.no_grad():
            outputs = gl.model(x)
        return outputs
    with torch.no_grad():
        x = x.to(device)
        outputs = gl.model(x)
    _, predicted = outputs.max(1)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return predicted.item(), probabilities[0]


def load_custom_model(model_file, state_dict=None):
    try:
        model = torch.load(model_file,weights_only=False, map_location=torch.device('cpu'))
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def load_pretrained_model(model_name="ResNet18"):
    if model_name == "ResNet18":
        model = torchvision.models.resnet18(weights="DEFAULT")
    elif model_name == "MobileNetV3":
        model = torchvision.models.mobilenet_v2(weights="DEFAULT")
    elif model_name == "VGG16":
        model = torchvision.models.vgg16(weights="DEFAULT")
    model.eval()
    return model


def generate_shap_values(image, input_shape, class_names):

    x = preprocess(image, input_shape[:2])

    masker = shap.maskers.Image("blur(64,64)", input_shape)

    explainer = shap.Explainer(
        predict,
        masker,
        output_names=class_names,
    )

    shap_values = explainer(
        nchw_to_nhwc(x),
        max_evals=gl.n_evals,
        batch_size=gl.batch_size,
        outputs=shap.Explanation.argsort.flip[: gl.topk],
    )

    shap_values.data = postprocess(shap_values.data).cpu().numpy()
    shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]
    return shap_values


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image Classificaiton Explainer")
    parser.add_argument("--path", type=str, help="image path")
    parser.add_argument(
        "--device", type=str, help="device", nargs="?", const="cpu", default="mps"
    )
    parser.add_argument(
        "--topk", type=int, nargs="?", help="top k results", const=5, default=3
    )
    parser.add_argument(
        "--batch", type=int, nargs="?", help="batch", const=512, default=1024
    )
    parser.add_argument(
        "--nevals", type=int, nargs="?", help="no of evals", const=10000, default=10000
    )
    args = parser.parse_args()

    gl.device = args.device
    gl.topk = args.topk
    gl.batch_size = args.batch
    gl.n_evals = args.nevals

    input_shape = (128, 128, 3)

    class_names = load_imagenet_labels()

    image = Image.open(args.path).convert("RGB")

    gl.model = load_pretrained_model().to(gl.device)
    x = preprocess(image, input_shape[:2])
    class_idx, _ = predict(x)

    shap_values = generate_shap_values(image, input_shape, class_names)

    shap.image_plot(
        shap_values=shap_values.values,
        pixel_values=shap_values.data,
        labels=shap_values.output_names,
        true_labels=[class_names[class_idx]],
    )
