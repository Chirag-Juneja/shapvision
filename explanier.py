import torch
import torchvision
import argparse
import shap
from PIL import Image
import numpy as np
from utils import *


def load_imagenet_labels():
    with open("./assets/imagenet_classes.txt") as f:
        return [line.strip() for line in f.readlines()]


def predict(x):
    if isinstance(x, np.ndarray):
        x = nhwc_to_nchw(torch.Tensor(x)).to(device)
        with torch.no_grad():
            outputs = model(x)
        return outputs
    with torch.no_grad():
        x = x.to(device)
        outputs = model(x)
    _, predicted = outputs.max(1)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return predicted.item(), probabilities[0]


def load_pretrained_model(model_name="ResNet18"):
    if model_name == "ResNet18":
        model = torchvision.models.resnet18(weights="DEFAULT")
    elif model_name == "MobileNetV2":
        model = torchvision.models.mobilenet_v2(weights="DEFAULT")
    elif model_name == "VGG16":
        model = torchvision.models.vgg16(weights="DEFAULT")
    model.eval()
    return model


def get_explainer(model, class_names, input_shape=(128, 128, 3)):
    masker = shap.maskers.Image("blur(64,64)", input_shape)
    return shap.Explainer(
        predict,
        masker,
        output_names=class_names,
    )


def generate_shap_values(explainer, x):
    shap_values = explainer(
        x,
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )
    shap_values.data = postprocess(shap_values.data).cpu().numpy()
    shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]
    return shap_values


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image Classificaiton Explainer")
    parser.add_argument("--path", type=str, help="image path")
    parser.add_argument(
        "--device", type=str, help="device", nargs="?", const="cpu", default="cuda"
    )
    parser.add_argument(
        "--topk", type=int, nargs="?", help="top k results", const=5, default=5
    )
    parser.add_argument(
        "--batch", type=int, nargs="?", help="batch", const=512, default=512
    )
    parser.add_argument(
        "--nevals", type=int, nargs="?", help="no of evals", const=10000, default=10000
    )
    args = parser.parse_args()

    device = args.device
    topk = args.topk
    batch_size = args.batch
    n_evals = args.nevals

    input_shape = (128, 128, 3)

    class_names = load_imagenet_labels()

    image = Image.open(args.path).convert("RGB")

    model = load_pretrained_model().to(device)

    x = preprocess(image)

    class_idx, probabilities = predict(x)

    explainer = get_explainer(model, class_names, input_shape)
    shap_values = generate_shap_values(explainer, nchw_to_nhwc(x))

    shap.image_plot(
        shap_values=shap_values.values,
        pixel_values=shap_values.data,
        labels=shap_values.output_names,
        true_labels=[class_idx],
    )
