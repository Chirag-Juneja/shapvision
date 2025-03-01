import torch
import torchvision
from torchvision import transforms
import argparse
import shap
from PIL import Image
import numpy as np


def preprocess_image(img, custom_transforms=None):
    if custom_transforms:
        transform = custom_transforms
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transform(img).unsqueeze(0)


def load_imagenet_labels():
    with open("./assets/imagenet_classes.txt") as f:
        return [line.strip() for line in f.readlines()]
    default_class_names = load_imagenet_labels()
    return default_class_names


def predict(img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
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

def _predict(img):
    img = nhwc_to_nchw(torch.Tensor(img))
    img = img.to(device)
    model.to(device)
    return model(img)

def get_explainer(model, class_names, input_shape=(3, 256, 256)):
    # Create a masker that is used to mask out parts of the input image
    masker = shap.maskers.Image("blur(64,64)", input_shape)
    # Create an explainer with the model and masker
    return shap.Explainer(
        _predict,
        masker,
        output_names=class_names,
    )

def generate_shap_values(explainer, img_tensor):
    # Generate SHAP values for the top predicted class
    topk=5
    batch_size=1024
    n_evals=100000

    shap_values = explainer(
        img_tensor,
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )
    return shap_values


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Image Classificaiton Explainer")
    parser.add_argument("--path", type=str, help="image path")
    args = parser.parse_args()

    device = "cuda"
    class_names = load_imagenet_labels()

    image = Image.open(args.path).convert("RGB")

    model = load_pretrained_model()

    img_tensor = preprocess_image(image)

    class_idx, probabilities = predict(img_tensor)

    top_k = min(5, len(probabilities))
    top_probs, top_classes = torch.topk(probabilities, top_k)

    for i, (prob, class_id) in enumerate(zip(top_probs, top_classes)):
        class_name = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"Class {class_id}"
        )
        print(f"{i+1}. {class_name}: {prob.item()*100:.2f}%")
        top_class_name = (
            class_names[top_classes[0]]
            if top_classes[0] < len(class_names)
            else f"Class {top_classes[0]}"
        )
    print(
        f"Top prediction: **{top_class_name}** with {top_probs[0].item()*100:.2f}% confidence"
    )
    input_shape = (128, 128,3)
    print(img_tensor.shape)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = [
        torchvision.transforms.Lambda(nhwc_to_nchw),
        torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Lambda(nchw_to_nhwc),
    ]

    inv_transform = [
        torchvision.transforms.Lambda(nhwc_to_nchw),
        torchvision.transforms.Normalize(
            mean=(-1 * np.array(mean) / np.array(std)).tolist(),
            std=(1 / np.array(std)).tolist(),
        ),
        torchvision.transforms.Lambda(nchw_to_nhwc),
    ]
    transform = torchvision.transforms.Compose(transform)
    inv_transform = torchvision.transforms.Compose(inv_transform)

    explainer = get_explainer(
        model, class_names, input_shape
    )
    shap_values = generate_shap_values(
        explainer, nchw_to_nhwc(img_tensor)
    )
    print((shap_values.data.shape, shap_values.values.shape))
    shap_values.data = inv_transform(shap_values.data).cpu().numpy()
    shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]
    print((shap_values.data.shape, shap_values.values[0].shape))
    shap.image_plot(
    shap_values=shap_values.values,
    pixel_values=shap_values.data,
    labels=shap_values.output_names,
    true_labels=[top_class_name],
    )
