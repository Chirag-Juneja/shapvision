from torchvision import transforms


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
