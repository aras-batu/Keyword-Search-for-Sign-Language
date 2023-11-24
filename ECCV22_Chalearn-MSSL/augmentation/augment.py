import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

mean = [0.5, 0.5, 0.5]
std = [1.0, 1.0, 1.0]

train_transform = A.Compose(
    [
        # A.Resize(256, 256, always_apply=True, p=1.0),
        # A.CenterCrop(224, 224, always_apply=True, p=1.0),
        A.Rotate(limit=5, p=0.4),
        A.Resize(256, 256, always_apply=True, p=1.0),
        A.RandomCrop(224, 224, always_apply=True, p=1.0),
        A.HorizontalFlip(always_apply=False, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
        A.ToGray(p=0.2),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
            always_apply=True,
            p=1.0,
        ),
        ToTensorV2(p=1.0, always_apply=True),
    ],
)

valid_transform = A.Compose(
    [
        A.Resize(256, 256, always_apply=True, p=1.0),
        A.CenterCrop(224, 224, always_apply=True, p=1.0),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
            always_apply=True,
            p=1.0,
        ),
        ToTensorV2(p=1.0, always_apply=True),
    ],
)
