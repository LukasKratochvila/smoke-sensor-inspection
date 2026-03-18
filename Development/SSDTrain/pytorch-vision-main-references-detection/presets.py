from collections import defaultdict

import torch
import transforms as reference_transforms
import albumentations as A
import numpy as np

def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors

        return torchvision.transforms.v2, torchvision.tv_tensors
    else:
        return reference_transforms, None


class DetectionPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter.
    def __init__(
        self,
        *,
        data_augmentation,
        hflip_prob=0.5,
        mean=(123.0, 117.0, 104.0),
        backend="pil",
        use_v2=False,
    ):

        self.aug=data_augmentation
        T, tv_tensors = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        if data_augmentation == "hflip":
            transforms += [T.RandomHorizontalFlip(p=hflip_prob)]
        elif data_augmentation == "lsj":
            transforms += [
                T.ScaleJitter(target_size=(1024, 1024), antialias=True),
                # TODO: FixedSizeCrop below doesn't work on tensors!
                reference_transforms.FixedSizeCrop(size=(1024, 1024), fill=mean),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "multiscale":
            transforms += [
                T.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssd":
            fill = defaultdict(lambda: mean, {tv_tensors.Mask: 0}) if use_v2 else list(mean)
            transforms += [
                T.Resize((300,300)),
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=fill),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssdlite":
            transforms += [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "alb":
            alb_transforms = [
                #A.Blur(p=0.2),
                #A.MedianBlur(p=0.2),
                #A.ToGray(p=0.2),
                #A.CLAHE(p=0.2),
                #A.RandomBrightnessContrast(0.2),
                #A.RandomGamma(p=0.2),
                #A.ImageCompression(quality_range=(75,100), p=0.2),
                A.Resize(height=300, width=300),
                
                A.PhotoMetricDistort(p=0.2),
                A.RandomScale(scale_limit=(-0.9,0)),
                A.BBoxSafeRandomCrop(p=0.2),
                A.HorizontalFlip(p=0.5),                
                
                A.AutoContrast(p=0.2),
                A.Illumination(p=0.2),
                A.MotionBlur(p=0.2),
                A.Defocus(p=0.2),
                A.ChromaticAberration(p=0.2),
                A.ISONoise(p=0.2),
                A.pytorch.transforms.ToTensorV2()
            ]
            self.transforms = A.Compose(alb_transforms, bbox_params=A.BboxParams("pascal_voc"#'coco', # Specify input format label_fields=['class_labels'] # Specify label argument name(s)
            ))
            return
        #else:
        #    raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2.
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [
                T.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
                T.SanitizeBoundingBoxes(),
                T.ToPureTensor(),
            ]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, tgr):
        if self.aug == "alb":
            b=tgr["boxes"].numpy()
            #print(b, img.size)
            #if b[0,0]+b[0,2]>img.size[0]:
            #    b[0,2] = img.size[0]-b[0,0]-2
            #if b[0,1]+b[0,3]>img.size[1]:
            #    b[0,3] = img.size[1]-b[0,1]-2
            #print(b)
            transformed = self.transforms(image=np.array(img), bboxes=b)
            #print(transformed["bboxes"].astype(np.uint8))
            image = transformed["image"]
            target = tgr
            target["boxes"] = torch.from_numpy(transformed["bboxes"])
            return reference_transforms.ToDtype(torch.float, scale=True)(image, target)
        return self.transforms(img, tgr)


class DetectionPresetEval:
    def __init__(self, backend="pil", use_v2=False):
        T, _ = get_modules(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]
        elif backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        else:
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        transforms += [T.ToDtype(torch.float, scale=True)]

        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
