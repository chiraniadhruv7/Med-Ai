import io
import ssl
import base64
import certifi
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CXRClassifier:

    def __init__(self, use_pretrained: bool = True):
        self.model = None
        self.ready = False
        self._use_pretrained = use_pretrained
        self._load_model()

    def _load_model(self):
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if self._use_pretrained else None
        self.model = models.densenet121(weights=weights)
        num_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(num_features, len(PATHOLOGY_LABELS))

        if not self._use_pretrained:
            torch.nn.init.xavier_uniform_(self.model.classifier.weight)

        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.ready = True

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        tensor = IMAGE_TRANSFORM(image)
        return tensor.unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        if not self.ready:
            return {label: 0.0 for label in PATHOLOGY_LABELS}

        tensor = self.preprocess(image)
        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().tolist()

        results = {}
        for label, prob in zip(PATHOLOGY_LABELS, probabilities):
            results[label] = round(prob, 4)

        return results

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.predict(image)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_device(self) -> torch.device:
        return DEVICE
