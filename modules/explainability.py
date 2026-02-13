import io
import copy
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAMExplainer:

    def __init__(self):
        self._activations = None
        self._gradients = None

    def generate(
        self,
        model: torch.nn.Module,
        image: Image.Image,
        input_tensor: torch.Tensor,
        target_class: int = 0,
    ) -> str:
        cam_model = copy.deepcopy(model)
        cam_model.eval()

        for name, module in cam_model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

        target_layer = cam_model.features[-1]

        original_forward = cam_model.forward

        def patched_forward(x):
            features = cam_model.features(x)
            out = F.relu(features, inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = cam_model.classifier(out)
            return out

        cam_model.forward = patched_forward

        hook_a = target_layer.register_forward_hook(self._save_activation)
        hook_g = target_layer.register_full_backward_hook(self._save_gradient)

        output = cam_model(input_tensor)
        cam_model.zero_grad()

        target_score = output[0, target_class]
        target_score.backward()

        hook_a.remove()
        hook_g.remove()

        gradients = self._gradients[0].cpu()
        activations = self._activations[0].cpu()

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.detach().numpy()
        cam_resized = cv2.resize(cam_np, (image.width, image.height))

        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        original_np = np.array(image.convert("RGB"))
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

        overlay_image = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_image.save(buffer, format="PNG")
        buffer.seek(0)
        b64_string = base64.b64encode(buffer.read()).decode("utf-8")

        return b64_string

    def _save_activation(self, module, input, output):
        self._activations = output.clone().detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].clone().detach()

    def generate_for_top_pathology(
        self,
        model: torch.nn.Module,
        image: Image.Image,
        input_tensor: torch.Tensor,
        predictions: dict,
    ) -> str:
        if not predictions:
            return ""

        top_label = max(predictions, key=predictions.get)
        pathology_labels = list(predictions.keys())
        target_idx = pathology_labels.index(top_label)

        return self.generate(model, image, input_tensor, target_class=target_idx)
