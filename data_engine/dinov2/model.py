import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from .backbones import dinov2_vitl14_reg


class DINOv2(nn.Module):
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.model = dinov2_vitl14_reg()
        self._load_model(model_path)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device

    def _load_model(self, model_path: str):
        state_dict = torch.load(model_path, weights_only=False, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def forward(self, image: torch.tensor):
        features = self.model.forward_features(image.to(self.device))
        # print(f"features keys: {features.keys()}")
        # print(f"features x_prenorm shape: {features['x_prenorm'].shape}, num_register_tokens: {self.model.num_register_tokens}")
        return features["x_prenorm"][:, self.model.num_register_tokens + 1 :]

    def to(self, device: str):
        self.model.to(device)
        self.device = device


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import minmax_scale
    from torchvision import transforms
    model = DINOv2(
        "/path/to/dinov2/dinov2_vitl14_reg4_pretrain.pth"
    )
    model.to("cuda")
    img = Image.open(
        "./test_data/0.png"
    )
    # DINO preprocess
    DINO_IMG_SIZE = 518
    DINO_DOWNSAMPLE_RATE = 14
    DINO_PATCH_SIZE = DINO_IMG_SIZE // DINO_DOWNSAMPLE_RATE
    transform_totensor = transforms.Compose(
        [
            transforms.Resize((DINO_IMG_SIZE, DINO_IMG_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features = model(transform_totensor(torch.tensor(np.array(img) / 255.0).permute(2, 0, 1)).unsqueeze(0).float())
    b, n, c = features.shape
    dw, dh = DINO_PATCH_SIZE, DINO_PATCH_SIZE
    fg_pca = PCA(n_components=3)
    fg_pca_images = fg_pca.fit_transform(features.squeeze(0).cpu().numpy())
    fg_pca_images = minmax_scale(fg_pca_images)
    fg_pca_images = fg_pca_images.reshape(dh, dw, 3)
    fg_pca_images = Image.fromarray((fg_pca_images * 255).astype("uint8"))
    fg_pca_images.save("fg_pca_images_dinov2.png")
