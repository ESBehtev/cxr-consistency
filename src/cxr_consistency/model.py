import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm
from transformers import AutoModel


IMAGE_ENCODER_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "densenet121": 1024,
    "efficientnet_b0": 1280,
    "mobilenet_v3_small": 576,
    "convnext_tiny": 768,
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "swin_tiny": 768,
    "biomedclip_vit": 768,
}

TEXT_ENCODER_DIMS = {
    "simple": 256,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_image_encoder(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    name = name.lower()

    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        model.fc = nn.Identity()
        return model, 512

    if name == "resnet50":
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet50(weights=weights)
        model.fc = nn.Identity()
        return model, 2048

    if name == "densenet121":
        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = tv_models.densenet121(weights=weights)
        model.classifier = nn.Identity()
        return model, 1024

    if name == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        return model, 1280

    if name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        model.classifier = nn.Identity()
        return model, 576

    if name == "convnext_tiny":
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)
        model.classifier = nn.Identity()
        return model, 768

    if name == "vit_tiny":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        return model, 192

    if name == "vit_small":
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        return model, 384

    if name == "vit_base":
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        return model, 768

    if name == "swin_tiny":
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        return model, 768

    if name == "biomedclip_vit":
        model = timm.create_model(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        return model, 768

    raise ValueError(f"Unknown image encoder: {name}")


class SimpleTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_token_id,
        )

        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)

        _, hidden = self.encoder(embedded)

        return hidden[-1]


class TransformerTextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs.last_hidden_state[:, 0, :]


def build_text_encoder(
    name: str,
    tokenizer_vocab_size: int = 30522,
    pad_token_id: int = 0,
) -> tuple[nn.Module, int]:
    name = name.lower()

    if name == "simple":
        encoder = SimpleTextEncoder(
            vocab_size=tokenizer_vocab_size,
            embed_dim=256,
            hidden_dim=256,
            pad_token_id=pad_token_id,
        )
        return encoder, 256

    hf_names = {
        "distilbert": "distilbert-base-uncased",
        "bert": "bert-base-uncased",
        "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
        "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "cxrbert": "microsoft/BiomedVLP-CXR-BERT-specialized",
        "biomedclip_text": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    }

    model_name = hf_names.get(name, name)

    encoder = TransformerTextEncoder(model_name)

    return encoder, encoder.output_dim


class CXRConsistencyModel(nn.Module):
    def __init__(
        self,
        image_encoder_name: str = "mobilenet_v3_small",
        text_encoder_name: str = "simple",
        pretrained_image: bool = True,
        freeze_image_encoder: bool = False,
        freeze_text_encoder: bool = False,
        tokenizer_vocab_size: int = 30522,
        pad_token_id: int = 0,
        projection_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.image_encoder_name = image_encoder_name
        self.text_encoder_name = text_encoder_name

        self.image_encoder, image_dim = build_image_encoder(
            image_encoder_name,
            pretrained=pretrained_image,
        )

        self.text_encoder, text_dim = build_text_encoder(
            text_encoder_name,
            tokenizer_vocab_size=tokenizer_vocab_size,
            pad_token_id=pad_token_id,
        )

        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fusion_dim = projection_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(image)

        if features.ndim == 4:
            features = features.mean(dim=[2, 3])

        return self.image_projection(features)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self.text_projection(features)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_emb = self.encode_image(image)
        text_emb = self.encode_text(input_ids, attention_mask)

        fused = torch.cat(
            [
                image_emb,
                text_emb,
                torch.abs(image_emb - text_emb),
                image_emb * text_emb,
            ],
            dim=1,
        )

        logits = self.classifier(fused).squeeze(1)

        return logits