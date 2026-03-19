import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)   # [B, C, H, W] → [B, embed_dim, 8, 8]

        x = x.flatten(2)   # [B, embed_dim, 64]

        x = x.transpose(1, 2)  # [B, 64, embed_dim]

        return x

class ViT(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.patch_embed = PatchEmbedding()

        self.cls_token = nn.Parameter(torch.randn(1,1,128))
        self.pos_embed = nn.Parameter(torch.randn(1,65,128))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first = True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )

        self.mlp_head = nn.Linear(128, num_classes)


    def forward(self, x):

        x = self.patch_embed(x)

        B = x.size(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        x = self.transformer(x)

        cls_output = x[:,0]

        out = self.mlp_head(cls_output)

        return out
