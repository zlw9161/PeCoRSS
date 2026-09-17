import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(Module):
    def __init__(self, img_size=(128, 1024), patch_size=16, in_chans=2, n_classes=2, embed_dim=128, depth=6, num_heads=8, mlp_ratio=4., emb_dropout=0.2, dropout=0.2):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.n_classes = n_classes

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.grid_h = image_height // patch_height  # 128//16=8
        self.grid_w = image_width // patch_width    # 1024//16=64

        num_patches = self.grid_h * self.grid_w
        patch_dim = in_chans * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # num_cls_tokens  = 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)) 

        self.dropout = nn.Dropout(emb_dropout)

        dim_head = embed_dim // num_heads  
        mlp_dim = int(embed_dim * mlp_ratio)  
        self.transformer = Transformer(embed_dim, depth, num_heads, dim_head, mlp_dim, dropout)

        self.seg_head = nn.Sequential(
            Rearrange('b (h w) d -> b d h w', h=8, w=64, d=128),
            # 8×64 → 16×128
            nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 16×128 → 32×256
            nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 32×256 → 128×1024（
            nn.Upsample(scale_factor=(4,4), mode='bilinear', align_corners=False),
            nn.Conv2d(32, self.n_classes, 1, padding=0), 
        )



    def forward(self, img):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:, :seq, :]
        x = self.dropout(x)

        x = self.transformer(x)

        # # classification task
        # x = x[:, 0]
        # x = self.to_latent(x)
        # x = self.mlp_head(x)

        # segmentation task
        x = x[:, 1:, :]  # drop CLS, shape (batch, num_patches, embed_dim)
        x = self.seg_head(x)

        return x
    
if __name__ == "__main__":
    model = ViT()
    input_img = torch.randn(2, 2, 128, 1024)  
    print("input shape:", input_img.shape)  # output: torch.Size([2, 2, 128, 1024])
    output = model(input_img)
    print("output shape:", output.shape)  # output: torch.Size([2, 2, 128, 1024])
