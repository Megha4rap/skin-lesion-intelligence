"""
multimodal_lib.py
Shared code for: Multimodal Deep Learning for Early Skin Cancer Detection (HAM10000).
Used by BOTH the Colab training notebook and the Streamlit app.

Image backbones : ResNet50, EfficientNet-B0
Text branch     : DistilBERT on a clinical note built from patient metadata
Fusion          : (a) simple concat baseline  (b) cross-modal attention + gated fusion
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import DistilBertModel, DistilBertConfig

# ----------------------------------------------------------------------------
# Constants (HAM10000)
# ----------------------------------------------------------------------------
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = {
    'akiec': "Actinic keratoses / Bowen's disease",
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesion',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevus (mole)',
    'vasc': 'Vascular lesion',
}
RISK = {
    'akiec': 'Pre-cancerous', 'bcc': 'Malignant', 'mel': 'Malignant',
    'bkl': 'Benign', 'df': 'Benign', 'nv': 'Benign', 'vasc': 'Benign',
}
LOCALIZATIONS = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital',
                 'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'unknown',
                 'upper extremity']
SEXES = ['male', 'female', 'unknown']
BACKBONES = ['resnet50', 'efficientnet_b0']
IMG_SIZE = 224
MAX_LEN = 32
N_TOKENS = (IMG_SIZE // 32) ** 2          # 7x7 = 49 spatial tokens for both backbones
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Metadata -> clinical note text. (`dx_type` is deliberately NOT used: label leakage.)
# ----------------------------------------------------------------------------
def build_text(age, sex, localization):
    if age is None or (isinstance(age, float) and math.isnan(age)):
        age_txt = 'age unknown'
    else:
        age_txt = f'{int(age)} year old'
    who = 'patient' if sex in (None, 'unknown') else sex
    site = 'an unknown body site' if localization in (None, 'unknown') else f'the {localization}'
    return f'{age_txt} {who} with a skin lesion located on {site}.'


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(), T.Normalize(MEAN, STD),
])


# ----------------------------------------------------------------------------
# Dataset -> (image, token ids, attention mask, label)
# ----------------------------------------------------------------------------
class SkinDataset(Dataset):
    def __init__(self, df, tokenizer, tf, use_img=True, max_len=MAX_LEN):
        self.paths = df['image_path'].tolist()
        self.labels = df['label'].values
        self.tf, self.use_img = tf, use_img
        enc = tokenizer(df['text'].tolist(), padding='max_length', truncation=True,
                        max_length=max_len, return_tensors='pt')
        self.ids, self.mask = enc['input_ids'], enc['attention_mask']

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.use_img:
            img = self.tf(Image.open(self.paths[i]).convert('RGB'))
        else:
            img = torch.zeros(1)  # placeholder for text-only models
        return img, self.ids[i], self.mask[i], int(self.labels[i])


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    """ResNet50 or EfficientNet-B0 returning the last spatial feature map (B,C,7,7)."""
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        self.backbone = backbone
        if backbone == 'resnet50':
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.features = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                          m.layer1, m.layer2, m.layer3, m.layer4)
            self.out_dim = 2048
        elif backbone == 'efficientnet_b0':
            m = tvm.efficientnet_b0(
                weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = m.features
            self.out_dim = 1280
        else:
            raise ValueError(backbone)

    @property
    def cam_layer(self):  # layer used by Grad-CAM
        return self.features[-1][-1] if self.backbone == 'resnet50' else self.features[-1]

    def forward(self, x):
        return self.features(x)


def _bert(pretrained):
    if pretrained:
        return DistilBertModel.from_pretrained('distilbert-base-uncased')
    return DistilBertModel(DistilBertConfig())


def _groups(model, lr, enc_mult=1.0, bert_mult=0.3, head_mult=5.0):
    """Discriminative learning rates: encoder / BERT / new layers."""
    enc = list(model.enc.parameters())
    bert = list(model.bert.parameters()) if hasattr(model, 'bert') else []
    used = {id(p) for p in enc + bert}
    rest = [p for p in model.parameters() if id(p) not in used]
    g = [{'params': enc, 'lr': lr * enc_mult}, {'params': rest, 'lr': lr * head_mult}]
    if bert:
        g.append({'params': bert, 'lr': lr * bert_mult})
    return g


# ----------------------------------------------------------------------------
# Models. All share forward(img, ids, mask) so ONE training loop fits all.
# ----------------------------------------------------------------------------
def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.MaxPool2d(2))


class CustomCNN(nn.Module):
    """Baseline: CNN from scratch (image only)."""
    def __init__(self, n_classes=7):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128),
                                      _block(128, 256), _block(256, 512))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.4), nn.Linear(512, n_classes))

    def forward(self, img, ids=None, mask=None):
        return self.head(self.features(img))

    def param_groups(self, lr):
        return [{'params': self.parameters(), 'lr': lr}]


class ImageClassifier(nn.Module):
    """Baseline: pretrained ResNet50 / EfficientNet-B0 (image only)."""
    def __init__(self, backbone='resnet50', n_classes=7, pretrained=True):
        super().__init__()
        self.enc = ImageEncoder(backbone, pretrained)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(self.enc.out_dim, n_classes))

    def forward(self, img, ids=None, mask=None):
        return self.head(self.enc(img).mean((2, 3)))

    def param_groups(self, lr):
        return _groups(self, lr)


class TextClassifier(nn.Module):
    """Baseline: DistilBERT text classification on the clinical note (metadata only)."""
    def __init__(self, n_classes=7, pretrained=True):
        super().__init__()
        self.bert = _bert(pretrained)
        self.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, n_classes))

    def forward(self, img, ids, mask):
        return self.head(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

    def param_groups(self, lr):
        return [{'params': self.bert.parameters(), 'lr': lr},
                {'params': self.head.parameters(), 'lr': lr * 10}]


class ConcatFusion(nn.Module):
    """Ablation baseline: pooled image vector || DistilBERT [CLS]  ->  MLP."""
    def __init__(self, backbone='resnet50', n_classes=7, pretrained=True):
        super().__init__()
        self.enc = ImageEncoder(backbone, pretrained)
        self.bert = _bert(pretrained)
        self.img_proj = nn.Sequential(nn.Linear(self.enc.out_dim, 512), nn.ReLU(), nn.Dropout(0.3))
        self.txt_proj = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                        nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, img, ids, mask):
        fi = self.img_proj(self.enc(img).mean((2, 3)))
        ft = self.txt_proj(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return self.classifier(torch.cat([fi, ft], 1))

    def param_groups(self, lr):
        return _groups(self, lr)


class CrossModalFusion(nn.Module):
    """
    PROPOSED MODEL (use with backbone='resnet50' or 'efficientnet_b0'):

      image -> CNN feature map (7x7) -> 49 region tokens (+ positional emb)
      text  -> DistilBERT token states -> L word tokens
      text tokens  --attend-->  image regions   (which lesion regions matter for this patient?)
      image regions --attend--> text tokens     (which clinical facts matter for this region?)
      learned sigmoid GATE decides, per sample, how much to trust image vs metadata
      fused = gate*img + (1-gate)*txt  ||  img*txt (interaction)  ||  global CNN vector
      -> MLP -> 7 classes
    """
    def __init__(self, backbone='resnet50', n_classes=7, pretrained=True, d=256, heads=4):
        super().__init__()
        self.enc = ImageEncoder(backbone, pretrained)
        self.bert = _bert(pretrained)
        self.img_tok = nn.Conv2d(self.enc.out_dim, d, 1)
        self.pos = nn.Parameter(torch.zeros(1, N_TOKENS, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.txt_tok = nn.Linear(768, d)
        self.attn_t2i = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.attn_i2t = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.ln_t, self.ln_i = nn.LayerNorm(d), nn.LayerNorm(d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.glob = nn.Sequential(nn.Linear(self.enc.out_dim, d), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(3 * d, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                        nn.Dropout(0.4), nn.Linear(256, n_classes))
        self.last_gate = None  # mean image-weight of the gate (for interpretability)

    def forward(self, img, ids, mask):
        fmap = self.enc(img)                                          # B,C,h,w
        g_img = self.glob(fmap.mean((2, 3)))                          # B,d
        I = self.img_tok(fmap).flatten(2).transpose(1, 2) + self.pos  # B,49,d
        Tt = self.txt_tok(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state)  # B,L,d
        pad = mask == 0
        t2i, _ = self.attn_t2i(Tt, I, I)
        t2i = self.ln_t(Tt + t2i)
        m = mask.unsqueeze(-1).to(t2i.dtype)
        t_vec = (t2i * m).sum(1) / m.sum(1).clamp(min=1)
        i2t, _ = self.attn_i2t(I, Tt, Tt, key_padding_mask=pad)
        i_vec = self.ln_i(I + i2t).mean(1)
        g = self.gate(torch.cat([i_vec, t_vec], 1))
        self.last_gate = g.detach().float().mean(1)                   # (B,) weight on image
        z = torch.cat([g * i_vec + (1 - g) * t_vec, i_vec * t_vec, g_img], 1)
        return self.classifier(z)

    def param_groups(self, lr):
        return _groups(self, lr)


def load_fusion(backbone, path, device):
    """Used by the Streamlit app."""
    m = CrossModalFusion(backbone, pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


# ----------------------------------------------------------------------------
# Explainability: Grad-CAM on the CNN's last conv block
# ----------------------------------------------------------------------------
def grad_cam(model, img, ids, mask, target_layer, cls=None):
    """img: (1,3,H,W). Returns (cam in [0,1] HxW numpy, class index used)."""
    model.eval()
    store = {}

    def fwd_hook(module, inp, out):
        store['a'] = out
        out.register_hook(lambda g: store.__setitem__('g', g))

    h = target_layer.register_forward_hook(fwd_hook)
    with torch.enable_grad():
        model.zero_grad()
        logits = model(img, ids, mask)
        if cls is None:
            cls = int(logits.argmax(1).item())
        logits[0, cls].backward()
    h.remove()
    w = store['g'].mean((2, 3), keepdim=True)
    cam = torch.relu((w * store['a']).sum(1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=img.shape[-2:], mode='bilinear',
                                    align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), cls


def overlay(pil_img, cam, alpha=0.4):
    """Blend heat-map on the resized image. Returns float array in [0,1]."""
    import matplotlib
    heat = matplotlib.colormaps['jet'](cam)[..., :3]
    base = np.asarray(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))) / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)
