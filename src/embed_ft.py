"""Track A - contrastive fine-tune of DINOv2-small so that 'same source photo' scores above
'similar photo of the same product'. About 1-1.5 h on a T4."""

import random, io, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel


def jpeg(img):  # random re-compression, like a re-uploaded screenshot
    b = io.BytesIO()
    img.save(b, "JPEG", quality=random.randint(20, 85))
    b.seek(0)
    return Image.open(b).convert("RGB")


VIEW = T.Compose(
    [
        T.Lambda(jpeg),
        T.RandomResizedCrop(224, scale=(0.45, 1.0)),
        T.RandomRotation(15),
        T.ColorJitter(0.3, 0.3, 0.2),
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class Embedder(nn.Module):
    def __init__(self, name="facebook/dinov2-small", dim=128, train_last=4):
        super().__init__()
        self.bb = AutoModel.from_pretrained(name)
        for p in self.bb.parameters():
            p.requires_grad = False
        for blk in self.bb.encoder.layer[-train_last:]:
            for p in blk.parameters():
                p.requires_grad = True
        self.proj = nn.Linear(self.bb.config.hidden_size, dim)

    def forward(self, x):
        return F.normalize(self.proj(self.bb(pixel_values=x).last_hidden_state[:, 0]), dim=-1)


def supcon_loss(z, ids, temp=0.07):
    """z: (N,d) unit vectors; ids: (N,) source ids. Views of the same SOURCE are positives; every other image
    in the batch (incl. other photos of the same product = hard negatives) is a negative."""
    sim = z @ z.T / temp
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (ids[:, None] == ids[None, :]) & ~self_mask
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def make_batch(sources, n_products=24, per_product=2, views=2):
    """sources: dict product_id -> list of (source_id, PIL image).
    Put >=2 different sources of the SAME product in each batch."""
    xs, ids, n = [], [], 0
    for pid in random.sample(list(sources), min(n_products, len(sources))):
        for sid, img in random.sample(sources[pid], min(per_product, len(sources[pid]))):
            for _ in range(views):
                xs.append(VIEW(img))
                ids.append(n)  # integer id per source image within this batch (sid may be a string)
            n += 1
    return torch.stack(xs), torch.tensor(ids)


def train(sources, steps=3000, lr=3e-5, device="cuda", ckpt="embedder.pt", resume=False, n_products=24):
    """sources: dict product_id -> list of (source_id, PIL image). Saves ckpt every 100 steps."""
    import os

    m = Embedder().to(device)
    if resume and os.path.exists(ckpt):
        m.load_state_dict(torch.load(ckpt, map_location=device))
        print("resumed from", ckpt)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    for s in range(steps):
        x, ids = make_batch(sources, n_products=n_products)
        x, ids = x.to(device), ids.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            z = m(x)
        loss = supcon_loss(z.float(), ids)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if s % 100 == 0:
            print(s, round(loss.item(), 4))
            torch.save(m.state_dict(), ckpt)
    return m
