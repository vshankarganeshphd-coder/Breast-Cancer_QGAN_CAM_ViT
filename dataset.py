from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset

def anisotropic_diffusion(image,iterations=10,kappa=30.0,gamma=0.15):
    x=image.astype(np.float32)/255.0
    for _ in range(iterations):
        n=np.zeros_like(x); s=np.zeros_like(x)
        e=np.zeros_like(x); w=np.zeros_like(x)
        n[1:]=x[:-1]-x[1:]; s[:-1]=x[1:]-x[:-1]
        w[:,1:]=x[:,:-1]-x[:,1:]; e[:,:-1]=x[:,1:]-x[:,:-1]
        cn=np.exp(-(n/kappa)**2); cs=np.exp(-(s/kappa)**2)
        cw=np.exp(-(w/kappa)**2); ce=np.exp(-(e/kappa)**2)
        x=np.clip(x+gamma*(cn*n+cs*s+cw*w+ce*e),0,1)
    return x

class BUSIUDataset(Dataset):
    """
    Expected manifest CSV columns:
    image,mask,label
    label is an integer class index.
    """
    def __init__(self,manifest,image_size=256,augment=False):
        self.rows=manifest
        self.image_size=image_size
        self.augment=augment

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        image=cv2.imread(str(row["image"]),cv2.IMREAD_GRAYSCALE)
        mask=cv2.imread(str(row["mask"]),cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image/mask at row {i}")
        image=cv2.resize(image,(self.image_size,self.image_size))
        mask=cv2.resize(mask,(self.image_size,self.image_size),
                        interpolation=cv2.INTER_NEAREST)
        image=anisotropic_diffusion(image)
        if self.augment and np.random.rand()<0.5:
            image=np.fliplr(image).copy()
            mask=np.fliplr(mask).copy()
        image=torch.tensor(image,dtype=torch.float32).unsqueeze(0)
        mask=torch.tensor((mask>127).astype("float32")).unsqueeze(0)
        label=torch.tensor(int(row["label"]),dtype=torch.long)
        return image,mask,label
