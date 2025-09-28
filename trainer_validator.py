import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from skimage.draw import polygon
from segmentation_models_pytorch import MAnet
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import get_stats, iou_score
import segmentation_models_pytorch as smp
import random

def retrieve_meta_data(path):
    MetaJson = json.load(open(path, "r"))
    class_titles = []
    class_ids = []
    for cls in MetaJson["classes"]:
        class_titles.append(cls["title"])
        class_ids.append(cls['id'])
    return class_titles, class_ids

class CarDataLoader(Dataset):
    def __init__(self, transforms, imgs_path, annotations_path, classes, sizes, files=None):
        self.imgs_path = imgs_path
        self.annotations_path = annotations_path
        self.classes = classes
        self.transforms = transforms
        self.sizes = sizes
        if files is not None:
            self.images = sorted(files)  # Use the specified files
        else:
            self.images = sorted(os.listdir(imgs_path))  # Original behavior
            
        self.annotations = [x + ".json" for x in self.images]

    @staticmethod
    def getMask(sizes, annfile, classes):
        img_height, img_width = annfile["size"]["height"], annfile["size"]["width"]
        mask = torch.zeros((img_height, img_width), dtype=torch.long)
        mask_numpy = mask.numpy()
        for object_ in annfile["objects"]:
            class_id = classes.index(object_["classId"])
            points = np.asarray(object_["points"]["exterior"])
            rr, cc = polygon(points[:, 1], points[:, 0], (img_height, img_width))
            mask_numpy[rr, cc] = class_id + 1
        
        mask_tensor = transforms.Resize(sizes, interpolation=transforms.InterpolationMode.NEAREST)(torch.from_numpy(mask_numpy).unsqueeze(0))
        
        # Create one-hot encoded mask
        num_classes = len(classes) + 1 # Add 1 for background
        one_hot_mask = torch.nn.functional.one_hot(mask_tensor.long(), num_classes=num_classes)
        one_hot_mask = one_hot_mask.permute(0, 3, 1, 2).squeeze(0).float()
        return one_hot_mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_path, self.images[idx])
        ann_path = os.path.join(self.annotations_path, self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        ann = json.load(open(ann_path, "r"))
        
        mask_tensor = CarDataLoader.getMask(self.sizes, ann, self.classes)
        img_tensor = self.transforms(img)
        
        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.images)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tp, fp, fn, tn = get_stats(outputs, masks.long(), mode="multilabel", threshold=0.5)
        iou = iou_score(tp, fp, fn, tn, reduction="micro")
        
        running_loss += loss.item()
        running_iou += iou.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    return epoch_loss, epoch_iou

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            tp, fp, fn, tn = get_stats(outputs, masks.long(), mode="multilabel", threshold=0.5)
            iou = iou_score(tp, fp, fn, tn, reduction="micro")
            
            running_loss += loss.item()
            running_iou += iou.item()
            
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    return epoch_loss, epoch_iou

class config:
    def __init__(self, img_dir, ann_dir, meta_path, sizes, batch_size, learning_rate, backbone, head_epochs, full_model_epochs, save_path):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.meta_path = meta_path
        self.sizes = sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.backbone = backbone
        self.head_epochs = head_epochs
        self.full_model_epochs = full_model_epochs
        self.save_path = save_path

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    class_titles, classes_ids = retrieve_meta_data(config.meta_path)
    num_classes = len(classes_ids) + 1 # +1 for the background
    
    train_transform = transforms.Compose([
    transforms.Resize(config.sizes),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# One without augmentations for validation
    val_transform = transforms.Compose([
    transforms.Resize(config.sizes),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
    
    # --- Dataset Splitting ---
    all_image_files = sorted(os.listdir(config.img_dir))
    random.shuffle(all_image_files)
    split_idx = int(0.8 * len(all_image_files))
    train_files = all_image_files[:split_idx]
    val_files = all_image_files[split_idx:]

    train_dataset = CarDataLoader(train_transform, config.img_dir, config.ann_dir, classes_ids, config.sizes, train_files)
    val_dataset = CarDataLoader(val_transform, config.img_dir, config.ann_dir, classes_ids, config.sizes, val_files)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # --- Model, Loss, Optimizer ---
    model = smp.MAnet(encoder_name=config.backbone, encoder_weights="imagenet", classes=num_classes)
    model.to(device)
    criterion = DiceLoss(mode="multilabel")
    
    print("--- Starting Stage 1: Training the Head ---")
    for param in model.encoder.parameters():
        param.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.learning_rate)
    
    for epoch in range(config.head_epochs):
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        print(f"Head Training Epoch {epoch+1}/{config.head_epochs} - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
    
    print("\n--- Starting Stage 2: Fine-Tuning Full Model ---")
    for param in model.encoder.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate / 10)
    best_iou = -1.0
    
    for epoch in range(config.full_model_epochs):
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        print(f"Full Training Epoch {epoch+1}/{config.full_model_epochs} - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config.save_path)
            print(f"  New best model saved with IoU: {best_iou:.4f}")

# Declaring folder paths for images and masks
training_config = config("archive/Car parts dataset/File1/img",
                         "archive/Car parts dataset/File1/ann",
                         "archive/Car parts dataset/meta.json",
                         (320, 320),
                         16,
                         1e-4,
                         'resnet50',
                         5,
                         20,
                         'MANet Model.pth'
                     )
main(training_config)

# Declaring folder paths for images and masks
training_config = config("archive/Car damages dataset/File1/img",
                         "archive/Car damages dataset/File1/ann",
                         "archive/Car damages dataset/meta.json",
                         (320, 320),
                         16,
                         1e-4,
                         'resnet50',
                         5,
                         20,
                         'MANet Model1.pth'
                     )
main(training_config)