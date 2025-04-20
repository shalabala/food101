import os
from PIL import Image
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, root_dir, classes, labels, skip, take, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = labels
        self.classes = classes
        self.class_to_idx = {clsname: i for i, clsname in enumerate(classes)}
        self.img_paths = []
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            s, t = 0, 0
            if os.path.isdir(cls_dir):
                dirnames = sorted(os.listdir(cls_dir))                  
                for img_name in dirnames:
                    img_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(img_path):
                        if s >= skip:
                            self.img_paths.append((img_path, self.class_to_idx[cls_name]))
                            t += 1
                        else:
                            s += 1
                        if t >= take:
                            break
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def get_idx(self, class_name):
        return self.class_to_idx[class_name]
    
    def get_class_label(self, idx):
        return self.labels[idx]
    
    def get_class_name(self, idx):
        return self.classes[idx]
    
    def name(self):
        return self.__class__.__name__