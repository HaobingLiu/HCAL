import shutil

from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import defaultdict

from pathlib import Path
BASE_DATA_PATH = os.path.join(Path(__file__).resolve().parent.parent, "data", "aircraft/")

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height


def load_box(dir="data/aircraft/"):
    box = {}
    box_file = dir + "images_box.txt"
    with open(box_file, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            img_id = parts[0]
            box[img_id] = parts[1:]  # 后续4个值为坐标
    return box


def convert_bbox(bbox):
    return [int(x) - 1 for x in bbox]


def crop(box, dir="data/aircraft/"):
    files = os.listdir(dir + "images")
    os.makedirs(dir + 'crop', exist_ok=True)
    for file in files:
        image = file[:-4]
        img = Image.open(dir + "images/" + file)
        bbox = tuple(convert_bbox(box[int(image)]))
        crop_img = img.crop(bbox)
        crop_img.save(dir + "crop/" + file)
        print(file," have saved in ",dir + "crop/" + file)

def label_dict(file):
    img2label = {}
    with open(file, 'r') as f:
        for line in f:
            parts = list(map(str, line.strip().split()))
            img2label[parts[0]] = "".join(parts[1:])
    label2img = invert(img2label)
    return img2label,label2img

def invert(dic):
    inv = {}
    for k, v in dic.items():
        if v not in inv:
            inv[v] = [k]
        else:
            inv[v].append(k)
    return inv

def flat(img2label,label2img):
    all_labels = set(img2label.values()) | set(label2img.keys())
    label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    new_dict_img2label = {
        img_name: label_to_id[label]
        for img_name, label in img2label.items()
    }
    new_dict_label2imgs = {
        label_to_id[label]: img_list
        for label, img_list in label2img.items()
    }
    return new_dict_img2label,new_dict_label2imgs

def valtotrain(dir="data/aircraft"):
    files = os.listdir(dir+"/val")
    for file in files:
        src_path = os.path.join(dir, "val", file)
        print(src_path)
        dst_path = os.path.join(dir, "train", file)
        print(dst_path)
        shutil.move(src_path, dst_path)
        print("move ", file)


def data_seg(dir="data/aircraft/",type="train"):
    path = "images_"+ type + ".txt"
    files = os.listdir(dir+"crop")
    with open(dir+path, 'r') as f:
        target_files = [line.strip() for line in f.readlines()]
    os.makedirs(dir+type, exist_ok=True)
    type_files = []
    for file in files:
        if file[:-4] in target_files:
            src_path = os.path.join(dir+"crop",file)
            dst_path = os.path.join(dir+type,file)
            if not os.path.exists(dst_path):
                shutil.move(src_path, dst_path)
                type_files.append(file)
                print("move ",file)
            else:
                print(f"文件 {file} 已存在，跳过移动")

def predata(dir="data/aircraft/",type = "train"):
    train_family = dir + "images_family_"+type+".txt"
    img2family, family2img = label_dict(train_family)
    img2intfamily,intfamily2img = flat(img2family, family2img)
    train_manu = dir + "images_manufacturer_"+type+".txt"
    img2manu, manu2img = label_dict(train_manu)
    img2intmanu,intmanu2img = flat(img2manu,manu2img)
    return img2intfamily,img2intmanu,intfamily2img,intmanu2img

def f2c(fine_img2label,coarse_img2label):
    fine2coarse = {}
    for img_name, fine_label in fine_img2label.items():
        coarse_label = coarse_img2label.get(img_name)
        if coarse_label is not None:
            if fine_label in fine2coarse:
                if fine2coarse[fine_label] != coarse_label:
                    print(f"冲突：标签 {fine_label} 对应多个子标签")
            else:
                fine2coarse[fine_label] = coarse_label
    return fine2coarse

def c2f(fine2coarse):
    coarse2fine = defaultdict(list)
    for k, v in fine2coarse.items():
        coarse2fine[v].append(k)
    return dict(coarse2fine)

class AirDataset(Dataset):
    def __init__(self, path = "data/aircraft/", transform = None, type = "train"):
        self.path = path
        self.sub_path = type
        self.transform = transform
        train_family = self.path + "images_family_" + self.sub_path + ".txt"
        img2family, family2img = label_dict(train_family)
        self.img2intfamily, self.intfamily2img = flat(img2family, family2img)
        train_manu = self.path + "images_manufacturer_" + self.sub_path + ".txt"
        img2manu, manu2img = label_dict(train_manu)
        self.img2intmanu, self.intmanu2img = flat(img2manu, manu2img)
        train_v = self.path + "images_variant_" + self.sub_path + ".txt"
        img2v, v2img = label_dict(train_v)
        self.img2intv, self.intv2img = flat(img2v, v2img)
        self.family2manu = f2c(self.img2intfamily, self.img2intmanu)
        self.manu2family = c2f(self.family2manu)
        self.v2family = f2c(self.img2intv, self.img2intfamily)
        self.family2v = c2f(self.v2family)
        self.fine_to_coarse_labels = f2c(self.img2intfamily, self.img2intmanu)
        self.coarse_to_fine_labels = c2f(self.fine_to_coarse_labels)


    def __getitem__(self, index):
        # 步骤1：获取图片名称
        img_names = list(self.img2intfamily.keys())
        jpg_names = [i + ".jpg" for i in img_names]
        img_name = img_names[index]
        jpg_name = jpg_names[index]

        # 步骤2：构建图片路径（假设图片存储在path/sub_path/目录）
        img_path = os.path.join(self.path, self.sub_path, jpg_name)

        # 步骤3：读取图片（PIL格式）
        image = Image.open(img_path).convert('RGB')

        # 步骤4：应用数据增强/预处理
        if self.transform:
            image = self.transform(image)

        # 步骤5：获取标签层级
        fine_label = self.img2intfamily[img_name]  # 细粒度标签（如飞机家族）
        coarse_label = self.img2intmanu[img_name]  # 粗粒度标签（如制造商）

        return image, fine_label, coarse_label

    def __len__(self):
        return len(self.img2intfamily.keys())

    def len_fine(self):
        return len(self.intfamily2img.keys())

    def len_coarse(self):
        return len(self.intmanu2img.keys())

    def coarse2fine(self):
        return self.coarse_to_fine_labels

    def fine2coarse(self):
        return self.fine_to_coarse_labels

class Air3Dataset(Dataset):
    def __init__(self, path = "data/aircraft/", transform = None, type = "train"):
        self.path = path
        self.sub_path = type
        self.transform = transform
        train_family = self.path + "images_family_" + self.sub_path + ".txt"
        img2family, family2img = label_dict(train_family)
        self.img2intfamily, self.intfamily2img = flat(img2family, family2img)
        train_manu = self.path + "images_manufacturer_" + self.sub_path + ".txt"
        img2manu, manu2img = label_dict(train_manu)
        self.img2intmanu, self.intmanu2img = flat(img2manu, manu2img)
        train_v = self.path + "images_variant_" + self.sub_path + ".txt"
        img2v, v2img = label_dict(train_v)
        self.img2intv, self.intv2img = flat(img2v, v2img)
        self.family2manu = f2c(self.img2intfamily, self.img2intmanu)
        self.manu2family = c2f(self.family2manu)
        self.v2family = f2c(self.img2intv, self.img2intfamily)
        self.family2v = c2f(self.v2family)
        self.fine_to_coarse_labels = f2c(self.img2intfamily, self.img2intmanu)
        self.coarse_to_fine_labels = c2f(self.fine_to_coarse_labels)


    def __getitem__(self, index):
        # 步骤1：获取图片名称
        img_names = list(self.img2intfamily.keys())
        jpg_names = [i + ".jpg" for i in img_names]
        img_name = img_names[index]
        jpg_name = jpg_names[index]

        # 步骤2：构建图片路径（假设图片存储在path/sub_path/目录）
        img_path = os.path.join(self.path, self.sub_path, jpg_name)

        # 步骤3：读取图片（PIL格式）
        image = Image.open(img_path).convert('RGB')

        # 步骤4：应用数据增强/预处理
        if self.transform:
            image = self.transform(image)

        # 步骤5：获取标签层级
        v_label = self.img2intv[img_name]
        family_label = self.img2intfamily[img_name]  # 细粒度标签（如飞机家族）
        manu_label = self.img2intmanu[img_name]  # 粗粒度标签（如制造商）

        return image, v_label, family_label, manu_label

    def __len__(self):
        return len(self.img2intfamily.keys())

    def len_1(self):
        return len(self.intv2img.keys())

    def len_2(self):
        return len(self.intfamily2img.keys())

    def len_3(self):
        return len(self.intmanu2img.keys())

    def h1to2(self):
        return self.v2family

    def h2to3(self):
        return self.family2manu

    def h2to1(self):
        return self.family2v

    def h3to2(self):
        return self.manu2family


class AirSubset(Subset):
    """继承原始数据集的属性和方法"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset

    def __getattr__(self, name):
        """访问自定义方法时，代理到原始数据集"""
        return getattr(self.dataset, name)


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

def make_transform(is_train=True):
    # Resolution Resize List : 256, 292, 361, 512
    # Resolution Crop List: 224, 256, 324, 448

    resnet_sz_resize = 256
    resnet_sz_crop = 224
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    resnet_transform = transforms.Compose([
        transforms.RandomResizedCrop(resnet_sz_crop) if is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.Resize(resnet_sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(resnet_sz_crop) if not is_train else Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

    return resnet_transform

batchsize=32
train_dataset = AirDataset(path=BASE_DATA_PATH, transform=make_transform(is_train=True),type="train")
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_dataset = AirDataset(path=BASE_DATA_PATH, transform=make_transform(is_train=True),type="val")
val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
test_dataset = AirDataset(path=BASE_DATA_PATH, transform=make_transform(is_train=False),type="test")
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

train_dataset_3 = Air3Dataset(path=BASE_DATA_PATH, transform=make_transform(is_train=True),type="train")
train_loader_3 = DataLoader(train_dataset_3, batch_size=batchsize, shuffle=True)
test_dataset_3 = Air3Dataset(path=BASE_DATA_PATH, transform=make_transform(is_train=False),type="test")
test_loader_3 = DataLoader(test_dataset_3, batch_size=batchsize, shuffle=True)


if __name__ == "__main__":
    # dir = "data/aircraft/"
    # box = load_box(dir)
    # crop(box)
    # all = ["train","test","val"]
    # for t in all:
    #     data_seg(type=t)
    # valtotrain()
    print(train_loader_3.dataset.h1to2())
    # for images, fine_labels, coarse_labels in train_loader:
    #     print(train_loader.dataset.len_fine(),train_loader.dataset.len_coarse())
    #     print("images: ", images)
    #     print(f"Fine Labels: {fine_labels}")
    #     print(f"Coarse Labels: {coarse_labels}")

