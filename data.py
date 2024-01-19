from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import random
import matplotlib.pyplot as plt

#Path to the dataset
TRAIN_IMG_FOLDER_PATH = "C:\Users\goyto\PycharmProjects\Defect_Classification\defe iden segmen\data_X_SDD"
VALID_IMG_FOLDER_PATH = "C:\Users\goyto\PycharmProjects\Defect_Classification\defe iden segmen\data_X_SDD"

train_augs = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(p = 0.5),
    T.RandomRotation(degrees=(-20, + 20)),
    T.ToTensor() # Convert a PIL image or numpy.ndarray to tensor (h, w, c) --> (c, h, w)
])

valid_augs = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

trainset = ImageFolder(TRAIN_IMG_FOLDER_PATH, transform = train_augs)
validset = ImageFolder(VALID_IMG_FOLDER_PATH, transform = valid_augs)

#Dataloader
train_loader = DataLoader(trainset, batch_size = 64, shuffle = True)
test_loader = DataLoader(validset, batch_size = 64)
