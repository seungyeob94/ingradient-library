import torch.optim as optim
from torch.utils.data import random_split
import torch
from dataloads import *
from unet3d import *
from preprocessing import *
from dataaugmentation import *
from deepsupervision_loss import *
from visualization import *

DATASET_PATH = '/content/gdrive/MyDrive/DeepVibe/Datasets/BRATS2020'
SAVED_MODEL = '/content/gdrive/MyDrive/DeepVibe/Model State Dictionary/3D UNet with Deep Supervision Weight Loss/epoch_14_deepsupervision_UNet3d.pkl'
HISTORY_PATH = '/content/gdrive/MyDrive/DeepVibe/Model State Dictionary/3D UNet with Deep Supervision Weight Loss'

torch.manual_seed(0)
ds = CustomDataset(DATASET_PATH, normalizer=Normalizer())
tr_ds, val_ds = random_split(ds, [len(ds) - len(ds)//9, len(ds)//9])
da = DataAugmentation()
tr_dl = DataLoader3D(tr_ds, da, batch_size = 2, num_iteration = 1)
val_dl = DataLoader3D(val_ds, batch_size = 2, num_iteration = 1)

model = UNet3DDeepsupervision().to(0)
model_dict = torch.load('/content/gdrive/MyDrive/DeepVibe/Model State Dictionary/3D UNet with Deep Supervision Weight Loss/epoch_21_deepsupervision_UNet3d.pkl')
model.load_state_dict(model_dict)
optimizer = optim.Adam(params = model.parameters(), lr = 0.01)
epoch = 1000

with open(os.path.join(HISTORY_PATH,"epoch_history.pkl"), "rb") as fp:   #Pickling
    train_loss, valid_loss = pickle.load(fp)

for e in range(15, epoch):
    optimizer.param_groups[0]['lr'] = 0.01 * ((e/epoch)**0.9)
    tr_dl.new_epoch()
    iter_loss = []
    model.train()
    while not tr_dl.is_end():
        optimizer.zero_grad()
        images, seg = tr_dl.generate_train_batch()
        output = model(images)
        loss = deep_supervision_loss(output, seg)
        iter_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    val_dl.new_epoch()
    val_iter_loss = []

    visualization_index = 24
    while not val_dl.is_end():
        with torch.no_grad():
            images, seg = val_dl.generate_train_batch()
            output = model(images)
            loss = deep_supervision_loss(output, seg)
            val_iter_loss.append(loss.item())
            if val_dl.current_index == visualization_index:
                print("iteration : ", val_dl.current_index + 1)
                deep_supervision_visualization(output, seg)

    
    file_name = 'epoch_'+ str(e) + '_deepsupervision_UNet3d.pkl'
    torch.save(model.state_dict(), os.path.join(HISTORY_PATH,file_name))
    train_loss.append(np.array(iter_loss).mean())
    valid_loss.append(np.array(val_iter_loss).mean())
    print("epoch :", e+1, " Train Loss :",train_loss[-1], "Valid Loss :", valid_loss[-1])
    with open(os.path.join(HISTORY_PATH, "epoch_history.pkl"), "wb") as fp:
        pickle.dump([train_loss, valid_loss], fp)
        
