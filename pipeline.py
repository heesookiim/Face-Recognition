import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from kaggle import CelebrityDataset, TestDataset

data_dir = '/scratch/gilbreth/kim2903/train_cropped'
csv_file = '/scratch/gilbreth/kim2903/data/train.csv'
category_file = '/scratch/gilbreth/kim2903/data/category.csv'

batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 8


# Model, optimizer, and scheduler setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
resnet = InceptionResnetV1(
    classify=True,
    num_classes=100
).to(device)
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

# Initialize dataset and data loaders
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

dataset = CelebrityDataset(csv_file=csv_file, img_dir=data_dir, category_file=category_file, transform=transform)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
writer.close()

test_dir = '/scratch/gilbreth/kim2903/test_cropped'
test_dataset = TestDataset(img_dir=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=workers)
category_names = pd.read_csv(category_file)['Category'].tolist()

# Predict and save submission, ensuring sorted order of image names
default_celebrity = "Angelina Jolie"
# Assuming 'default_celebrity' is defined as "Angelina Jolie"
submission = {'Id': [], 'Category': []}

# Initialize the submission with default values
test_img_files = sorted(os.listdir('/scratch/gilbreth/kim2903/test'))
submission['Id'] = [int(file.split('.')[0]) for file in test_img_files]
submission['Category'] = [default_celebrity for _ in test_img_files]

# Prediction and update submission dictionary
resnet.eval()

for images, img_ids in test_loader:
    images = images.to(device)
    with torch.no_grad():
        outputs = resnet(images)
    _, predicted_indices = torch.max(outputs, 1)
    predicted_categories = [category_names[idx] for idx in predicted_indices.cpu().numpy()]
    
    for img_id, predicted_category in zip(img_ids, predicted_categories):
        # Find the index in submission using img_id and update the category
        idx = submission['Id'].index(img_id)
        submission['Category'][idx] = predicted_category

df_submission = pd.DataFrame(submission).sort_values('Id')
df_submission.to_csv('/scratch/gilbreth/kim2903/submission.csv', index=False)
