#####################################
# main code to train CNN for Master thesis of Eike Bergen
# Architechture based on:
# "The Boombox: Visual Reconstruction from Acoustic Vibrations" (https://arxiv.org/abs/2105.08052)
#####################################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

# check for CUDA availability, set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name_specs = 'model_1second_4layers_1079samples_90percent_100_epochs_adjusted_learn_rate_threshold'

model_folder = model_name_specs
os.makedirs(model_folder, exist_ok=True)

# helper functions for creating layers
def conv2d_bn_relu(inch, outch, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(outch),
        nn.ReLU()
    )

def deconv_relu(inch, outch, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(outch),
        nn.ReLU()
    )

def deconv_sigmoid(inch, outch, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.Sigmoid()
    )

# CNN Architechture
class EncoderDecoder(nn.Module):
    def __init__(self, in_channels, target_size):
        super(EncoderDecoder, self).__init__()
        self.target_size = target_size
        self.conv_stack1 = nn.Sequential(
            conv2d_bn_relu(in_channels, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack2 = nn.Sequential(
            conv2d_bn_relu(32, 64, 4, stride=2),
            conv2d_bn_relu(64, 64, 3)
        )
        self.conv_stack3 = nn.Sequential(
            conv2d_bn_relu(64, 128, 4, stride=2),
            conv2d_bn_relu(128, 128, 3)
        )
        self.conv_stack4 = nn.Sequential(
            conv2d_bn_relu(128, 128, 4, stride=2),
            conv2d_bn_relu(128, 128, 3)
        )
        
        self.deconv_4 = deconv_relu(128, 128, 4, stride=2)
        self.deconv_3 = deconv_relu(131, 64, 4, stride=2)
        self.deconv_2 = deconv_relu(67, 32, 4, stride=2)
        self.deconv_1 = deconv_sigmoid(35, 1, 4, stride=2)

        self.predict_4 = nn.Conv2d(128, 3, 3, stride=1, padding=1)
        self.predict_3 = nn.Conv2d(131, 3, 3, stride=1, padding=1)
        self.predict_2 = nn.Conv2d(67, 3, 3, stride=1, padding=1)

        self.up_sample_4 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)

        deconv4_out = self.deconv_4(conv4_out)
        predict_4_out = self.up_sample_4(self.predict_4(conv4_out))

        concat_3 = torch.cat([deconv4_out, predict_4_out], dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))

        concat_2 = torch.cat([deconv3_out, predict_3_out], dim=1)
        deconv2_out = self.deconv_2(concat_2)
        predict_2_out = self.up_sample_2(self.predict_2(concat_2))

        concat_1 = torch.cat([deconv2_out, predict_2_out], dim=1)
        predict_out = self.deconv_1(concat_1)

        # resize output to target size
        predict_out = F.interpolate(predict_out, size=self.target_size, mode='bilinear', align_corners=False)

        return predict_out

class AudioImageDataset(Dataset):
    def __init__(self, audio_dir, image_dir, transform=None, target_transform=None):
        self.audio_dirs = [os.path.join(audio_dir, d) for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

        self.audio_files = [[os.path.join(subdir, file) for file in sorted(os.listdir(subdir)) if file.endswith('.png')] for subdir in self.audio_dirs]
        self.image_files = [os.path.join(image_dir, file) for file in sorted(os.listdir(image_dir)) if file.endswith('.png')]

        num_files = len(self.audio_files[0])
        if not all(len(files) == num_files for files in self.audio_files) or not len(self.image_files) == num_files:
            raise ValueError("Mismatch in the number of files across subfolders or with the ground truth images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        spectrogram_images = [Image.open(files[idx]).convert('RGB') for files in self.audio_files]
        if self.transform:
            spectrogram_images = [self.transform(img) for img in spectrogram_images]

        spectrograms = torch.cat(spectrogram_images, dim=0)

        ground_truth_image = Image.open(self.image_files[idx]).convert('L')
        if self.target_transform:
            ground_truth_image = self.target_transform(ground_truth_image)

        return spectrograms, ground_truth_image

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float())
])

# Dataset and DataLoader
audio_dir = 'Train_Data_1second'
image_dir = 'Train_label_1second'
target_size = (480, 640)
dataset = AudioImageDataset(audio_dir, image_dir, transform=transform, target_transform=target_transform)


# split dataset into training and validation sets
val_percentage = 0.1  # percentage of validation samples
val_size = int(len(dataset) * val_percentage)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# initialize the model, loss criterion, and optimizer
model = EncoderDecoder(in_channels=12, target_size=target_size).to(device)  # Move the model to the appropriate device
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)  # Starting learning rate of 0.001

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.001, min_lr=1e-6)


epoch_number = 100

# Training Loop
for epoch in range(epoch_number):  # Example: 200 epochs
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
    
    # step the scheduler with the epoch loss
    scheduler.step(epoch_loss)

    # save the model at regular intervals
    if (epoch + 1) % 5 == 0:
        model_path = os.path.join(model_folder, f'model_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss
        }, model_path)
        print(f'Saved model at epoch {epoch + 1} to {model_path}')


model_name = model_name_specs = ".pth"

# save the final trained model
torch.save({
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': epoch_loss
}, model_name)

# function to evaluate the model
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss}')
    return avg_loss


# evaluate the model
evaluate_model(model, val_loader, criterion)