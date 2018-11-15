# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import skimage.color
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy.misc import toimage

device = torch.device('cuda')     # Default CUDA device
Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor


### Parameters
SIZE = 512
BETA1 = 0.5
BETA2 = 0.999
LAMBDA = 10
ALPHA = 1000
BATCH_SIZE = 5
NUM_EPOCHS = 5
LATENT_DIM = 100
TRAIN_NUM = 50


# ### Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #  Convolutional layers 
        
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1, padding = 2)
        self.conv1_bn = nn.BatchNorm2d(16)
        
        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv2_bn = nn.BatchNorm2d(32)
        
        # input 265x256x32  output 128x128x64
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.conv3_bn = nn.BatchNorm2d(64)
        
        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Conv2d(64, 128, 5, stride = 2, padding = 2)
        self.conv4_bn = nn.BatchNorm2d(128)
        
        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Conv2d(128, 128, 5, stride = 2, padding = 2)
        self.conv5_bn = nn.BatchNorm2d(128)
        
        # convs for global features
        # input 32x32x128 output 16x16x128
        self.conv51 = nn.Conv2d(128,128,5, stride =2 , padding =2 )
        
        # input 16x16x128 output 8x8x128
        self.conv52 = nn.Conv2d(128,128,5, stride =2 , padding =2 )
        
        # input 8x8x128 output 1x1x128
        self.conv531 = nn.Conv2d(128,128,5, stride =2 , padding =1 )
        
        # input 1x1x128 output 1x1x128
        self.conv532 = nn.Conv2d(128,128,5, stride =2 , padding =1 )
        
        # input 32x32x128 output 32x32x128
        # the global features should be concatenated to the feature map aftere this layer
        # the output after concat would be 32x32x256
        self.conv6 = nn.Conv2d(128, 128, 5, stride = 1, padding =2)
        
        # input 32x32x256 output 32x32x128
        self.conv7 = nn.Conv2d(256, 128, 5, stride = 1, padding = 2)
        
        # deconvolutional layers
        # input 32x32x128 output 64x64x128
        self.dconv1 = nn.ConvTranspose2d(128, 128, 4, stride = 2, padding = 1)
        self.dconv1_bn = nn.BatchNorm2d(128)
        
        # input 64x64x256 ouput 128x128x128
        self.dconv2 = nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1)
        self.dconv2_bn = nn.BatchNorm2d(256)
        
        # input 128x128x192 output 256x256x64
        self.dconv3 = nn.ConvTranspose2d(192, 64, 4, stride = 2, padding = 1)
        self.dconv3_bn = nn.BatchNorm2d(192)
        
        # input 256x256x96 ouput 512x512x32
        self.dconv4 = nn.ConvTranspose2d(96, 32, 4, stride = 2, padding = 1)
        self.dconv4_bn = nn.BatchNorm2d(96)
        
        # final convolutional layers
        # input 512x512x48 output 512x512x16
        self.conv8 = nn.Conv2d(48, 16, 5, stride = 1, padding = 2)
        self.conv8_bn = nn.BatchNorm2d(48)
        
        # input 512x512x16 output 512x512x3
        self.conv9 = nn.Conv2d(16, 3, 5, stride = 1, padding = 2)    
        self.conv9_bn = nn.BatchNorm2d(16)
        # SELU
                
    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x = self.conv1_bn(F.selu(self.conv1(x)))

        # input 512x512x16 to output 256x256x32
        x1 = self.conv2_bn(F.selu(self.conv2(x)))

        # input 256x256x32 to output 128x128x64
        x2 = self.conv3_bn(F.selu(self.conv3(x1)))

        # input 128x128x64 to output 64x64x128
        x3 = self.conv4_bn(F.selu(self.conv4(x2)))

        # input 64x64x128 to output 32x32x128
        x4 = self.conv5_bn(F.selu(self.conv5(x3)))

        #convolutions for global features
        # input 32x32x128 to output 16x16x128
        x51 = self.conv51(x4)
        
        # input 16x16x128 to output 8x8x128
        x52 = self.conv52(x51)
        
        # input 8x8x128 to output 1x1x128
        x53 = self.conv532(F.selu(self.conv531(x52)))
        x53_temp = torch.cat([x53]*32,dim = 2 )
        x53_temp = torch.cat([x53_temp]*32,dim=3)

        # input 32x32x256 to output 32x32x128
        x5 = self.conv6(x4)
        
        # input 32x32x128 to output 32x32x128
        x5 = self.conv7(torch.cat([x5,x53_temp],dim=1))

        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(self.dconv1_bn(F.selu(x5)))

        # input 64x64x256 to output 128x128x128
        xd = self.dconv2(self.dconv2_bn(F.selu(torch.cat([xd,x3], dim=1))))

        # input 128x128x192 to output 256x256x64
        xd = self.dconv3(self.dconv3_bn(F.selu(torch.cat([xd,x2],dim=1))))

        # input 256x256x64 to output 512x512x32
        xd = self.dconv4(self.dconv4_bn(F.selu(torch.cat([xd,x1],dim=1))))

        # input 512x512x48 to output 512x512x16
        xd = self.conv8(self.conv8_bn(F.selu(torch.cat([xd,x],dim=1))))

        # input 512x512x16 to output 512x512x3
        xd = self.conv9(self.conv9_bn(F.selu((xd))))
        return xd


# ### Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #  Convolutional layers 
        
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1, padding = 2)
        self.conv1_in = nn.InstanceNorm2d(16)
        
        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv2_in = nn.InstanceNorm2d(32)
        
        # input 265x256x32  output 128x128x64
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.conv3_in = nn.InstanceNorm2d(64)
        
        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Conv2d(64, 128, 5, stride = 2, padding = 2)
        self.conv4_in = nn.InstanceNorm2d(128)
        
        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Conv2d(128, 128, 5, stride = 2, padding = 2)
        self.conv5_in = nn.InstanceNorm2d(128)
        
        # input 32x32x128  output 16x16x128
        # the output of this layer we need layers for global features
        self.conv6 = nn.Conv2d(128, 128, 5, stride = 2, padding = 2)
        self.conv6_in = nn.InstanceNorm2d(128)
        
        # input 16x16x128  output 1x1x1
        # the output of this layer we need layers for global features
        self.conv7 = nn.Conv2d(128, 1, 16)
        self.conv7_in = nn.InstanceNorm2d(1)
        
    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x = self.conv1_in(F.leaky_relu(self.conv1(x)))

        # input 512x512x16 to output 256x256x32
        x = self.conv2_in(F.leaky_relu(self.conv2(x)))
        
        # input 256x256x32 to output 128x128x64
        x = self.conv3_in(F.leaky_relu(self.conv3(x)))
        
        # input 128x128x64 to output 64x64x128
        x = self.conv4_in(F.leaky_relu(self.conv4(x)))
        
        # input 64x64x128 to output 32x32x128
        x = self.conv5_in(F.leaky_relu(self.conv5(x)))
        
        # input 32x32x128 to output 16x16x128
        x = self.conv6_in(F.leaky_relu(self.conv6(x)))
        
        # input 16x16x128 to output 1x1x1
        x = self.conv7(x)
        x = F.leaky_relu(x)
        
        return x

generator1 = Generator()
discriminator = Discriminator()


# ### Loading Training and Test Set Data
# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
transform = transforms.Compose([transforms.Resize((SIZE,SIZE), interpolation=2),transforms.ToTensor()])

trainset_1_gt =torchvision.datasets.ImageFolder(root='G:/DeepLearningData/images/Expert-C/Training1/', transform=transform, target_transform=None)    
trainset_2_gt =torchvision.datasets.ImageFolder(root='G:/DeepLearningData/images/Expert-C/Training2/', transform=transform, target_transform=None)    
testset_gt =torchvision.datasets.ImageFolder(root='G:DeepLearningData/images/Expert-C/Testing/', transform=transform, target_transform=None)      
trainset_1_inp =torchvision.datasets.ImageFolder(root='G:/DeepLearningData/images/tif-images/Training1/', transform=transform, target_transform=None)    
trainset_2_inp =torchvision.datasets.ImageFolder(root='G:/DeepLearningData/images/tif-images/Training2/', transform=transform, target_transform=None)    
testset_inp =torchvision.datasets.ImageFolder(root='G:/DeepLearningData/images/tif-images/Testing/', transform=transform, target_transform=None)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


trainLoader1 = torch.utils.data.DataLoader(
             ConcatDataset(
                 trainset_1_gt,
                 trainset_1_inp
             ),
             batch_size=BATCH_SIZE, shuffle=True,)

trainLoader2 = torch.utils.data.DataLoader(
             ConcatDataset(
                 trainset_2_gt,
                 trainset_2_inp
             ),
             batch_size=BATCH_SIZE, shuffle=True,)

testLoader = torch.utils.data.DataLoader(
             ConcatDataset(
                 testset_gt,
                 testset_inp
             ),
             batch_size=BATCH_SIZE, shuffle=True,)


# ### MSE Loss and Optimizer
criterion = nn.MSELoss()

optimizer_g1 = optim.Adam(generator1.parameters(), lr = 0.001, betas=(BETA1,BETA2))
optimizer_d = optim.Adam(discriminator.parameters(), lr = 0.001, betas=(BETA1,BETA2))


# ### Gradient Penalty
# Computes gradient penalty loss for A-WGAN
def computeGradientPenalty(D, realSample, fakeSample):
    alpha = Tensor(np.random.random((realSample.shape)))
    interpolates = (alpha * realSample + ((1 - alpha) * fakeSample)).requires_grad_(True)
    dInterpolation = D(interpolates)
    fakeOutput = Variable(Tensor(realSample.shape[0],1,1,1).fill_(1.0), requires_grad=False)
    
    gradients = autograd.grad(
        outputs = dInterpolation,
        inputs = interpolates,
        grad_outputs = fakeOutput,
        create_graph = True,
        retain_graph = True,
        only_inputs = True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    maxVals = []
    normGradients = gradients.norm(2, dim=1)-1
    for i in range(len(normGradients)):
        if(normGradients[i] > 0):
            maxVals.append(normGradients[i].detach().numpy())
        else:
            maxVals.append(0)

    gradientPenalty = np.mean(maxVals)
    return gradientPenalty


# ### Generator Loss
def generatorAdversarialLoss( output_images):
    validity = discriminator(output_images)
    gen_adv_loss = torch.mean(validity)
    
    return gen_adv_loss


def computeGeneratorLoss(inputs, outputs_g1):
    # generator 1
    gen_adv_loss1 = generatorAdversarialLoss(outputs_g1)
    
    i_loss = criterion(inputs, outputs_g1)
    gen_loss = -gen_adv_loss1 + ALPHA*i_loss
    
    return gen_loss


# ### Discriminator Loss
def discriminatorLoss(d1Real, d1Fake, gradPenalty):
    return (torch.mean(d1Fake) - torch.mean(d1Real)) + (LAMBDA*gradPenalty)


# ## Generator Training Loop
# trained on the first 2250 images
batches_done = 0
running_loss = 0.0
for epoch in range(NUM_EPOCHS):
    for i,  (target, input) in enumerate(trainLoader1, 0):
        unenhanced_image = input[0]
        enhanced_image = target[0] 
        
        optimizer_g1.zero_grad()
        
        generated_enhanced_image = generator1(enhanced_image)
        
        loss = torch.log10(criterion(generated_enhanced_image, enhanced_image))
        loss.backward()
        optimizer_g1.step()
        
        
        # print statistics
        running_loss += loss.item()
        if i % 2 == 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 2))
            running_loss = 0.0


# ### Training Network
batches_done = 0
for epoch in range(NUM_EPOCHS):
    for i, (data, gt1) in enumerate(zip(trainloader2_inp, trainloader1_gt)):
        input, dummy = data
        groundTruth, dummy = gt1
        trainInput = Variable(input.type(Tensor))
        realImgs = Variable(groundTruth.type(Tensor))
        
        ### TRAIN DISCRIMINATOR
        optimizer_d.zero_grad()
        fakeImgs = generator1(trainInput)
        
        # Real Images
        realValid = discriminator(realImgs)
        # Fake Images
        fakeValid = discriminator(fakeImgs)
        
        gradientPenalty = computeGradientPenalty(discriminator, realImgs.data, fakeImgs.data)
        dLoss = discriminatorLoss(realValid, fakeValid, gradientPenalty)
        dLoss.backward()
        optimizer_d.step()
        optimizer_g1.zero_grad()
        
        ### TRAIN GENERATOR
        if TRAIN_NUM % 50 == 0:
            print("Training Generator on Iteration: %d" % (i))
            # Generate a batch of images
            fake_imgs = generator1(trainInput)
            residual_learning_output = fake_imgs + trainInput
            gLoss = computeGeneratorLoss(trainInput,residual_learning_output)

            gLoss.backward()
            optimizer_g1.step()
    
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: |%f] [G loss: %f]" % (epoch, NUM_EPOCHS , i, len(trainloader2_inp), dLoss.item(), gLoss.item()))
            f = open("logStatus.txt","a+")
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: |%f] [G loss: %f]\n" % (epoch, NUM_EPOCHS , i, len(trainloader2_inp), dLoss.item(), gLoss.item()))
            f.close()
            
            if batches_done % 200 == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += TRAIN_NUM
            print("Done training generator on iteration: %d" % (i))    

