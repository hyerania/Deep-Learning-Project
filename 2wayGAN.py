#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import toimage
import skimage.color
from torch import autograd
from torch.autograd import Variable

device = torch.device('cuda')     # Default CUDA device
Tensor_gpu = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Tensor = torch.FloatTensor


# ### Parameters

# In[2]:


SIZE = 512
BETA1 = 0.5
BETA2 = 0.999
LAMBDA = 10
ALPHA = 1000
BATCH_SIZE = 5
NUM_EPOCHS = 5
LATENT_DIM = 100
TRAIN_NUM = 50


# # Generator 1 Network

# In[3]:


class Generator(nn.Module):
    
    direction = 1;
    
    def changeDirection(self,i):
        self.direction = i
        
    def __init__(self):
        super(Generator, self).__init__()
        #  Convolutional layers 
        
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1, padding = 2)
        self.conv1_bn1 = nn.BatchNorm2d(16)
        self.conv1_bn2 = nn.BatchNorm2d(16)
        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv2_bn1 = nn.BatchNorm2d(32)
        self.conv2_bn2 = nn.BatchNorm2d(32)

        # input 265x256x32  output 128x128x64
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.conv3_bn1 = nn.BatchNorm2d(64)
        self.conv3_bn2 = nn.BatchNorm2d(64)
        
        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Conv2d(64, 128, 5, stride = 2, padding = 2)
        self.conv4_bn1 = nn.BatchNorm2d(128)
        self.conv4_bn2 = nn.BatchNorm2d(128)
        
        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Conv2d(128, 128, 5, stride = 2, padding = 2)
        self.conv5_bn1 = nn.BatchNorm2d(128)
        self.conv5_bn2 = nn.BatchNorm2d(128)
        
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
        self.dconv1_bn1 = nn.BatchNorm2d(128)
        self.dconv1_bn2 = nn.BatchNorm2d(128)
        
        # input 64x64x256 ouput 128x128x128
        self.dconv2 = nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1)
        self.dconv2_bn1 = nn.BatchNorm2d(256)
        self.dconv2_bn2 = nn.BatchNorm2d(256)
        
        # input 128x128x192 output 256x256x64
        self.dconv3 = nn.ConvTranspose2d(192, 64, 4, stride = 2, padding = 1)
        self.dconv3_bn1 = nn.BatchNorm2d(192)
        self.dconv3_bn2 = nn.BatchNorm2d(192)
        
        # input 256x256x96 ouput 512x512x32
        self.dconv4 = nn.ConvTranspose2d(96, 32, 4, stride = 2, padding = 1)
        self.dconv4_bn1 = nn.BatchNorm2d(96)
        self.dconv4_bn2 = nn.BatchNorm2d(96)
        
        # final convolutional layers
        # input 512x512x48 output 512x512x16
        self.conv8 = nn.Conv2d(48, 16, 5, stride = 1, padding = 2)
        self.conv8_bn1 = nn.BatchNorm2d(48)
        self.conv8_bn2 = nn.BatchNorm2d(48)
        
        # input 512x512x16 output 512x512x3
        self.conv9 = nn.Conv2d(16, 3, 5, stride = 1, padding = 2)    
        self.conv9_bn1 = nn.BatchNorm2d(16)
        self.conv9_bn2 = nn.BatchNorm2d(16)
        
        # SELU
    
    def forward(self, x):
        if(self.direction == 1):
               return self.forward_step_dir_1(x)
        else:
               return self.forward_step_dir_2(x)

            
    def forward_step_dir_1(self, x):
            # input 512x512x3 to output 512x512x16
        x = self.conv1_bn1(F.selu(self.conv1(x)))
        
#         print("x")
#         print(x.shape)
        # input 512x512x16 to output 256x256x32
        x1 = self.conv2_bn1(F.selu(self.conv2(x)))
#         print("x1")
#         print(x1.shape)
        # input 256x256x32 to output 128x128x64
        x2 = self.conv3_bn1(F.selu(self.conv3(x1)))
#         print("x2")
#         print(x2.shape)
        # input 128x128x64 to output 64x64x128
        x3 = self.conv4_bn1(F.selu(self.conv4(x2)))
#         print("x3")
#         print(x3.shape)
        # input 64x64x128 to output 32x32x128
        x4 = self.conv5_bn1(F.selu(self.conv5(x3)))
#         print("x4")
#         print(x4.shape)
        #convolutions for global features
        # input 32x32x128 to output 16x16x128
        x51 = self.conv51(x4)
#         print("x51")
#         print(x51.shape)
        # input 16x16x128 to output 8x8x128
        x52 = self.conv52(x51)
#         print("x52")
#         print(x52.shape)
        # input 8x8x128 to output 1x1x128
        x53 = self.conv532(F.selu(self.conv531(x52)))
#         print("x53")
#         print(x53.shape)
        x53_temp = torch.cat([x53]*32,dim = 2 )
        x53_temp = torch.cat([x53_temp]*32,dim=3)
#         print("x53_temp")
#         print(x53_temp.shape)
        
        # input 32x32x256 to output 32x32x128
        x5 = self.conv6(x4)
#         print("x5")
#         print(x5.shape)
        # input 32x32x128 to output 32x32x128
        x5 = self.conv7(torch.cat([x5,x53_temp],dim=1))
#         print("x5")
#         print(x5.shape)
        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(self.dconv1_bn1(F.selu(x5)))
#         print("xd1")
#         print(xd.shape)
        # input 64x64x256 to output 128x128x128
        xd = self.dconv2(self.dconv2_bn1(F.selu(torch.cat([xd,x3], dim=1))))
#         print("xd2")
#         print(xd.shape)
        # input 128x128x192 to output 256x256x64
        xd = self.dconv3(self.dconv3_bn1(F.selu(torch.cat([xd,x2],dim=1))))
#         print("xd3")
#         print(xd.shape)
        # input 256x256x64 to output 512x512x32
        xd = self.dconv4(self.dconv4_bn1(F.selu(torch.cat([xd,x1],dim=1))))
#         print("xd4")
#         print(xd.shape)
        # input 512x512x48 to output 512x512x16
        xd = self.conv8(self.conv8_bn1(F.selu(torch.cat([xd,x],dim=1))))
#         print("xd 8")
#         print(xd.shape)
        # input 512x512x16 to output 512x512x3
        xd = self.conv9(self.conv9_bn1(F.selu((xd))))
#         print("xd 9")
#         print(xd.shape)
        return xd


    def forward_step_dir_2(self, x):
    # input 512x512x3 to output 512x512x16
        x = self.conv1_bn2(F.selu(self.conv1(x)))
        
#         print("x")
#         print(x.shape)
        # input 512x512x16 to output 256x256x32
        x1 = self.conv2_bn2(F.selu(self.conv2(x)))
#         print("x1")
#         print(x1.shape)
        # input 256x256x32 to output 128x128x64
        x2 = self.conv3_bn2(F.selu(self.conv3(x1)))
#         print("x2")
#         print(x2.shape)
        # input 128x128x64 to output 64x64x128
        x3 = self.conv4_bn2(F.selu(self.conv4(x2)))
#         print("x3")
#         print(x3.shape)
        # input 64x64x128 to output 32x32x128
        x4 = self.conv5_bn2(F.selu(self.conv5(x3)))
#         print("x4")
#         print(x4.shape)
        #convolutions for global features
        # input 32x32x128 to output 16x16x128
        x51 = self.conv51(x4)
#         print("x51")
#         print(x51.shape)
        # input 16x16x128 to output 8x8x128
        x52 = self.conv52(x51)
#         print("x52")
#         print(x52.shape)
        # input 8x8x128 to output 1x1x128
        x53 = self.conv532(F.selu(self.conv531(x52)))
#         print("x53")
#         print(x53.shape)
        x53_temp = torch.cat([x53]*32,dim = 2 )
        x53_temp = torch.cat([x53_temp]*32,dim=3)
#         print("x53_temp")
#         print(x53_temp.shape)
        
        # input 32x32x256 to output 32x32x128
        x5 = self.conv6(x4)
#         print("x5")
#         print(x5.shape)
        # input 32x32x128 to output 32x32x128
        x5 = self.conv7(torch.cat([x5,x53_temp],dim=1))
#         print("x5")
#         print(x5.shape)
        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(self.dconv1_bn2(F.selu(x5)))
#         print("xd1")
#         print(xd.shape)
        # input 64x64x256 to output 128x128x128
        xd = self.dconv2(self.dconv2_bn2(F.selu(torch.cat([xd,x3], dim=1))))
#         print("xd2")
#         print(xd.shape)
        # input 128x128x192 to output 256x256x64
        xd = self.dconv3(self.dconv3_bn2(F.selu(torch.cat([xd,x2],dim=1))))
#         print("xd3")
#         print(xd.shape)
        # input 256x256x64 to output 512x512x32
        xd = self.dconv4(self.dconv4_bn2(F.selu(torch.cat([xd,x1],dim=1))))
#         print("xd4")
#         print(xd.shape)
        # input 512x512x48 to output 512x512x16
        xd = self.conv8(self.conv8_bn2(F.selu(torch.cat([xd,x],dim=1))))
#         print("xd 8")
#         print(xd.shape)
        # input 512x512x16 to output 512x512x3
        xd = self.conv9(self.conv9_bn2(F.selu((xd))))
#         print("xd 9")
#         print(xd.shape)
        return xd


# ### Discriminator Network

# In[4]:


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
#         print("x1")
#         print(x.shape)
        # input 512x512x16 to output 256x256x32
        x = self.conv2_in(F.leaky_relu(self.conv2(x)))
#         print("x2")
#         print(x.shape)
        
        # input 256x256x32 to output 128x128x64
        x = self.conv3_in(F.leaky_relu(self.conv3(x)))
#         print("x3")
#         print(x.shape)
        
        # input 128x128x64 to output 64x64x128
        x = self.conv4_in(F.leaky_relu(self.conv4(x)))
#         print("x4")
#         print(x.shape)
        
        # input 64x64x128 to output 32x32x128
        x = self.conv5_in(F.leaky_relu(self.conv5(x)))
#         print("x5")
#         print(x.shape)
        
        # input 32x32x128 to output 16x16x128
        x = self.conv6_in(F.leaky_relu(self.conv6(x)))
#         print("x6")
#         print(x.shape)
        
        # input 16x16x128 to output 1x1x1
        x = self.conv7(x)
        x = F.leaky_relu(x)
        
        return x


# In[5]:


generator1 = Generator()
generator1.changeDirection(1)
generator2 = Generator()
generator2.changeDirection(1)
discriminator = Discriminator()
# print(generator1)
# print(generator2)
# print(discriminator)

if torch.cuda.is_available():
    generator1.to(device)
    generator2.to(device)
    discriminator.to(device)


# ### Loading Training and Test Set Data

# In[6]:


# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
transform = transforms.Compose([transforms.Resize((SIZE,SIZE), interpolation=2),transforms.ToTensor()])

trainset_1_gt =torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Training1/', transform=transform, target_transform=None)    
trainset_2_gt =torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Training2/', transform=transform, target_transform=None)    
testset_gt =torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Testing/', transform=transform, target_transform=None)    

# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
trainset_1_inp =torchvision.datasets.ImageFolder(root='./images_LR/input/Training1/', transform=transform, target_transform=None)    
trainset_2_inp =torchvision.datasets.ImageFolder(root='./images_LR/input/Training2/', transform=transform, target_transform=None)    
testset_inp =torchvision.datasets.ImageFolder(root='./images_LR/input/Testing/', transform=transform, target_transform=None)    


# In[7]:


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

trainLoader_full = torch.utils.data.DataLoader(
             ConcatDataset(
                 trainset_2_inp,
                 trainset_1_gt,
                 trainset_2_gt,
                 trainset_1_inp
             ),
             batch_size=BATCH_SIZE, shuffle=True,)


# In[8]:


print(trainLoader1)

dataiter = iter(trainLoader1)
print(dataiter)
(target,input) = dataiter.next()
print(target[0].shape)
print(target[1].shape)
print(input[0].shape)
print(input[1].shape)


# ### MSE Loss and Optimizer

# In[ ]:


criterion = nn.MSELoss()

optimizer_g1 = optim.Adam(generator1.parameters(), lr = 0.001, betas=(BETA1,BETA2))
optimizer_g2 = optim.Adam(generator2.parameters(), lr = 0.001, betas=(BETA1,BETA2))
optimizer_d = optim.Adam(discriminator.parameters(), lr = 0.001, betas=(BETA1,BETA2))

# Tensor = torch.FloatTensor


# ### Gradient Penalty
# Computes gradient penalty loss for A-WGAN

# In[9]:


def computeGradientPenalty(D, realSample, fakeSample):
    alpha = Tensor_gpu(np.random.random((realSample.shape)))
    interpolates = (alpha * realSample + ((1 - alpha) * fakeSample)).requires_grad_(True)
    dInterpolation = D(interpolates)
    fakeOutput = Variable(Tensor_gpu(realSample.shape[0],1,1,1).fill_(1.0), requires_grad=False)
    
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
            maxVals.append(Variable(normGradients[i].type(Tensor)).detach().numpy())
        else:
            maxVals.append(0)

    gradientPenalty = np.mean(maxVals)
    return gradientPenalty


# ### Generator Loss 2 WAY

# In[10]:


#losses of the GAN
def generatorAdversarialLoss(output_images):
    validity = discriminator(output_images)
    gen_adv_loss = torch.mean(validity)
    
    return gen_adv_loss

# loss between the input and the input that is generated after passing the input to both the generators
def computeConsistencyLoss(input, generated_input):
    return criterion(input, generated_input)

# loss between the input and the output generated from one generator
def computeIdentityLoss(input, generated_output):
    return criterion(input, generated_output)

# includes the adversarial loss and identity mapping loss for one generator
# can use it on both the generators

def computeGeneratorLoss(inputs,outputs):
    # generator
    gen_adv_loss = generatorAdversarialLoss(outputs)
    
     # generator Identity mapping loss
    i_loss = computeIdentityLoss(inputs, outputs)
    
    gen_loss = -gen_adv_loss1 + ALPHA*i_loss
    
    return gen_loss


# ### Discriminator Loss

# In[11]:


def discriminatorLoss(d1Real, d1Fake, gradPenalty):
    return (-torch.mean(d1Fake) + torch.mean(d1Real)) - (LAMBDA*gradPenalty)


# ## Generator Pretraining

# In[ ]:


# trained on the first 2250 images
batches_done = 0
running_loss1 = 0.0
running_loss2 = 0.0
running_losslist1 = []
running_losslist2 = []
for epoch in range(NUM_EPOCHS):
    for i,  (target, input) in enumerate(trainLoader1, 0):
#         print(i)
#         print(target[0].shape)
#         print(input[0].shape)
        unenhanced_image = input[0]
        enhanced_image = target[0] 
        unenhanced = Variable(unenhanced_image.type(Tensor_gpu))
        enhanced = Variable(enhanced_image.type(Tensor_gpu))
        
        optimizer_g1.zero_grad()
        
        generated_enhanced_image = generator1(unenhanced)
        loss1 = torch.log10(criterion(generated_enhanced_image, enhanced))
        loss1.backward()
        optimizer_g1.step()

        generated_unenhanced_image = generator2(enhanced)
        loss2 = torch.log10(criterion(generated_unenhanced_image, unenhanced))
        loss2.backward()
        optimizer_g2.step()

        # print statistics
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        running_losslist1.append(loss1.item())
        running_losslist2.append(loss2.item())
        f = open("two_logPreTraining.txt","a+")
        f.write('[%d, %5d] loss1: %.5f  loss2: %.5f\n'  % (epoch + 1, i + 1, loss1.item(), loss2.item()))
        f.close()

        if i % 225 == 224:    # print every 2000 mini-batches
            print('[%d, %5d] loss1: %.5f  loss2: %.5f\n'  % (epoch + 1, i + 1, running_loss1 / 225, running_loss2 / 225))
            running_loss1 = 0.0
            running_loss2 = 0.0
            torch.save(generator1.state_dict(), './two_gan1_pretrain'+ str(i) +'.pth')
            torch.save(generator2.state_dict(), './two_gan2_pretrain'+ str(i) + '.pth')

f = open("log_twoPreTrainingLossList1.txt","a+")
for item in running_losslist1:
    f.write('%d\n' % (item))
f.close()

f = open("log_twoPreTrainingLossList2.txt","a+")
for item in running_losslist2:
    f.write('%d\n' % (item))
f.close()

# ## Training Network
# 

# In[ ]:


# trained on the first 2250 images

batches_done = 0
for epoch in range(NUM_EPOCHS):
    for i, (data1, gt1, gt2, data2) in enumerate(trainLoader_full, 0):
#      for i,  (target, inp) in enumerate(trainLoader1, 0):
        input, dummy = data1
        groundTruth1, dummy = gt1
                 
        input2, dummy = data2
        groundTruth2, dummy = gt2
        
        input = Variable(input.type(Tensor_gpu))
        groundTruth2 = Variable(groundTruth2.type(Tensor_gpu))
        
        realImgs1 = Variable(groundTruth1.type(Tensor_gpu))
        realImgs2 = Variable(input2.type(Tensor_gpu))
                
        
        ### TRAIN DISCRIMINATOR
        optimizer_d.zero_grad()
        # generator 1
        fakeImgs1 = generator1(input)
        res_learn_out1 = fake_imgs1 + input
        realValid1 = discriminator(realImgs1)
        fakeValid1 = discriminator(res_learn_out1)
        gradientPenalty1 = computeGradientPenalty(discriminator, realImgs1.data, fakeImgs1.data)
        dLoss1 = discriminatorLoss(realValid1, fakeValid1, gradientPenalty1)
        
        # generator 2
        fakeImgs2 = generator2(groundTruth2)
        res_learn_out2 = fake_imgs2 + groundTruth2
        realValid2 = discriminator(realImgs2)
        fakeValid2 = discriminator(res_learn_out2)
        gradientPenalty2 = computeGradientPenalty(discriminator, realImgs2.data, fakeImgs2.data)
        dLoss2 = discriminatorLoss(realValid2, fakeValid2, gradientPenalty2)

        dLoss = (dLoss1 + dLoss2)/2
        dLoss.backward()

        optimizer_d.step()
        optimizer_g1.zero_grad()
        optimizer_g2.zero_grad()
        
        if batches_done % 50 == 0:
            ### TRAIN FORWARD
            print("Training Generator on Iteration: %d" % (i))
            # Generate a batch of images
            generator1.changeDirection(1)
            generator2.changeDirection(1)
            fake_imgs_g1f = generator1(input)
            res_out_g1f = fake_imgs_g1f + input
            gLoss1 = computeGeneratorLoss(input,res_out_g1f)

            # generator 2
            fake_imgs_g2f = generator2(res_out_g1)
            res_out_g2f = fake_imgs_g2f + res_out_g1f
            consistency1 = computeConsistencyLoss(input, res_out_g2f)

            ### TRAIN BACKWARD
            print("Training Generator on Iteration: %d" % (i))
            #change the batch normalizations
            generator1.changeDirection(2)
            generator2.changeDirection(2)
            # Generate a batch of images
            fake_imgs_g2b = generator2(groundTruth2)
            res_out_g2b = fake_imgs_g2b + groundTruth2
            gLoss2 = computeGeneratorLoss(groundTruth2,res_out_g2b)

            # generator 1
            fake_imgs_g1b = generator1(res_out_g2b)
            res_out_g1b = fake_imgs_g1b + res_out_g2b
            consistency2 = computeConsistencyLoss(groundTruth2, res_out_g1b)

            #losses
            gLoss = gLoss1 + gLoss2 + 10*ALPHA*(consistency1+ consistency2)
            gLoss.backward()
            optimizer_g1.step()
            optimizer_g2.step()
            
    
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: |%f] [G1 loss: %f] [G2 loss: %f] [G loss: %f]\n" % (epoch, NUM_EPOCHS , i, len(trainloader2_inp), dLoss.item(), gLoss1.item(), gLoss2.item(), gLoss.item()))
            f = open("two_logStatus.txt","a+")
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: |%f] [G1 loss: %f] [G2 loss: %f] [G loss: %f]\n" % (epoch, NUM_EPOCHS , i, len(trainloader2_inp), dLoss.item(), gLoss1.item(), gLoss2.item(), gLoss.item()))
            f.close()
            
            if batches_done % 200 == 0:
                save_image(res_learn_out1.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                save_image(res_learn_out2.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                torch.save(generator1.state_dict(), './two_gan1'+ str(i)+'.pth')
                torch.save(generator2.state_dict(), './two_gan2'+ str(i) + '.pth')
                torch.save(discriminator.state_dict(), './two_discriminator'+ str(i) + '.pth')
            
        batches_done += 1
        print("Done training generator on iteration: %d" % (i))
