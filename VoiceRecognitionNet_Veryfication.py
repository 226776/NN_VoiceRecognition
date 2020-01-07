#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa as lr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot
from tkinter import filedialog


class VoiceSamples(Dataset):
    
    def __init__(self, core_name, samples_path=None, Automatic=None):
        
        self.Log = logging.getLogger()
        #logging.basicConfig(level=logging.INFO)
        
        self.noiseThreshold = 1
        
        self.core_name = core_name
        self.samples_path = samples_path
        
        self.soundSamples = []
        self.sampleRate = []
        self.path = []
        
        self.chopedSamples = []
        self.chopedSr = []
        
        self.tensorMelgrams = []
        
        
        self.info = " VoiceSamples Object successfully created "
        self.Log.info(self.info)
        
        
        if Automatic:
            self.LoadSoundSamples()
            self.ChopToOneSecFragments()
            self.ChopedSignalsToTenosor()
        
    def __len__(self):
        return len(self.tensorMelgrams)
    
    def __getitem__(self, idx):
        if self.tensorMelgrams:
            return self.tensorMelgrams[idx]

    def LoadSoundSamples(self):
    
        n = 1

        while(True):
            try:
                if  self.samples_path:
                    path =  self.samples_path + self.core_name + str(n)
                else:
                    path = self.core_name + str(n)

                soundSample, sampleRate = lr.load(path)

                n += 1
                self.soundSamples.append(soundSample)
                self.sampleRate.append(sampleRate) 
                self.path.append(path)

                self.info = " Sample : " + path + " : successfully added"
                self.Log.info(self.info)

            except FileNotFoundError:
                if self.soundSamples:
                    self.info = "That's the end of database : " + str(n-1) + " : Samples added"
                    self.Log.info(self.info)
                    n = 0
                    
                    return self.soundSamples, self.sampleRate, self.path

                else:
                    self.Log.exception("Files are missing")
                    n = 0

                break

            except Exception as ex:      
                self.Log.exception("Unexpected error")
                break
        
    def getSoundSample(self, idx):
        return self.soundSamples[idx], self.sampleRate[idx]
    
    def getSoundSampleLen(self):
        try:
            if len(self.soundSamples) == len(self.sampleRate):
                return len(self.soundSamples)
            else:
                self.Log.warning("Lists: sundSamples and sampleRate are not equal!")
                
        except Exception as e:
            self.Log.exception("Unexpected error" + e)
    
    def ChopToOneSecFragments(self):
        
        # TODO: make shure user goes step by step 
        
        try:
            if len(self.soundSamples) == len(self.sampleRate):
                for idx in range(len(self.soundSamples)):
                    
                    soundSample = self.soundSamples[idx]
                    sr = self.sampleRate[idx]
                    
                    frag_max = math.trunc(len(soundSample)/float(sr))
                    step = math.trunc(sr/2);
                    last_sample = len(soundSample)

                    for frag in range(frag_max*2):
                        start = step * frag
                        stop = start + sr
                        if sr<len(soundSample):
                            if self.checkIfNotNoise(soundSample[start:stop]):
                                self.chopedSamples.append(soundSample[start:stop])
                                self.chopedSr.append(sr)
                                self.info = self.path[idx] + " : " + str(frag+1) + " : successfully choped"
                                self.Log.info(self.info)
                            else:
                                self.info = self.path[idx] + " : " + str(frag+1) + " : NOISE!"
                                self.Log.info(self.info)
                        else:
                            self.Log.warning("Something went wrong")
                            
                    if self.checkIfNotNoise(soundSample[last_sample-sr:last_sample]):
                         # incuding samples cuted by math.trunc() 
                        self.chopedSamples.append(soundSample[last_sample-sr:last_sample])
                        self.chopedSr.append(sr)
                        self.info = self.path[idx] +  " : "  + str(frag_max*2+1) + " : successfully choped"
                        self.Log.info(self.info)
                    else:
                        self.info = self.path[idx] + " : "  + str(frag+1) + " : NOISE!"
                        self.Log.info(self.info)
                
                if self.chopedSamples:
                    self.Log.info("Sucessfully choped all loaded signals and eliminated the noise!")
                    return self.chopedSamples, self.chopedSr 
                    
            else:
                self.Log.warning("Lists: sundSamples and sampleRate are not equal!")
                
        except Exception as e:
            self.e = "Unexpected error : " + str(e)
            self.Log.exception(self.e)
            
    def getChoped(self, idx):
        return self.chopedSamples[idx], self.chopedSr[idx]
        
    def getChopedLen(self):
        try:
            if len(self.chopedSamples) == len(self.chopedSr):
                    return len(self.chopedSamples)
            else:
                self.Log.warning("Lists: sundSamples and sampleRate are not equal!")
                
        except Exception as e:
            self.Log.exception("Unexpected error" + e)
            
        
    def ChopedSignalsToTenosor(self):
        
        # TODO: make shure user goes step by step 
        
        try:
        
            if len(self.chopedSamples) == len(self.chopedSr):
                for idx in range(len(self.chopedSamples)):

                    # hop length adjusted
                    STFT_signal = np.abs(lr.stft(self.chopedSamples[idx], n_fft = 512, hop_length = round(self.chopedSr[idx]/256))) 
                    STFT_signal = lr.power_to_db(STFT_signal**2,ref=np.max)

                    Melgram = STFT_signal[0:256,0:256]
                    TMelgram = torch.tensor(Melgram)
                    self.tensorMelgrams.append(TMelgram)
                    
                    self.info = " " + self.samples_path +  " : ChopedSample " + str(idx) + " : " + " : converted to tensor"
                    self.Log.info(self.info)
                
                if self.tensorMelgrams:
                    self.Log.info("Sucessfully converted all ChopedSamples to Tensors!")
                    return self.tensorMelgrams
                
            else:
                self.Log.warning("Lists: chopedSamples and chopedSr are not equal!")
                
        except Exception as e:
            self.e = "Unexpected error : " + str(e)
            self.Log.exception(self.e)
                
    
    
    def checkIfNotNoise(self, chopedSample):
    
        chopedSamplePow2 = []

        for n in range(len(chopedSample)):
            chopedSamplePow2.append(chopedSample[n]**2)
        sk = sum(chopedSamplePow2)
        if sk > self.noiseThreshold:
            return True 
        else:
            return False


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceRecogModel(nn.Module):

    def __init__(self):
        super(VoiceRecogModel, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20*62*62, 2000)  # ?? from image dimension
        self.fc2 = nn.Linear(2000, 300)
        self.fc3 = nn.Linear(300, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    # create parrter recognition model 

PATH = filedialog.askdirectory()
PATH = PATH + "/"
net = VoiceRecogModel()
net.load_state_dict(torch.load(PATH))
net.eval()



# loading test samples of Krystians Voice
vsKrystianTest = VoiceSamples("vsKrystianTest", samples_path="database/Test/" , Automatic=True)


K = 0 # Recognizes as Krystian
N = 0 # Recognizes as Nicia

# Recognition loop

for k in range(len(vsKrystianTest)):  
    
    vs = vsKrystianTest[k]
    
    input = vs.view(-1,1,256,256)
    output = net(input)
    v,i = output[0].max(0)
    if int(i) == 0:
        K += 1
    else:
        N += 1
    
print("Wynik rozpoznawania!")
print("Krystian:",K," Nicia:",N)
    


# ## Weryfikcja głosu nr 2
# 
# W drugim podejściu próbka należy do Nicia.
# 
# Rezyltat jest równie satsfakcjonujący: na __24__ uciętych fragmętów z kilku sekundowej próbki __23__ zostało rozpoznanych jako głos Nici.

# In[ ]:


# loading test samples of Nicia's Voice
vsNiciaTest = VoiceSamples("vsNiciaTest", samples_path="database/Test/" , Automatic=True)


K = 0 # Recognizes as Krystian
N = 0 # Recognizes as Nicia

# Recognition loop

for k in range(len(vsNiciaTest)):  
    
    vs = vsNiciaTest[k]
    
    input = vs.view(-1,1,256,256)
    output = net(input)
    v,i = output[0].max(0)
    if int(i) == 0:
        K += 1
    else:
        N += 1
    
print("Wynik rozpoznawania!")
print("Krystian:",K," Nicia:",N)


# 

# In[ ]:


# example outputs 

input = vsKrystianTest[5].view(-1,1,256,256)
output = net(input)

print("Przykładowe wyjście nauczonej sieci dla próbki Krystiana:")
print(output)

input = vsNiciaTest[5].view(-1,1,256,256)
output = net(input)

print("Przykładowe wyjście nauczonej sieci dla próbki Nici:")
print(output)


# In[ ]:




