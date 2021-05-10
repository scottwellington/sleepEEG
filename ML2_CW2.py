
##############################################################################################
### Instructions of the argv parameters need to run this scirpt are included at the bottom ###
##############################################################################################

# convenient logging imports
import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',level=logging.INFO, stream=sys.stdout)

# Functional imports
import mne
import os
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import braindecode
import sklearn
import pdb
import random
import re
import sys
import argparse
import time
import warnings
import itertools
import ast
import pyedflib
import GPUtil

# Utility imports
from mne.io import concatenate_raws, read_raw_edf
from datetime import datetime
from pprint import pprint
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.utils import _pair, _triple
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from io import StringIO
from mne.time_frequency import psd_welch
from scipy.signal import butter, filtfilt

# Architectures

class FCN(nn.Module):

	def __init__(self,
				layers=1,
				input_size=1,
				output_size=1,
				dropout=.5
				):

		super(FCN, self).__init__()
		input = input_size[0]*input_size[1]
		output = int((output_size[0] + input)/2)
		d = OrderedDict()
		idx = 0
		for i in range(layers-1):
			d[str(idx)] = nn.Linear(input, output)
			d[str(idx+1)] = nn.BatchNorm1d(output)
			d[str(idx+2)] = nn.LeakyReLU()
			d[str(idx+3)] = nn.Dropout(dropout)
			input = output
			output = int((output_size[0] + input)/2)
			idx += 4
		d[str(idx)] = nn.Linear(input, output_size[0])
		d[str(idx+1)] = nn.Softmax() # different nonlinearity in last layer
		self.feedforward = nn.Sequential(d)
	
	def forward(self, x):

		x = x.view(x.size(0), -1)
		x = self.feedforward(x)
		
		return x


class CNN(nn.Module):
	
	def __init__(self,
				in_channels=(1,1),
				out_channels=(1,1),
				input_size=(1,1),
				kernel_size=(1,1),
				pool_size=(1,1),
				stride=(1,1),
				dilation=(1,1),
				padding=(0,0),
				dropout=.5
				):
	
		super(CNN, self).__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.input_size = input_size
		self.kernel_size = kernel_size
		self.pool_size = pool_size
		self.stride = stride
		self.dilation = dilation
		self.padding = padding
		
		in_1, in_2, in_3 = self.in_channels
		out_1, out_2, out_3 = self.out_channels

		#Convolution 1
		self.cnn1 = nn.Conv2d(in_1, out_1, self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding)
		#Batch Normalisation 1
		self.bn1 = nn.BatchNorm2d(out_1)
		#Activation function 1
		self.leakyrelu1 = nn.LeakyReLU()
		#Max pool 1
		self.maxpool1 = nn.MaxPool2d(kernel_size=self.pool_size)
		#Convolution 2
		self.cnn2 = nn.Conv2d(in_2, out_2, self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding)
		#Batch Normalisation 2
		self.bn2 = nn.BatchNorm2d(out_2)
		#Activation function 2
		self.leakyrelu2 = nn.LeakyReLU()
		#Max pool 2
		self.maxpool2 = nn.MaxPool2d(kernel_size=self.pool_size)
		#Convolution 3
		self.cnn3 = nn.Conv2d(in_3, out_3, self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding)
		#Batch Normalisation 3
		self.bn3 = nn.BatchNorm2d(out_3)
		#Activation function 3
		self.leakyrelu3 = nn.LeakyReLU()
		#Max pool 3
		self.maxpool3 = nn.MaxPool2d(kernel_size=self.pool_size)
		#Dropout for regularization
		self.dropout = nn.Dropout(dropout)
	
		#Feedforward network following convolutions+maxpooling
		self.feedforward = nn.Sequential(		
			nn.Linear(2160,1024),
			nn.LeakyReLU(),
			nn.Dropout(dropout),
			nn.Linear(1024,512),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(512,256),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(256,128),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(128,64),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(64,32),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(32,16),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(16,8),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(8,4),
			)
			
	
	def forward(self, x):
		#Convolution 1
		out = self.cnn1(x)
		#Batch norm 1
		out = self.bn1(out)
		#Activation 1
		out = self.leakyrelu1(out)
		#Max pool 1
		out = self.maxpool1(out)
		#Convolution 2
		out = self.cnn2(out)
		#Batch norm 2
		out = self.bn2(out)
		#Activation 2
		out = self.leakyrelu2(out)
		#Max pool 2
		out = self.maxpool2(out)
		#Convolution 3
		out = self.cnn3(out)
		#Batch norm 3
		out = self.bn3(out)
		#Activation 3
		out = self.leakyrelu3(out)
		#Max pool 3
		out = self.maxpool3(out)
		#Resize
		out = out.view(out.size(0), -1)
		#Dropout
		out = self.dropout(out)
		#Feedforward
		out = self.feedforward(out)
		return out

class CNN3d(nn.Module):
	
	def __init__(self,
				in_channels=1,
				out_channels=1,
				input_size=(1,1),
				kernel_size=(1,1),
				pool_size=(1,1),
				stride=(1,1),
				dilation=(1,1),
				padding=(0,0),
				dropout=.5
				):
	
		super(CNN3d, self).__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.input_size = input_size
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.padding = padding
		self.pool_size = pool_size
		
		in_1, in_2, in_3 = self.in_channels
		out_1, out_2, out_3 = self.out_channels
		
		#Convolution 1
		self.cnn1 = nn.Conv3d(in_1, out_1, self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding)
		#Batch Normalisation 1
		self.bn1 = nn.BatchNorm3d(out_1)
		#Activation function 1
		self.leakyrelu1 = nn.LeakyReLU()
		#Max pool 1
		self.maxpool1 = nn.MaxPool3d(kernel_size=self.pool_size)	
		#Convolution 2
		self.cnn2 = nn.Conv3d(in_2, out_2, self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding)
		#Batch Normalisation 2
		self.bn2 = nn.BatchNorm3d(out_2)
		#Activation function 2
		self.leakyrelu2 = nn.LeakyReLU()
		#Max pool 2
		self.maxpool2 = nn.MaxPool3d(kernel_size=self.pool_size)
		#Convolution 3
		self.cnn3 = nn.Conv3d(in_3, out_3, self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding)
		#Batch Normalisation 3
		self.bn3 = nn.BatchNorm3d(out_3)
		#Activation function 3
		self.leakyrelu3 = nn.LeakyReLU()
		#Max pool 3
		self.maxpool3 = nn.MaxPool3d(kernel_size=self.pool_size)
		#Dropout for regularization
		self.dropout = nn.Dropout(dropout)

		#Feedforward network following convolutions+maxpooling
		self.feedforward = nn.Sequential(		
			nn.Linear(3312,1024),
			nn.LeakyReLU(),
			nn.Dropout(dropout),
			nn.Linear(1024,512),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(512,256),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(256,128),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(128,64),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(64,32),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(32,16),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(16,8),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(8,4),
			)

	def forward(self, x):
		x = x.unsqueeze(1)
		#Convolution 1
		out = self.cnn1(x)
		#Batch norm 1
		out = self.bn1(out)
		#Activation 1
		out = self.leakyrelu1(out)
		#Max pool 1
		out = self.maxpool1(out)
		#Convolution 2
		out = self.cnn2(out)
		#Batch norm 2
		out = self.bn2(out)
		#Activation 2
		out = self.leakyrelu2(out)
		#Max pool 2
		out = self.maxpool2(out)
		#Convolution 3
		out = self.cnn3(out)
		#Batch norm 3
		out = self.bn3(out)
		#Activation 3
		out = self.leakyrelu3(out)
		#Max pool 3
		out = self.maxpool3(out)
		#Resize
		out = out.view(out.size(0), -1)
		#Dropout
		out = self.dropout(out)
		#Feedforward
		out = self.feedforward(out)
		return out


class CNNLSTM(nn.Module):
	
	def __init__(self,
				in_channels=1,
				out_channels=1,
				input_size=(1,1),
				kernel_size=(1,1),
				padding=(1,1),
				layers=1,
				dropout=.5,
				):
		
		super(CNNLSTM, self).__init__()
		
		self.__dict__.update(locals())
		del self.self
		
		in_1, in_2 = self.in_channels
		out_1, out_2 = self.out_channels
		
		self.conv1 = nn.Conv2d(in_1, out_1, kernel_size=self.kernel_size,padding=self.padding)
		self.conv2 = nn.Conv2d(in_2, out_2, kernel_size=self.kernel_size,padding=self.padding)
		self.conv2_drop = nn.Dropout2d(self.dropout)
		#self.fc1 = nn.Linear(320, 50)
		#self.fc2 = nn.Linear(50, 10)
		self.rnn = nn.LSTM(input_size=72, hidden_size=48, num_layers=self.layers, batch_first=True)
		self.feedforward = nn.Sequential(
			nn.Linear(48,32),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(32,16),
			nn.LeakyReLU(),
			nn.Dropout(dropout),			
			nn.Linear(16,8),
			nn.LeakyReLU(),
			nn.Dropout(dropout),
			nn.Linear(8,4),
			)

	def forward(self, x):

		x = x.unsqueeze(-1)
		#batch_size, timesteps, C, H, W = x.size()
		batch_size, H, W, timesteps, C = x.size()
		#x = x.view(batch_size * timesteps, C, H, W)
		x = x.view(batch_size * timesteps, H, W, C)
		x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
		x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		# x = F.relu(self.fc1(x))
		# x = F.dropout(x, training=self.training)
		# x = self.fc2(x)
		# return F.log_softmax(x, dim=1)
		x = x.view(batch_size, timesteps, -1)
		x, (h_n, h_c) = self.rnn(x)
		x = self.feedforward(x[:, -1, :])
		return x

class Models:

	def __init__(self, **parameters):
		
		if parameters:
			self.model = model = parameters['model']
			chans = len(parameters['channels'])
			freqs = len(parameters['frequency_bands'])
			time = parameters['window_size']
			crop = parameters['crop_size']
			drop = parameters['dropout']

		else: # initial call with values==0 prior to args
			model=chans=freqs=time=crop=drop = 0
	
		self.models = {
			'FCN':{
				'layers':5,
				'input_size':(crop,chans*freqs),
				'output_size':(crop,),
				'dropout':drop,
				},
			'CNN':{
				'in_channels':(chans,chans*2), #(layer 1, layer 2)
				'out_channels':(chans*2,chans*3), #(layer 1, layer 2)
				'input_size':(chans,freqs,time),
				'kernel_size':_pair(3),
				'pool_size':_pair(2),
				'stride':_pair(1),
				'dilation':_pair(1),
				'padding':_pair(1),
				'dropout':drop,
				},
			'CNN3d':{
				'in_channels':(chans,chans*2), #(layer 1, layer 2)
				'out_channels':(chans*2,chans*3), #(layer 1, layer 2)
				'input_size':(chans,freqs,time,1),
				'kernel_size':(3,3,1), # depth, height, width,
				'pool_size':_triple(2),
				'stride': _triple(1),
				'dilation':_triple(1),
				'padding':_triple(1),
				'dropout':drop,
				},
			'CNNLSTM':{
				'in_channels':(chans,chans*2), # cnn layer 1, cnn layer 2,
				'out_channels':(chans*2,chans*3), # cnn layer 1, cnn layer 2,
				'input_size':(time,freqs,8,16),
				'kernel_size':_pair(3),
				'padding':_pair(2),
				'layers':3,
				'dropout':drop,
				},
		}
	
		for i in parameters:
			if i in self.models[model] and parameters[i]: # if parameter evaluates to True:
				self.models[model][i] = parameters[i] # overwrite (with arg flag); otherwise, use default value
	
	def get_model(self):

		model = eval(self.model)(**self.models[self.model]) # call class instance with its parameters

		return self.models[self.model]['input_size'], model



class Transform:
	
	'''
	Takes the physionet EEG sleep stage data and does online preprocessing.
	Note that this is pretty bad practice for signal processing: much better
	to have the files preprocessed and saved, and run those batches through
	the network with a simple np.load. However, for our purposes, this will do.
	'''	

	def __init__(self,
		model_name = None,
		channels_transform = False,
		frequency_bands_transform = False,
		eeg_indices = []
		):
	
		self.model_name = model_name
		self.channels_transform = channels_transform # either all channels, or EEG channels only
		self.frequency_bands_transform = frequency_bands_transform # either all frequency bands, or high gamma only
		self.eeg_indices = lambda x: np.array([x[i,:,:] for i in eeg_indices]) # legacy; just in case the channels are unlabelled and we need to select these by index only

		# Map the channel (sensor) names to their data types
		# (just ensures MNE knows know to process these properly):
		self.mapping = {
				'EEG Fpz-Cz': 'eeg',
				'EEG Pz-Oz': 'eeg',
				'EOG horizontal': 'eog',
				'Resp oro-nasal': 'resp',
				'EMG submental': 'emg',
				'Temp rectal': 'misc',
				'Event marker': 'stim',
				}

		# reminder of order: ['delta','theta','alpha','sigma','beta','gamma']

		self.freq_bands = {
				"delta": [0.5, 4.5],
				"theta": [4.5, 8.5],
				"alpha": [8.5, 11.5],
				"sigma": [11.5, 15.5],
				"beta": [15.5, 32.5],
				"gamma": [32.5, 49.5],
				}


		# From annotations.description, we can see that we have 7 classes. One of these classes
		# is the 'unknown' class ('Sleep stage ?'), so we'll disregard this and assign the others:
		self.event_ids = {'Sleep stage W': 0,
				'Sleep stage 1': 1,
				'Sleep stage 2': 1,
				'Sleep stage 3': 2,
				'Sleep stage 4': 2,
				'Sleep stage R': 3,
				}

		self.fs = 100 # hardcode cheat; we already know the data is at 100 Hz
	
		self.window_transforms = {
		'CNN':lambda X,y : (X, y), # D x H x W = sensors x freqbands x samples
		'CNN3d':lambda X,y : (np.moveaxis(np.split(X,np.shape(X)[1],axis=1),[0,1,2,3],[1,0,3,2]), y), # 128x5x256x1
		#'Deep4Net':lambda X,y : (np.expand_dims(X.T,-1),y),
		'CNNLSTM':lambda X,y : (np.moveaxis(np.split(X,int(np.shape(X)[2]/2)),[0,1,2,3],[1,3,2,0]), y), # 256x5x2x64
		}
		self.cropped_window_transforms = {
		'CNN':lambda X,y,_z : (X,y), # D x H x W = sensors x freqbands x samples
		'CNN3d':lambda X,y,_z : (X,y), # D x H x W = sensors x freqbands x samples
		'FCN':lambda X,y,_z : (X,y),
		}

	def window_transform(self, input, target):
		
		input = np.split(input,5,axis=1) # separate out the frequency bands
		input = np.moveaxis(input, [0,1,2], [1,2,0]) # change to (channels x frequency bands x samples)
		
		if self.channels_transform: # extract brocas area
			input = self.brocas_area(input)
		if self.frequency_bands_transform: # extract high gamma
			input = np.swapaxes([input[:,-1,:]],0,1)

		target = np.reshape(target,-1)
		
		if self.model_name not in self.window_transforms:
			X, y = input.reshape(-1), target
		else:
			X, y = self.window_transforms[self.model_name](input, target)

		return X, y
	
	def cropped_window_transform(self, input, target, **sliding):
	
		X_start = sliding['idx'] * sliding['scaling_factor']
		X_end = (sliding['idx'] + sliding['length']) * sliding['scaling_factor']
		y_start = sliding['idx']
		y_end = sliding['idx'] + sliding['length']

		input, target = input[:,:,:,X_start:X_end], target#[y_start:y_end]
		
		X, y = self.cropped_window_transforms[self.model_name](input, target, sliding['length'])
		
		return X, y
	
	def epoch(self, input, target):

		eeg = mne.io.read_raw_edf(input) #, preload=True, stim_channel='auto')
		annotations = mne.read_annotations(target)

		eeg.set_annotations(annotations, emit_warning=False)
		eeg.set_channel_types(self.mapping)

		# Top-and-tail the waking data to remove class immbalances. We'll only keep 1/100th of this class:

		first100 = annotations.onset[1]/100
		last100 = (annotations.onset[-1]-annotations.onset[-2])/100
		annotations.crop(annotations[1]['onset'] - first100, annotations[-2]['onset'] + last100)

		eeg.set_annotations(annotations, emit_warning=False) # Set the event ids:

		# EPOCHING! First use MNE's built-in convenience function to epoch (chunk) the data according to the event:
		events = mne.events_from_annotations(eeg, event_id=self.event_ids, chunk_duration=30.)

		# Then for the sake of getting a classification report later, we merge sleep stages 3 & 4:
		self.event_ids_merged = {'Sleep stage W': 0,
				'Sleep stage 1/2': 1,
				'Sleep stage 3/4': 2,
				'Sleep stage R': 3,
				}

		# self.event_ids_merged = {'Sleep stage W': 0,
		# 		'Sleep stage 1': 1,
		# 		'Sleep stage 2': 2,
		# 		'Sleep stage 3': 3,
		# 		'Sleep stage 4': 4,
		# 		'Sleep stage R': 5,
		# 		}

		deltas = [abs(j-i) for i,j in zip(annotations.onset, annotations.onset[1:])] # List of time differences between all sleep stages
		tmax = 30. - 1. / eeg.info['sfreq'] # Only chunk in steps of 0 to 29.99 secs

		picks = 'eeg' if self.channels_transform else ['eeg','eog','emg','resp','misc']

		epochs = mne.Epochs(raw=eeg, events=events[0], event_id=self.event_ids_merged,
			tmin=0., tmax=tmax, baseline=None, picks=picks) # Get epochs:

		return epochs

	def butter_bandpass_filter(self, data, lowcut, highcut, order=5):
		# Adapted from https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
		nyq = 0.5 * self.fs
		low = lowcut / nyq
		high = highcut / nyq
		b, a = butter(order, [low, high], btype='band')
		y = filtfilt(b, a, data, axis=0)
		return y

	def subband_decomposition(self, eeg_data):
		""" Takes in a single subject dictionary (where the keys are runs)
		Returns a dictionary of dictionaries, where the keys of the sub-
		dictionaries are EEG bands.
		"""
		filtered_data = []

		for filter in self.freq_bands.keys():
			low = self.freq_bands[filter][0]
			high = self.freq_bands[filter][1]
			filtered_data.append(self.butter_bandpass_filter(eeg_data,low,high))

		filtered_data = np.array(filtered_data)
		filtered_data = np.moveaxis(filtered_data,[0,1,2,3],[2,0,1,3]) # reshape data to be (eventsxsensorsxfreqsxtime)
		#filtered_data = np.expand_dims(filtered_data, axis=0)
		# note: reshape data to be (eventsxfreqsxsensorsxtime) = [1,0,2,3]
		# note: reshape data to be (eventsxsensorsxfreqsxtime) = [2,0,1,3]

		if len(args.frequency_bands)==1:
			filtered_data = filtered_data[:,:,[*self.freq_bands.keys()].index(args.frequency_bands[0]),:] # just extract the signal of interest

		return(filtered_data)

	def power_spectral_decomposition(self, eeg_data):

		psds, freqs = psd_welch(eeg_data, fmin=0.5, fmax=49.5) # Welch PSD (EEG only)
		psds /= np.sum(psds, axis=-1, keepdims=True) # Normalize the PSDs

		X = []

		for fmin, fmax in self.freq_bands.values():
			psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
			X.append(psds_band.reshape(len(psds), -1))

		return np.concatenate(X, axis=1)

class Dataset(Dataset):
	
	def __init__(self, input_paths, target_paths, transform=False):
		
		self.input_paths = input_paths
		self.target_paths = target_paths
		self.transform = transform
		
	def __len__(self): # no. of samples	
		
		return len(self.input_paths)
	
	def __getitem__(self, index): # as ndarray
	

		X, y = self.input_paths[index], self.target_paths[index]

		#if transform: # retained as placeholder

		return X, y
	
class EarlyStopping:

	def __init__(self, patience=20):
		
		self.patience = patience
		self.patience_count = 0
		self.bar_for_val_loss = 1e50
		self.out_of_patience = False
		
	def __call__(self, current_val_loss, model, models_dir, model_name):
		
		if current_val_loss <= self.bar_for_val_loss:
			self.bar_for_val_loss = current_val_loss
			self.patience_count = 0
		else:
			self.patience_count += 1
			if self.patience_count == self.patience:
				#save_path = os.path.join(models_dir, '{}_checkpoint.h5'.format(args.build))
				#torch.save(model.state_dict(), save_path)
				self.out_of_patience = True
	
class Network:

	def __init__(self):
		
		self.train_loss = []
		self.val_loss = []

	def train(self, input, target):
		input, target = Variable(input), Variable(target)

		input = input.float() # HOTFIX - change from double to float

		# =====================forward======================
		decoded = model(input.cuda(args.gpu)) if cuda else model(input)	# run batched input through the network
		loss = loss_function(decoded, target.cuda(args.gpu)) 				# loss function defined below (MSE)
		# =====================backward=====================
		loss.backward() 									# backpropagate and compute new gradients
		optimizer.step() 									# apply updated gradients
		# =======================log========================
		optimizer.zero_grad() 								# reset gradients from earlier training step
		return loss.item()									# training loss

	def validate(self, input, target):
		input, target = Variable(input), Variable(target)

		input = input.float() # HOTFIX - change from double to float

		#if cuda:
		#	input.cuda()
		# =====================forward======================
		decoded = model(input.cuda(args.gpu)) if cuda else model(input)		# run batched input through the network
		loss = loss_function(decoded, target.cuda(args.gpu)) 				# loss function defined below (MSE)
		# =======================log========================
		return loss.item()									# validation loss

	def run(self):
		for epoch in range(args.epochs):
			train_loss, val_loss = [], []	# store training and validation loss over batch iterations
			
			# =====================TRAINING=====================
			model.train() 											# Set model for training
			for input, target in train_dataloader:

				# =============NON-CONVOLUTIONAL MODEL==============
				if args.crop:
					idx = 0
					for i in range(int(np.floor((args.window_size-args.crop_size)/args.crop_shift))+1):
						_input, _target = transform.cropped_window_transform(input, target, idx=idx, scaling_factor=sample_length, length=args.crop_size)
						train_loss.append(self.train(_input, _target))
						idx += args.crop_shift
				
				# ===============CONVOLUTIONAL MODEL===============		
				else:
					train_loss.append(self.train(input, target))		

			# ====================VALIDATION====================
			model.eval()											# Set model to evaluation for validation
			
			for input, target in val_dataloader:
				if args.crop:
					idx = 0
					for i in range(int(np.floor((args.window_size-args.crop_size)/args.crop_shift))+1):
						_input, _target = transform.cropped_window_transform(input, target, idx=idx, scaling_factor=sample_length, length=args.crop_size)
						val_loss.append(self.validate(_input, _target))
						idx += args.crop_shift
				else:
					val_loss.append(self.validate(input, target))	
			# ========================LOG=======================
			
			log = 'epoch [{}/{}]\ntraining loss: {:.10f}\nvalidation loss: {:.10f}\n'
			print(log.format(epoch + 1, args.epochs, np.average(train_loss), np.average(val_loss)))
			
			self.train_loss.append(np.average(train_loss)) # store avg train loss for later plotting
			self.val_loss.append(np.average(val_loss))  # store avg val loss for later plotting
			
			early_stopping(np.average(val_loss), model, models_dir, args.model) 			# if validation loss decreases, save model and stop training
			if early_stopping.out_of_patience:
				print('Early stopping: no validation improvement for {} epochs.'.format(args.patience))
				break
			
			scheduler.step()										# prompt scheduler step
			
		#model.load_state_dict(torch.load(os.path.join(models_dir, '{}_checkpoint.h5'.format(args.build))))			# if early stopping, load the last checkpoint

def write_to_file():

	with open("decoded.csv", 'w') as f:
		wr = csv.writer(f, lineterminator = '\n')	
		for input, target in test_dataloader:
			input, target = Variable(input), Variable(target)
			input = input.float() # HOTFIX - change from double to float
			# =====================forward======================
			decoded = model(input.cuda(args.gpu)) if cuda else model(input) # decode
			predicted, true = decoded.argmax(axis=1).cpu().numpy(), target.detach().cpu().numpy()
			# ======================write=======================
			# Change of plans: we'll simply print to screen rather than write to file:
			print("Accuracy score: {}".format(accuracy_score(true, predicted)))
			print()
			print(confusion_matrix(true, predicted))
			print()
			print(classification_report(true, predicted, target_names=transform.event_ids_merged.keys(),
				zero_division=0))
			print()
			# for i in range(len(input)):
			# 	wr.writerow(decoded[i].tolist())	# write to file

def plot_losses_over_time(model_name): # adapted from https://github.com/Bjarten/early-stopping-pytorch
	"""
	Plot losses over time for a given model. Was integrated as an argparse call, 
	but now requires a separate call (we used a different plotting in the end,
	but I'll leave this in here for future use because it's pretty useful for future work.)
	"""

	# load saved training and validation losses from the trained FCN
	d = np.load("{}_score.npy".format(args.model))
	train_loss, val_loss = [i[1] for i in d['train_loss']],[i[1] for i in d['val_loss']]
	
	# find position of lowest val loss; get max val; constrain plot
	max_val = max(train_loss+val_loss)
	min_val_idx = val_loss.index(min(val_loss))+1
	train_loss, val_loss = train_loss[:min_val_idx+5], val_loss[:min_val_idx+5]
	
	# visualize the loss as the network trained
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
	plt.plot(range(1,len(val_loss)+1),val_loss,label='Validation Loss')

	plt.axvline(min_val_idx, linestyle='--', color='r',label='Early Stopping Checkpoint')

	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.ylim(0, max_val+.5*max_val) # consistant scale
	plt.xlim(0, len(train_loss)+1) # consistant scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()
	fig.savefig('{}_loss_plot.png'.format(args.model), bbox_inches='tight')



def get_data(path, decomposition='subband'):
	"""
	path is a list of (input,target) tuples, where input is the path
	to the input data, and target is the path to the target data
	"""

	input_files, target_files = [], []
	
	for participant in path:
		if args.id:
			subject_id = re.findall(r'\d\d\d\d', participant[0])[0]
			for sub in args.id.split():
				if subject_id == sub:
					input_files.append(participant[0]) # eeg_path
					target_files.append(participant[1]) # annotations_path
		else:
			input_files.append(participant[0]) # eeg_path
			target_files.append(participant[1]) # annotations_path

	if args.debug:
		input_files, target_files = input_files[:10], target_files[:10]

	X, y = np.array([]), np.array([])

	for i in range(len(input_files)):

		try:
			epochs = transform.epoch(input_files[i], target_files[i])

			if decomposition=='subband':
				_X = transform.subband_decomposition(epochs.get_data())
			elif decomposition=='welch':
				_X = transform.power_spectral_decomposition(epochs)
			else:
				_X = epochs.get_data()  # transform from epochs object to numpy array of shape (events, channels, time)

			_y = epochs.events[:, 2]

			if not X.any():
				X = _X
			else:
				X = np.vstack((X,_X))
			if not y.any():
				y = _y
			else:
				y = np.hstack((y,_y))

			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=test_size, random_state=args.seed)
			
			X_train, X_val, y_train, y_val = train_test_split(
				X_train, y_train, train_size=train_size, random_state=args.seed)
		except:
			pass

	train_data = Dataset(X_train, y_train)
	val_data = Dataset(X_val, y_val)
	test_data = Dataset(X_test, y_test)
	
	return train_data, val_data, test_data

def get_available_device(max_memory=0.8):

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available=-1
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available 

def intrasubject_RF_classifier():

	transform = Transform()

	# Fetch the data with `wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/`
	# we won't put this as an passable argument in the script; best it stays in the readme

	path = os.path.join('.','physionet.org','files','sleep-edfx','1.0.0','sleep-cassette')

	# OK, so, assuming that this script is at '.' (i.e at the top-level of the directory),
	# the 'path' above has many pairs of EEG + annotations files. So, we're going to make
	# a list of tuples containing these.

	files = sorted(os.listdir(path))
	files.remove('index.html') # the only file we want to ignore

	psg_files = files[0::2] # polysomnogram files (electroencephalography; eeg data)
	hypnogram_files = files[1::2] # hypnogram files (expert-labelled annotation data)

	files = [*zip(psg_files, hypnogram_files)]
	# Cool, so now we've done that...

	n = len(files) # There are 153 subjects

	subject_4001 = files[0] # for now we'll just take the first 10; taking all 8gb of data may be overkill...

	eeg_path = os.path.join(path, subject_4001[0])
	annotations_path = os.path.join(path, subject_4001[1])

	# Let's get our baseline results. We'll use a Random Forest...:

	eeg = mne.io.read_raw_edf(eeg_path) #, preload=True, stim_channel='auto')
	annotations = mne.read_annotations(annotations_path)

	# Map the channel (sensor) names to their data types
	# (just ensures MNE knkows know to process these properly):
	mapping = transform.mapping

	eeg.set_annotations(annotations, emit_warning=False)
	eeg.set_channel_types(mapping)

	# Plot 1: first glance overview of the distribution of the sleep stages
	eeg.plot(duration=60, scalings='auto')

	# From annotations.description, we can see that we have 7 classes. One of
	# these classes is the 'unknown' class ('Sleep stage ?'), so we'll desregard
	# this and assign the others:

	event_ids = transform.event_ids
	# From Plot 1, we can see a hell of a lot of data for 'Sleep Stage W'
	# (i.e. the person is a awake a hell of a lot more than they are asleep).
	# Annotations.onset provides us with a list of start times for each sleep stage.
	# So, to prevent any class balance issues, we'll top-and-tail the awake data, keeping
	# only 10% of the awake data immediately before the first, and after the last,
	# sleep stage.

	first100 = annotations.onset[1]/100
	last100 = (annotations.onset[-1]-annotations.onset[-2])/100
	annotations.crop(annotations[1]['onset'] - first100, annotations[-2]['onset'] + last100)
	
	# Set the event ids:
	eeg.set_annotations(annotations, emit_warning=False)

	# events_from_annotations allows us to extract the relevant data for each
	# sleep stage for the purposes of plotting data, etc.

	events = mne.events_from_annotations(eeg, event_id=event_ids, chunk_duration=30.)

	# Odd syntax, but this is how we're going to plot the 'overview' of the data.
	fig = mne.viz.plot_events(events[0], event_id=event_ids, sfreq=eeg.info['sfreq'], first_samp=events[0][0][0])

	# Retain plotted legend colors for later plots (great utility! will be using this much more in future!)
	stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	# just for the purposes of plotting the PSD quickly, we'll only take a random 10 instances of each class:

	# store = np.array([], dtype=np.int64).reshape(0,events[0].shape[1])
	# for i in events[1]:
	# 	event = events[0][np.where(events[0][:,2]==events[1][i])]
	# 	event = event[np.random.choice(len(event), size=8, replace=False)]
	# 	store = np.vstack([store, event])

	# EPOCHING! Super important: this is how we're going to extract the data chunks corresponding
	# to the sleep stages.

	# List of time differences between all sleep stages:
	deltas = [abs(j-i) for i,j in zip(annotations.onset, annotations.onset[1:])]
	 # This is so that we only consider epochs less than the longest known stage (minus cps)
	tmax = max(deltas) - (1. / eeg.info['sfreq'])

	# Override; only chunk in steps of 0 to 29.99 secs; very important to do this so that all signal processing
	# only applies in chunks with max size 3000. Otherwise, there'll be no windowing and the process will die :-(
	# Note how the way we've implemented this means that a standard night's worth of EEG data roughly averages
	# to window sizes of ~2.5 secs.
	tmax = 30. - 1. / eeg.info['sfreq']

	# Get all epochs:
	# epochs = mne.Epochs(raw=eeg, events=events[0], event_id=event_ids, tmin=0., tmax=tmax, baseline=None)

	event_ids_merged = {'Sleep stage W': 0,
		'Sleep stage 1/2': 1,
		'Sleep stage 3/4': 2,
		'Sleep stage R': 3,
		}

	picks = ['eeg','eog','emg','resp','misc']

	#rf = RandomForestClassifier(n_estimators=100, random_state=42)
	rf = RandomForestClassifier(n_estimators=100, random_state=42)

	# we're going to be lazy and use scikit-learn's KFold to get our train and test data cross-valisations:
	kf = sklearn.model_selection.KFold(n_splits=5,random_state=42, shuffle=True)

	#events_train, events_test = sklearn.model_selection.train_test_split(events[0], test_size=.1)
	
	y_test_store = np.array([], dtype=int)
	y_pred_store = np.array([], dtype=int)
	accuracy = 0

	for train_idxs, test_idxs in kf.split(events[0]):

		events_train = events[0][train_idxs]
		events_test = events[0][test_idxs]

		# Get train/test epochs:
		epochs_train = mne.Epochs(raw=eeg,
			events=events_train,
			event_id=event_ids_merged,
			tmin=0.,
			tmax=tmax,
			baseline=None,
			picks=picks
			)
		
		epochs_test = mne.Epochs(raw=eeg,
			events=events_test,
			event_id=event_ids_merged,
			tmin=0.,
			tmax=tmax,
			baseline=None,
			picks=picks)

		# Train
		y_train = epochs_train.events[:, 2]
		y_test = epochs_test.events[:, 2]

		epochs_train = transform.subband_decomposition(epochs_train.get_data())
		epochs_train = epochs_train.reshape(epochs_train.shape[0], -1)

		epochs_test = transform.subband_decomposition(epochs_test.get_data())
		epochs_test = epochs_test.reshape(epochs_test.shape[0], -1)

		# Train
		rf.fit(epochs_train, y_train)

		# Test
		y_pred = rf.predict(epochs_test)

		# Assess the results
		accuracy += sklearn.metrics.accuracy_score(y_test, y_pred)

		y_test_store = np.hstack((y_test_store, y_test))
		y_pred_store = np.hstack((y_pred_store, y_pred))

	# Outputs:
	print()
	print("Accuracy score: {}".format(accuracy/kf.get_n_splits(events[0])))
	print()
	print(confusion_matrix(y_test_store, y_pred_store))
	print()
	print(classification_report(y_test_store, y_pred_store, target_names=event_ids_merged.keys()))
	print()

if __name__ == "__main__":

	# ==================================================
	# =============softcoded default paths==============
	# ==================================================

	# I would hope that either a UNIX or Windows machine would be OK with
	# the data being in a sister dir, but we'll leave this here just in case.
	# The joys of groupwork and not knowing what other systems are in play.

	_ = {'nt':'.','posix':'.'}
	root = _[os.name] # change root depending on operating system

	# Fetch the data with `wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/`
	# we won't put this as an passable argument in the script; best it stays in the readme

	path = os.path.join(root,'physionet.org','files','sleep-edfx','1.0.0','sleep-cassette')

	# OK, so, assuming that this script is at '.' (i.e at the top-level of the directory),
	# the 'path' above has many pairs of EEG + annotations files. So, we're going to make
	# a list of tuples containing these.

	files = sorted(os.listdir(path))
	files.remove('index.html') # the only file we want to ignore

	psg_files = files[0::2] # polysomnogram files (electroencephalography; eeg data)
	hypnogram_files = files[1::2] # hypnogram files (expert-labelled annotation data)

	files = [*zip([os.path.join(path, i) for i in psg_files],
			[os.path.join(path, i) for i in hypnogram_files])]

	# Cool, so now we've done that...

	n = len(files) # There are 153 subjects

	if not os.path.exists('models'):
		os.makedirs('models') 
	if not os.path.exists('results'):
		os.makedirs('results') 
	
	models_dir, results_dir = 'models', 'results'

	# ==================================================
	# ===============argparse options===================
	# ==================================================

	parser = argparse.ArgumentParser()
	mode = parser.add_mutually_exclusive_group()
	
	mode.add_argument("-t", "--train", action="store_true", help="train network from EEG data")
	mode.add_argument("-c", "--test", action="store_true", help="use network to classify test EEG data")
	
	parser.add_argument("--debug", action="store_true", help="debug (only use first 10 data samples)")
	
	# ============model architecture choice=============
	parser.add_argument("-m", "--model",type=str,choices=Models().models,required=True,
		help="Model architecture from one of: {}".format(', '.join(Models().models)))
	
	# ========thinker-(in)dependent model choice========
	parser.add_argument("--id", type=str,
		help="Subject ID number(s) to train thinker-dependent model (default = thinker-independent model). Available IDs are: {}".format(
			[re.findall(r'\d\d\d\d',file[0])[0] for file in files]))
	
	# =======convenience arguments for data paths=======
	parser.add_argument("-f", "--filter", type=str, choices=['single_bandpass', 'multi_bandpass'], default='multi_bandpass')
	parser.add_argument("-s", "--scope", type=str, choices=['local', 'global'], default='global')
	parser.add_argument("-n", "--normalisation", type=str, choices=['min_max', 'mean_variance'], default='mean_variance')
	parser.add_argument("-b", "--build", type=str, default='') # optional descriptor for model build
	
	# ===========parameters for convolutions============
	parser.add_argument("--in_channels", type=ast.literal_eval, default=None)# Tuple written as a string; e.g. "(2,3)"
	parser.add_argument("--out_channels", type=ast.literal_eval, default=None)# Tuple written as a string; e.g. "(2,3)"
	parser.add_argument("--kernel_size", type=ast.literal_eval, default=None)# Tuple written as a string; e.g. "(2,3)"
	parser.add_argument("--pool_size", type=ast.literal_eval, default=None)
	parser.add_argument("--stride", type=ast.literal_eval, default=None)
	parser.add_argument("--dilation", type=ast.literal_eval, default=None)
	parser.add_argument("--padding", type=ast.literal_eval, default=None)
	
	# ===============parameters for data===============
	parser.add_argument("--channels", default='all', choices=['all', 'psg_only', 'eeg_only'])
	parser.add_argument("--frequency_bands", default='all', choices=['all','delta','theta','alpha','sigma','beta','gamma'])
	parser.add_argument("--decomposition", default='subband', choices=['subband', 'welch'])
	parser.add_argument("--crop", action="store_true", default=False)
	
	# =========user-definable (hyper)parameters=========
	parser.add_argument("--window_size", type=int, default=256)
	parser.add_argument("--window_shift", type=int, default=246)	
	parser.add_argument("--crop_size", type=int, default=32)
	parser.add_argument("--crop_shift", type=int, default=16)
	parser.add_argument("--dropout", type=float, default=.2)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=1000)
	parser.add_argument("--learning_rate", type=float, default=1e-4)
	parser.add_argument("--patience", type=int, default=30)
	parser.add_argument("--seed", type=int, default=99)
	
	# ==================================================
	args = parser.parse_args()

	# =======================================================================================================
	# ============convenience variables for different participant combinations (thank you Tory!)=============
	# =======================================================================================================
	# full SC data set
	all_data = [4021,4022,4031,4032,4051,4052,4081,4082,4091,4092,4611,4612,4621,4622,4631,4632,4001,4002,4011,4012,4041,4042,4061,4062,4071,4072,4201,4202,4211,4212,4221,4222,4231,4232,4241,4242,4251,4252,4261,4262,4271,4272,4281,4282,4291,4292,4801,4802,4811,4812,4821,4822,4401,4402,4411,4412,4421,4422,4451,4452,4461,4462,4481,4482,4491,4492,4431,4432,4441,4442,4471,4472,4601,4602,4641,4642,4651,4652,4661,4662,4671,4672,4101,4102,4111,4112,4121,4122,4131,4141,4142,4181,4182,4191,4192,4731,4732,4741,4742,4751,4752,4761,4762,4151,4152,4161,4162,4171,4172,4301,4302,4311,4312,4321,4322,4341,4342,4351,4352,4362,4371,4372,4381,4382,4331,4332,4522,4531,4532,4571,4572,4581,4582,4591,4592,4501,4502,4511,4512,4541,4542,4551,4552,4561,4562,4701,4702,4711,4712,4721,4722,4771,4772]
	female_30 = [4021,4022,4031,4032,4051,4052,4081,4082,4091,4092]
	female_30_40 = [4001,4002,4011,4012,4041,4042,4061,4062,4071,4072]
	# note there is no data for people in thier 40s!
	female_40_50 = []
	female_50_60 = [4201,4202,4211,4212,4221,4222,4231,4232,4241,4242,4251,4252,4261,4262,4271,4272,4281,4282,4291,4292,4801,4802,4811,4812,4821,4822]
	female_60_70 = [4401,4402,4411,4412,4421,4422,4451,4452,4461,4462,4481,4482,4491,4492]
	female_70_80 = [4431,4432,4441,4442,4471,4472]
	female_80_90 = [4601,4602,4641,4642,4651,4652,4661,4662,4671,4672]
	female_90 = [4611,4612,4621,4622,4631,4632]

	male_30 = [4101,4102,4111,4112,4121,4122,4131,4141,4142,4181,4182,4191,4192]
	male_30_40 = [4151,4152,4161,4162,4171,4172]
	# note there is no data for people in thier 40s!
	male_40_50 = []
	male_50_60 = [4301,4302,4311,4312,4321,4322,4341,4342,4351,4352,4362,4371,4372,4381,4382]
	male_60_70 = [4331,4332,4522,4531,4532,4571,4572,4581,4582,4591,4592]
	male_70_80 = [4501,4502,4511,4512,4541,4542,4551,4552,4561,4562]
	male_80_90 = [4701,4702,4711,4712,4721,4722,4771,4772]
	male_90 = [4731,4732,4741,4742,4751,4752,4761,4762]

	# aggregations 

	females = np.array(female_30+female_30_40+female_40_50+female_50_60+female_60_70+female_70_80+female_80_90+female_90)
	males = np.array(male_30+male_30_40+male_40_50+male_50_60+male_60_70+male_70_80+male_80_90+male_90)
	under_30 = female_30+male_30
	between_30_40 = female_30_40+male_30_40
	between_40_50 = female_40_50+male_40_50
	between_50_60 = female_50_60+male_50_60
	between_60_70 = female_60_70+male_60_70
	between_70_80 = female_70_80+male_70_80
	between_80_90 = female_80_90+male_80_90
	over_90 = female_90+male_90

	total_gender = [females,males]
	min_gender = min(map(len, total_gender))
	# excluded 40-50 as that's zero
	total_age = [under_30,between_30_40,between_50_60,between_60_70,between_70_80,between_80_90,over_90]
	sum_age = sum(map(len, total_age))
	min_age = min(map(len, total_age))
	total_gender_age = [female_30,female_30_40,female_50_60,female_60_70,female_70_80,female_80_90,female_90,male_30,male_30_40,male_50_60,male_60_70,male_70_80,male_80_90,male_90]
	min_gender_age = min(map(len, total_gender_age))

	for gender in total_gender:
	    random.shuffle(gender)
	random_females = females[:min_gender]
	random_males = males[:min_gender]

	for age in total_age:
	    random.shuffle(age)
	random_under_30 = under_30[:min_age]
	random_between_30_40 = between_30_40[:min_age]
	# not including 40-50 as there are none
	random_between_50_60 = between_50_60[:min_age]
	random_between_60_70 = between_60_70[:min_age]
	random_between_70_80 = between_70_80[:min_age]
	random_between_80_90 = between_80_90[:min_age]
	random_over_90 = over_90[:min_age]

	for gender_age in total_gender_age:
	    random.shuffle(gender_age)
	random_female_30 = female_30[:min_gender_age]
	random_female_30_40 = female_30_40[:min_gender_age]
	# not including 40-50 as there are none
	random_female_50_60 = female_50_60[:min_gender_age]
	random_female_60_70 = female_60_70[:min_gender_age]
	random_female_70_80 = female_70_80[:min_gender_age]
	random_female_80_90 = female_80_90[:min_gender_age]
	random_female_90 = female_90[:min_gender_age]
	random_male_30 = male_30[:min_gender_age]
	random_male_30_40 = male_30_40[:min_gender_age]
	# not including 40-50 as there are none
	random_male_50_60 = male_50_60[:min_gender_age]
	random_male_60_70 = male_60_70[:min_gender_age]
	random_male_70_80 = male_70_80[:min_gender_age]
	random_male_80_90 = male_80_90[:min_gender_age]
	random_male_90 = male_90[:min_gender_age]

	gender_buckets = [random_females,random_males]
	# not including 40-50 as there are none
	age_buckets = [random_under_30, random_between_30_40, random_between_50_60, random_between_60_70, random_between_70_80, random_between_80_90, random_over_90]
	gender_age_buckets = [random_female_30, random_female_30_40, random_female_50_60, random_female_60_70, random_female_70_80, random_female_80_90, random_female_90, random_male_30, random_male_30_40, random_male_50_60, random_male_60_70, random_male_70_80, random_male_80_90, random_male_90]
	
	# size breaks
	random.shuffle(all_data)
	one = all_data[:1]
	random.shuffle(all_data)
	ten = all_data[:10]
	random.shuffle(all_data)
	twenty = all_data[:20]

	size_breaks = [one, ten, twenty]

	# inter/ intra

	random.shuffle(all_data)
	one_intra = all_data[:1]
	clean_age_gender = gender_age_buckets = [random_female_30, random_female_30_40, random_female_50_60, random_female_60_70, random_female_70_80, random_female_80_90, random_female_90, random_male_30, random_male_30_40, random_male_50_60, random_male_60_70, random_male_70_80, random_male_80_90, random_male_90]

	# ==================================================
	# ============hardcoded data parameters=============
	# ==================================================

	test_size = 0.1 # of all data, percentage for test set
	train_size = 0.9 # of the training data, the train/val split

	all_frequency_bands = ['delta','theta','alpha','sigma','beta','gamma']
	
	channels = {'all':['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal'],
				'psg_only':['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal'],
				'eeg_only':['EEG Fpz-Cz', 'EEG Pz-Oz']}

	# ==================================================
	# ================dynamic parameters================
	# ==================================================

	args.channels = channels[args.channels] # set channels used
	args.frequency_bands = all_frequency_bands if args.frequency_bands == 'all' else [args.frequency_bands]
	sampling_rate, sample_length = 100, 1#len(args.frequency_bands)*len(args.channels) # Hz
	
	# ==================================================
	# =================network setup====================
	# ==================================================

	# Sanity check to see if we're using the GPU or not
	# We're training everything over Hex, so it should be OK,
	# but useful also if people are wanting to play on their home computers.

	cuda = True if torch.cuda.is_available() else False

	# Initialize model
	models = Models(**vars(args))

	input_size, model = models.get_model()

	print('\nSanity check for current model params:')
	print(vars(models)['models'][vars(models)['model']])
	print('\nSanity check for current model:')
	print(model)

	#device = torch.device("cuda:1") # just leaving this here because some people were not playing 'nice'
	#model.to(device) # just leaving this here because some people were not playing 'nice'
	
	network = Network()
	
	args.gpu = get_available_device()

	if cuda:
		model.cuda(args.gpu) # Woo! Cuda! :-)

	# Define loss function for multiclass classification:
	loss_function = nn.CrossEntropyLoss()
	
	# Initialize optimizer & scheduler
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4) # these are good values for the deep models
	#optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
	restart_period = 5
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, restart_period, T_mult=1, eta_min=0, last_epoch=-1)

	# Initialize early stopping params
	early_stopping = EarlyStopping(patience=args.patience)

	# ==================================================
	# ===================data setup=====================
	# ==================================================

	transform = Transform(
		model_name = args.model,
		channels_transform = len(args.channels) < len(channels['all']),
		frequency_bands_transform = len(args.frequency_bands) < len(all_frequency_bands),
		eeg_indices = sorted([channels['all'].index(i) for i in channels['eeg_only']])
		)
	
	train_data, val_data, test_data = get_data(files, decomposition=args.decomposition) # Get data from user-specified path
	train_sampler, val_sampler = RandomSampler(train_data), SequentialSampler(val_data)	# Samplers for training and validation batches
	# ==================================================

	if args.train:
		train_dataloader = DataLoader(train_data, batch_size=args.batch_size, sampler = train_sampler) # online read of sampled minibatched training data
		val_dataloader = DataLoader(val_data, batch_size=args.batch_size, sampler = val_sampler) # online read of sampled minibatched validation data
		if os.path.isfile(os.path.join(models_dir, '{}_checkpoint.h5'.format(args.build))): # load in checkpointed model if training was inturrupted
			model.load_state_dict(torch.load(os.path.join(models_dir, '{}_checkpoint.h5'.format(args.build))))
		
		# train model
		t = time.time()
		network.run()
		t = time.time() - t

		# save model summary for visualisation
		avg_train_loss = [('epoch {}'.format(i),'{}'.format(j)) for i,j in [k for k in enumerate(network.train_loss,1)]] # convenience data storage
		avg_val_loss = [('epoch {}'.format(i),'{}'.format(j)) for i,j in [k for k in enumerate(network.val_loss,1)]] # convenience data storage
		old_stdout = sys.stdout
		sys.stdout = mystdout = StringIO()
		pprint(model)
		sys.stdout = old_stdout

		# get sample decoded output for evaluation statistics (normalised root mean square error, etc.)
		model.eval()
		
		input, target = next(iter(val_dataloader))
		input, target = Variable(input), Variable(target)
		input = input.float() # HOTFIX - change from double to float
		input, target = input.cuda(args.gpu), target.cuda(args.gpu)

		if args.crop:

			idx = 0
			#dims = models.transform_data(args.model, input, target, idx=idx, scaling_factor=sample_length, length=args.crop_size)[1].size()
			input_stack, target_stack = np.array([]), np.array([])
			for i in range(int(np.floor((args.window_size/args.crop_size)))):
				_input, _target = transform.cropped_window_transform(input, target, idx=idx, scaling_factor=sample_length, length=args.crop_size)
				_input, _target = torch.tensor(model(Variable(_input))).clone().detach().cpu().numpy(), _target.detach().cpu().numpy()
				
				if not input_stack.any():
					input_stack = _input
				else:
					input_stack = np.vstack((input_stack,_input))
				if not target_stack.any():
					target_stack = _target
				else:
					target_stack = np.hstack((target_stack,_target))
				idx += args.crop_size

			predicted, true = input_stack.argmax(axis=1), target_stack
		
		else:

			decoded = torch.tensor(model(input)).clone().detach().cpu().numpy() # detach and copy to cpu to convert to numpy array
			predicted, true = decoded.argmax(axis=1), target.detach().cpu().numpy()
		
		args.window_size, args.window_stride = args.crop_size, args.crop_shift # amend values following sample decoding


		# Save model parameters for tuning and plotting
		summary = {
			'model': args.model,
			'summary': mystdout.getvalue(),
			'channels':args.channels,
			'freqs': args.frequency_bands,
			'time': t,
			'windowsize': args.window_size,
			'windowstride': args.window_stride,
			'randomseed': args.seed,
			'epochs': len(avg_train_loss)-args.patience,
			'trainloss': avg_train_loss[:-args.patience],
			'valloss': avg_val_loss[:-args.patience],
			'trainlossall': avg_train_loss,
			'vallossall': avg_val_loss,			
			'val': avg_val_loss[:-args.patience][-1][1],
			'accuracy': accuracy_score(true, predicted),
			'batchsize': args.batch_size,
			'lr': args.learning_rate,
			'patience': args.patience,
			'dropout': args.dropout,
			'inputsize': input_size,
			'kernelsize': args.kernel_size,
			'poolsize': args.pool_size,
			'stride': args.stride,
			'dilation': args.dilation,
			'padding': args.padding,
			'filter': args.filter,
			'scope': args.scope,
			'normalisation': args.normalisation,
			}

		print("Accuracy score: {}".format(accuracy_score(true, predicted)))
		print()
		print(confusion_matrix(true, predicted))
		print()
		print(classification_report(true, predicted, target_names=transform.event_ids_merged.keys(),
			zero_division=0))
		print()

		savepath = os.path.join(results_dir, '{}_'.format(args.build))  #datetime.now().strftime('%m%d%H%M')))
		np.save(savepath + 'score', np.array([summary]))
		np.save(savepath + 'pred', predicted)
		np.save(savepath + 'true', true)

		# save final model state params, remove checkpoint
		torch.save(model.state_dict(), os.path.join(models_dir, '{}.h5'.format(args.build))) # save model if number of epochs have run to zero
		#os.remove(os.path.join(models_dir, '{}_checkpoint.h5'.format(args.build))) # remove temporary model checkpoint
		
	if args.test: # raw decoded - requires postprocessing argmax step
		test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False) # Online read of unshuffled minibatched data
		model.load_state_dict(torch.load(os.path.join(models_dir, '{}.h5'.format(args.build)))) # Load model from trained CNN
		model.eval() # Set model to evaluation before running inference
		write_to_file()




#################
### Preamble: ###
#################


# The dataset is obtainable here: https://www.physionet.org/content/sleep-edfx/1.0.0/

# The full datset is >8.1gb. Naturally, we don't expect you to download all this for
# marking purposes (and the preprocessing for each data file can take up to a minute),
# so I have randomly selected test data (as below), keeping the full package to download <50mb.

# With apologies, brain-signal processing requires some libraries: I have included a
# requirement.txt to help make this process easier (`pip install -r requirements.txt`)

# This model has been designed to work on the University of Bath's Hex compute cluster; I understand
# that all testing is to be done on this service, so this code should run without any issues.

# I used the script's own randomisation process to select a testing file (participant '4811').
# I've not run this test, so I hope you get some interesting results! Our model operates
# around a typical accuracy of ~70%.

# .py and .ipynb files are provided as requested deliverables (as per instructions).
# However, we recommend you clone the repo onto the Hex compute cluster from:

# https://github.com/scottwellington/sleepEEG/

# This will provide you with all scripts, models, and test data (including this report!)
# supplied within the expected directory structure.

# Thank you for your time: we hope you find our project interesting! We certainly did!


###################################
### Run this script as follows: ###
###################################


# To test on the randomly selected participant's data, please run the following command:

# python3 ML2_CW2.py -b model -m CNN --test --in_channels "(6,12,24)" --out_channels "(12,24,48)" --kernel_size "(1,6)" --pool_size "(1,2)" --stride 2 --dropout .2 --padding "(0,0)" --id "4811"



# To train on the randomly selected participant's data, please run the following command:

# python3 ML2_CW2.py -b model -m CNN --train --in_channels "(6,12,24)" --out_channels "(12,24,48)" --kernel_size "(1,6)" --pool_size "(1,2)" --stride 2 --dropout .2 --padding "(0,0)" --id "4811"
