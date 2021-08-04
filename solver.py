import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network3D import U_Net3D, U_Net3D_Att
import csv
import pandas as pd
from data_loader import get_evalutation_loader
import random
import os
import nibabel as nib
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.model_type = config.model_type

		if self.model_type in ['U_Net_Seg']:
			self.criterion = FocalLoss()
		else:
			self.criterion = torch.nn.MSELoss(reduction='mean')

		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		
		# Training settings
		self.num_epochs = config.num_epochs 
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size
		self.dis = config.dis
		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step
		self.test_only = config.test_only
		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print('model:', self.model_type, 'batch_size:', self.batch_size)
		if self.model_type != 'two_step':
			self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type == 'U_Net3D':
			self.unet = U_Net3D(img_ch=1, output_ch=1)
		elif self.model_type == 'U_Net3D_V2':
			self.unet = U_Net3D(img_ch=1, output_ch=1)
		elif self.model_type =='U_Net3D_Att':
			self.unet = U_Net3D_Att(img_ch=1, output_ch=1)
		elif self.model_type == 'U_Net3D_Att_Seg':
			self.unet = U_Net3D_Att(img_ch=7, output_ch=1)
		elif self.model_type == 'U_Net3D_Att_Seg_V2':
			self.unet = U_Net3D_Att(img_ch=7, output_ch=1)
		elif self.model_type == 'U_Net_Seg':
			self.unet = U_Net3D_Att(img_ch=1, output_ch=6)


		# self.best_threshold = 0.5
		self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		load_path = unet_path

		# U-Net Train
		if os.path.isfile(load_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(load_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,load_path))
			          
		# Train for Encoder
		lr = self.lr
		best_unet_loss = 100
		
		for epoch in range(self.num_epochs):
			self.unet.train(True)
			epoch_loss = 0
			print(epoch)
			for i, (images, GT, seg) in enumerate(self.train_loader):             
				images = images.to(self.device)
				GT = GT.to(self.device)
				seg = seg.to(self.device)

				# SR : Segmentation Result
				if self.model_type in ['U_Net3D_Att_Seg','U_Net3D_Att_Seg_V2']:
					images = torch.cat((images,seg), 1)
					SR = self.unet(images)
				elif self.model_type in ['U_Net_Seg']:
					SR = self.unet(images)
				else:
					SR = self.unet(images)

				SR_flat = SR.view(SR.size(0),-1)              
				GT_flat = GT.view(GT.size(0),-1)


				if self.model_type in ['U_Net_Seg']:
					loss = self.criterion(SR, seg)
				else:
					loss = self.criterion(SR_flat,GT_flat)

				epoch_loss += loss.item()
				self.reset_grad()
				loss.backward()
				self.optimizer.step()
				# abs_diff = torch.abs(SR_flat-GT_flat).sum()      
				print('epoch: ', epoch, 'train batch number: ', i, 'training loss:', loss.item())   


			# Print the log info
			print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, self.num_epochs, epoch_loss))

			# Decay learning rate
			if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				lr -= (self.lr / float(self.num_epochs_decay))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr
				print ('Decay learning rate to lr: {}.'.format(lr))
			
				
			# ===================================== Validation ====================================#
			self.unet.train(False)
			self.unet.eval()
			validloss = torch.nn.MSELoss(reduction='mean')
			length=0
			epoch_loss = 0
			
			for i, (images, GT, seg) in enumerate(self.valid_loader):   
				images = images.to(self.device)
				GT = GT.to(self.device)
				seg = seg.to(self.device)

				# SR : Segmentation Result
				if self.model_type in ['U_Net3D_Att_Seg','U_Net3D_Att_Seg_V2']:
					images = torch.cat((images,seg), 1)
					SR = self.unet(images)
				elif self.model_type in ['U_Net_Seg']:
					# seg = seg.to(torch.int64)
					SR = self.unet(images)
				else:
					SR = self.unet(images)
				SR_flat = SR.view(SR.size(0),-1)              
				GT_flat = GT.view(GT.size(0),-1)

				if self.model_type in ['U_Net_Seg']:
					loss = self.criterion(SR, seg)
					GT = torch.argmax(seg, dim=1)
					SR = torch.argmax(F.softmax(SR, dim=1), dim=1)
					GT = GT.data.cpu().numpy()
					SR = SR.data.cpu().numpy()			
					acc = get_accuracy(SR, GT)
					dicelist = []
					for i in range(6):
						DC = get_DC(SR, GT, i)
						dicelist.append(DC)
					print('acc:', acc, 'dice:', dicelist, 'null_white_gray_CSF_bone_skin')
				else:
					loss = validloss(SR_flat,GT_flat)

				epoch_loss += loss.item()
				              
				print('epoch: ', epoch, ' valid batch number: ', i, 'validation loss:', loss.item())

			length = len(self.valid_loader)

			epoch_loss = epoch_loss/length

			print('[Validation] ', epoch, 'epoch_loss:', epoch_loss,' best_unet_loss', best_unet_loss)

			# Save Best U-Net model
			if best_unet_loss > epoch_loss:
				best_unet_loss = epoch_loss
				best_epoch = epoch
				best_unet = self.unet.state_dict()
				print('Best %s model loss : %.4f'%(self.model_type,best_unet_loss))	
				torch.save(best_unet,unet_path)

	def test(self):
		#===================================== Test ====================================#
		# if self.model_type == 'U_Net_Seg':
		# 	return None

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path))
		print('Finish loading model parameters')
		self.unet.train(False)
		self.unet.eval()
		self.criterion = torch.nn.MSELoss(reduction='mean')
		self.print_network(self.unet, self.model_type)

		test_loader, T1_pathlist = get_evalutation_loader(self.batch_size, 4, 'test', augmentation_prob=0, dis=self.dis)

		scorelist = []
		for i, (images, GT, seg) in enumerate(test_loader):   

			images = images.to(self.device)
			GT = GT.to(self.device)
			seg = seg.to(self.device)

			# SR : Segmentation Result
			if self.model_type in ['U_Net3D_Att_Seg','U_Net3D_Att_Seg_V2']:
				images = torch.cat((images,seg), 1)
				SR = self.unet(images)
			elif self.model_type in ['U_Net_Seg']:
				seg = seg.to(torch.int64)
				SR = self.unet(images)
				SR = torch.argmax(F.softmax(SR, dim=1),dim=1, keepdims=True)
			else:
				SR = self.unet(images)
			
			loss = 0
			images = images.data.cpu()[:,0,:,:,:]
			GT = GT.data.cpu().numpy()[:,0,:,:,:]
			SR = SR.data.cpu().numpy()[:,0,:,:,:]    				
			seg = seg.data.cpu().numpy()
			seg = np.argmax(seg, axis=1)
			# print(seg.shape, SR.shape, GT.shape)
			
			corr = E_correlation(SR.reshape(-1), GT.reshape(-1))
			MAE = get_mean_absolute_error(SR, GT)
			MSE, PSNR = get_MSE_PSNR(SR, GT)
			MRE = get_mean_relative_absolute_deviation(SR, GT)
			tissue_MAE = tissue_error(seg, SR, GT)
			tis_MRE = tissue_MRE(seg, SR, GT)
			value_MAE = value_error( SR, GT)
			
			print('corr:', corr, 'MAE:', MAE, 'MSE:', MSE, 'PSNR:', PSNR, 'MRE:', MRE)
			print('tissue_MAE"', 'null:', tissue_MAE[0], 'white:', tissue_MAE[1], 'gray:', tissue_MAE[2], 'CSF:', tissue_MAE[3], 'bone:', tissue_MAE[4], 'skin:', tissue_MAE[5] )
			print('tissue_MRE"', 'null:', tis_MRE[0], 'white:', tis_MRE[1], 'gray:', tis_MRE[2], 'CSF:', tis_MRE[3], 'bone:', tis_MRE[4], 'skin:', tis_MRE[5] )
			print('value_MAE', '<0.2',value_MAE[0],'0.2-0.7',value_MAE[1], '0.7-1.2',value_MAE[2], '>1.2', value_MAE[3])
			
			scorelist.append([0, corr, MAE, MSE, PSNR, MRE, \
				tissue_MAE[0],tissue_MAE[1],tissue_MAE[2],tissue_MAE[3],tissue_MAE[4],tissue_MAE[5], \
				tis_MRE[0],tis_MRE[1],tis_MRE[2],tis_MRE[3],tis_MRE[4],tis_MRE[5], \
				value_MAE[0],value_MAE[1], value_MAE[2],value_MAE[3]])

			savedir = './results'
			if not os.path.exists(savedir):
				os.mkdir(savedir)
			model_predtion_dir = os.path.join(savedir, 'motor_{}_prediction'.format(self.model_type))
			if not os.path.exists(model_predtion_dir):
				os.mkdir(model_predtion_dir)

			for sub in os.listdir('./dataset/motor_precessed'):
				subdir = os.path.join(model_predtion_dir, sub)
				if not os.path.exists(subdir):
					os.mkdir(subdir)

			pathlist =  T1_pathlist[i]
			for j in range(len(pathlist)):
				path = pathlist[j]
				image_tmp = SR[j,  :, :, :]
				sub = path.split('/')[-2]
				key = 'Evalue_'+ path.split('/')[-1][3:]
				np.save(os.path.join(model_predtion_dir, sub+'/'+key), image_tmp)

				# if key in ['Evalue_55_56_90.npy','Evalue_50_50_0.npy','Evalue_55_60_180.npy']:
				# 	save_name = '{}/{}_{}_{}'.format(model_predtion_dir, self.model_type, sub, key[:-4])
				# 	self.data2nii(images[j,:,:,:], GT[j,:,:,:], SR[j,:,:,:], save_name)

		names = ['loss','corr','MAE','MSE','PSNR','MRE','MAE_Null','MAE_white','MAE_gray','MAE_CSF','MAE_bone','MAE_skin','MRE_Null','MRE_white','MRE_gray','MRE_CSF','MRE_bone','MRE_skin','<0.2','0.2-0.7','0.7-1.2','>1.2']
		print(np.array(scorelist).mean(axis=0))
		print(self.model_type)
		scorelist = pd.DataFrame(scorelist, columns =names)
		workdir = './resutls'
		scorelist.to_csv('{}/{}_score.csv'.format(workdir, self.model_type))

	def evaluation( self, savedir = './dataset'):
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path))
		print('Finish loading model parameters')
		self.unet.train(False)
		self.unet.eval()
		self.criterion = torch.nn.MSELoss(reduction='mean')

		savedir = './results'
		if not os.path.exists(savedir):
			os.mkdir(savedir)
		model_predtion_dir = os.path.join(savedir, 'motor_{}_prediction'.format(self.model_type))
		if not os.path.exists(model_predtion_dir):
			os.mkdir(model_predtion_dir)

		for sub in os.listdir('./dataset/motor_precessed'):
			subdir = os.path.join(model_predtion_dir, sub)
			if not os.path.exists(subdir):
				os.mkdir(subdir)

		dataloader, T1_pathlist = get_evalutation_loader(24, 4, 'test', augmentation_prob=0, dis=self.dis)

		for i, (images, GT, seg) in enumerate(dataloader):   

			images = images.to(self.device)
			GT = GT.to(self.device)
			seg = seg.to(self.device)

			if self.model_type in ['U_Net3D_Att_Seg','U_Net3D_Att_Seg_V2']:
				images = torch.cat((images,seg), 1)
				SR = self.unet(images)
			else:
				SR = self.unet(images)

			SR = SR.data.cpu()
			pathlist =  T1_pathlist[i]
			for j in range(len(pathlist)):
				path = pathlist[j]
				image_tmp = SR[j, 0, :, :, :]
				sub = path.split('/')[-2]
				key = 'Evalue_'+ path.split('/')[-1][3:]
				np.save(os.path.join(model_predtion_dir, sub+'/'+key), image_tmp)
				# if key in ['Evalue_55_56_90.npy','Evalue_50_50_0.npy','Evalue_55_60_180.npy']:
				# 	save_name = '{}/{}_{}_{}'.format(model_predtion_dir, self.model_type, sub, key[:-4])
				# 	self.data2nii(images[j,:,:,:], GT[j,:,:,:], SR[j,:,:,:], save_name)

		return None

	def segmentation(self):
		"""Combine the seg and reg Unet3D_Att"""
		t1 = time.time()
		seg_unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %('U_Net_Seg', self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
		reg_unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %('U_Net3D_Att_Seg_V2', self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
		print(seg_unet_path, reg_unet_path)
		# U-Net Train
		self.segmentation_unet = U_Net3D_Att(img_ch=1, output_ch=6)
		self.regression_unet = U_Net3D_Att(img_ch=7, output_ch=1)
		self.segmentation_unet.to(self.device)
		self.regression_unet.to(self.device)
		self.segmentation_unet.load_state_dict(torch.load(seg_unet_path))
		self.regression_unet.load_state_dict(torch.load(reg_unet_path))
		print('%s is Successfully Loaded from %s' % ( 'U_Net_Seg', seg_unet_path  ))
		print('%s is Successfully Loaded from %s' % ('U_Net3D_Att_Seg', reg_unet_path  ))
		# ===================================== Validation ====================================#
		self.segmentation_unet.train(False)
		self.segmentation_unet.eval()
		self.regression_unet.train(False)
		self.regression_unet.eval()	
		self.model_type = 'two_step'
		length=0
		epoch_loss = 0

		savedir = './results'
		if not os.path.exists(savedir):
			os.mkdir(savedir)
		model_predtion_dir = os.path.join(savedir, 'motor_{}_prediction'.format(self.model_type))
		if not os.path.exists(model_predtion_dir):
			os.mkdir(model_predtion_dir)

		for sub in os.listdir('./dataset/motor_precessed'):
			subdir = os.path.join(model_predtion_dir, sub)
			if not os.path.exists(subdir):
				os.mkdir(subdir)

		dataloader, T1_pathlist = get_evalutation_loader(self.batch_size, 4, 'test', augmentation_prob=0, dis=self.dis)
		scorelist = []
		with torch.no_grad():
			for i, (images, GT, seg) in enumerate(dataloader):   
				print('evaluating')
				images = images.to(self.device)
				GT = GT.to(self.device)
				seg = seg.to(self.device)
				seg = seg.to(torch.int64)
				t2 = time.time()
				preseg = self.segmentation_unet(images)
				preseg = torch.argmax(F.softmax(preseg, dim=1), dim=1)
				preseg = torch.nn.functional.one_hot(preseg, 6).permute(0,4,1,2,3).to(torch.float32)
				images = torch.cat((images,preseg), 1)
				t22 = time.time()
				SR = self.regression_unet(images)
				t3 = time.time()
				SR_flat = SR.view(SR.size(0),-1)              
				GT_flat = GT.view(GT.size(0),-1)
				
				print('two_step time',t3-t2)
				print('one_step time',t3-t22)
				images = images.data.cpu()
				GT = GT.data.cpu().numpy()[:,0,:,:,:]
				SR = SR.data.cpu().numpy()[:,0,:,:,:]    				
				seg = seg.data.cpu().numpy()
				seg = np.argmax(seg, axis=1)
				preseg = preseg.data.cpu().numpy()
				preseg = np.argmax(preseg, axis=1)
				# print(seg.shape, SR.shape, GT.shape)
				
				corr = E_correlation(SR.reshape(-1), GT.reshape(-1))
				MAE = get_mean_absolute_error(SR, GT)
				MSE, PSNR = get_MSE_PSNR(SR, GT)
				MRE = get_mean_relative_absolute_deviation(SR, GT)
				tissue_MAE = tissue_error(seg, SR, GT)
				tis_MRE = tissue_MRE(seg, SR, GT)
				value_MAE = value_error( SR, GT)

				print('corr:', corr, 'MAE:', MAE, 'MSE:', MSE, 'PSNR:', PSNR, 'MRE:', MRE)
				print('tissue_MAE"', 'null:', tissue_MAE[0], 'white:', tissue_MAE[1], 'gray:', tissue_MAE[2], 'CSF:', tissue_MAE[3], 'bone:', tissue_MAE[4], 'skin:', tissue_MAE[5] )
				print('tissue_MRE"', 'null:', tis_MRE[0], 'white:', tis_MRE[1], 'gray:', tis_MRE[2], 'CSF:', tis_MRE[3], 'bone:', tis_MRE[4], 'skin:', tis_MRE[5] )
				print('value_MAE', '<0.2', value_MAE[0], '0.2-0.7', value_MAE[1], '0.7-1.2', value_MAE[2], '>1.2', value_MAE[3])
				
				scorelist.append([0, corr, MAE, MSE, PSNR, MRE, \
					tissue_MAE[0],tissue_MAE[1],tissue_MAE[2],tissue_MAE[3],tissue_MAE[4],tissue_MAE[5], \
					tis_MRE[0],tis_MRE[1],tis_MRE[2],tis_MRE[3],tis_MRE[4],tis_MRE[5], \
					value_MAE[0],value_MAE[1], value_MAE[2],value_MAE[3]])

				pathlist =  T1_pathlist[i]
				for j in range(len(pathlist)):
					path = pathlist[j]
					image_tmp = SR[j, :, :, :]
					sub = path.split('/')[-2]
					key = 'Evalue_'+ path.split('/')[-1][3:]
					np.save(os.path.join(model_predtion_dir, sub+'/'+key), image_tmp)

				print('other time',time.time()-t3)
			print('total time:', time.time()-t1)
			names = ['loss','corr','MAE','MSE','PSNR','MRE','MAE_Null','MAE_white','MAE_gray','MAE_CSF','MAE_bone','MAE_skin','MRE_Null','MRE_white','MRE_gray','MRE_CSF','MRE_bone','MRE_skin','<0.2','0.2-0.7','0.7-1.2','>1.2']
			print(np.array(scorelist).mean(axis=0))
			scorelist = pd.DataFrame(scorelist, columns =names)
			workdir = './resutls'
			scorelist.to_csv('{}/{}_score.csv'.format(workdir,self.model_type))

				
		return None

	# def data2nii( self, image, label, predict, save_name):
	# 	sub = '101309'
	# 	T1_file = nib.load('./SUB_T1/{}_T1.nii.gz'.format(sub))
	# 	T1 = T1_file.get_data()
	# 	affine = T1_file.affine

	# 	NewImage = nib.Nifti1Image(image, affine = T1_file.affine, header = T1_file.header)
	# 	nib.save(NewImage, save_name+'_Image.nii.gz')


	# 	NewImage = nib.Nifti1Image(label, affine = T1_file.affine, header = T1_file.header)
	# 	nib.save(NewImage, save_name+'_GTruth.nii.gz')


	# 	NewImage = nib.Nifti1Image(predict, affine = T1_file.affine, header = T1_file.header)
	# 	nib.save(NewImage, save_name+'_Prediction.nii.gz')

	# 	return None


