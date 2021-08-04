import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import os




class motorFolder(data.Dataset):
	def __init__(self, mode, augmentation_prob=0.5, dis=False ):
		"""Initializes image paths and preprocessing module."""
		self.root = './dataset/motor_precessed'
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.mode = mode
		self.dis = dis
		if mode == 'train':
			sublist = ['101309', '117122', '120111','122620', '128632','131722', '130013', '138534','160123','198451']
			srange = [0,0.8]
		elif mode =='val':
			sublist = ['101309', '117122', '120111','122620', '128632','131722', '130013', '138534','160123','198451']
			srange = [0.8,1]
		elif mode == 'test':
			sublist = ['110411', '122317', '124422', '149337','151627']
			srange = [0,1]
		elif mode == 'all':
			sublist = ['101309', '117122', '120111','122620', '128632','131722', '130013', '138534','160123','198451','110411', '122317', '124422', '149337','151627']
			srange = [0,1]
		self.image_paths = []
		for sub in os.listdir(self.root):
			if sub not in sublist:
				continue
			subdir = os.path.join(self.root, sub)
			
			T1_list = []
			for file in os.listdir(subdir):
				if file.endswith('.npy')  & file.startswith('T1_'):
					T1_list.append(os.path.join(subdir, file))
			length = len(T1_list)
			left, right = int(length*srange[0]), int(length*srange[1])
			for i in range(left, right):
				self.image_paths.append(T1_list[i])

		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def test_exists(self):
		count = 0
		for path in self.image_paths:
			file = path.split('/')
			subdir = path[:-len(file[-1])]

			key = 'Evalue_'+file[-1][3:]
			labelpath = os.path.join(subdir, key)
			key = 'Seg_'+file[-1][3:]
			segpath = os.path.join(subdir, key)				
			if os.path.exists(path) & os.path.exists(labelpath) & os.path.exists(segpath):
				count += 1
			else:
				print(path)
		return count

	def __getitem__(self, index):
		# Transform = T.ToTensor()

		image_path = self.image_paths[index]
		T1_image = np.load(image_path)
		T1_image = np.expand_dims(T1_image[:80,:80,:40], axis=0)
		
		file = image_path.split('/')
		subdir = image_path[:-len(file[-1])]

		key = 'Evalue_'+file[-1][3:]
		labelpath = os.path.join(subdir, key)
		E_image = np.load(labelpath)
		E_image = np.expand_dims(E_image[:80,:80,:40], axis=0)
		
		key = 'Seg_'+file[-1][3:]
		segpath = os.path.join(subdir, key)
		Seg_image = np.load(segpath)[:80,:80,:40]
		
		p_transform = random.random()
		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			T1_image = T1_image[:,::-1,::-1,:40].copy()
			E_image = E_image[:,::-1,::-1,:40].copy()
			Seg_image = Seg_image[::-1,::-1,:40].copy()
		

		Seg_image = torch.nn.functional.one_hot(torch.from_numpy(Seg_image.astype('int64')), 6).permute(3,0,1,2).to(torch.float32)
		T1_image = torch.from_numpy(T1_image.astype('float32'))
		E_image = torch.from_numpy(E_image.astype('float32'))

		return T1_image, E_image, Seg_image

		
	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)






def get_data_loader(batch_size, num_workers, mode, augmentation_prob=0.5, dis=False):
	"""Builds and returns Dataloader."""
	
	dataset = motorFolder( mode=mode, augmentation_prob=augmentation_prob, dis=dis)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader


def get_evalutation_loader(batch_size, num_workers, mode, augmentation_prob=0.5, dis=False):
	"""Builds and returns Dataloader."""
	dataset = motorFolder( mode=mode, augmentation_prob=0, dis=dis)
	image_paths = np.array(dataset.image_paths)
	data_loader = data.DataLoader(dataset = dataset,
								  batch_size = batch_size,
								  shuffle = False,
								  num_workers = num_workers,
								  drop_last = True)
	batchnum = len(image_paths)//batch_size
	image_paths = image_paths[:batchnum*batch_size].reshape(-1, batch_size)
	return data_loader, image_paths