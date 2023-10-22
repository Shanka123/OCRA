import os
import argparse
from ocra_svrt import *
import time
import datetime
import torch.optim as optim
import torch
from PIL import Image
from torchvision.transforms import transforms
from util import log
from torch.utils.data import Dataset, DataLoader
import json
from scipy import misc
import glob
import torchvision.transforms.functional as TF
import random
import imageio
import cv2

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)


class SVRTdataset(Dataset):
	def __init__(self,  root_dir,dataset_type, img_size,configuration,n):
		self.root_dir = root_dir
		self.transforms = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Lambda(lambda X: 2 * X - 1.0),
					transforms.Resize((img_size,img_size)),
					
					


				]
			)
	


		self.file_names = np.load(os.path.join(root_dir,'{}_img_files_allclasses.npy'.format(dataset_type)))
		self.file_names_config_sample0 = sorted([f for f in self.file_names if (configuration==f.split('/')[1]) and ('sample_0' in f.split('/')[2][:8])])
		self.file_names_config_sample1 = sorted([f for f in self.file_names if (configuration==f.split('/')[1]) and ('sample_1' in f.split('/')[2][:8])])
			
		if dataset_type == 'train':
			self.train = True
			random.seed(2023)

			self.file_names_config_sampled_tuple = random.sample(list(zip(self.file_names_config_sample0,self.file_names_config_sample1)),n)
			self.file_names_config_sampled = []
			for i in range(len(self.file_names_config_sampled_tuple)):
				self.file_names_config_sampled.append(self.file_names_config_sampled_tuple[i][0])
				self.file_names_config_sampled.append(self.file_names_config_sampled_tuple[i][1])

			# print("For training sampled files>>",self.file_names_config_sampled)

		else:
			self.train=False
			self.file_names_config_sampled_tuple = list(zip(self.file_names_config_sample0,self.file_names_config_sample1))
			self.file_names_config_sampled = []
			for i in range(len(self.file_names_config_sampled_tuple)):
				self.file_names_config_sampled.append(self.file_names_config_sampled_tuple[i][0])
				self.file_names_config_sampled.append(self.file_names_config_sampled_tuple[i][1])
		
	
		
		
		# print(self.all_imgs.shape)  

	
		
		
	
	
			 
		# print(self.file_names[:100])
		# if dataset_type == 'train':
		# 	self.file_names = self.file_names[:10000]	
		
		# self.embeddings = np.load('./embedding.npy')

	def __len__(self):
		return len(self.file_names_config_sampled)

	def __getitem__(self, idx):
		# data_path = os.path.join(self.root_dir, self.file_names[idx])
		file_name = self.file_names_config_sampled[idx]
	
		if 'sample_0' in file_name.split('/')[2]:
			target = 0
		else:
			target = 1
		

		if self.train:
			if np.random.rand()<0.5:
				hflip_flag=1
			else:
				hflip_flag=0
			if np.random.rand()<0.5:
				vflip_flag=1
			else:
				vflip_flag=0
		
		else:
			
			hflip_flag=0
			vflip_flag =0
		
		if vflip_flag==1 and hflip_flag==1:
			img = TF.vflip(TF.hflip(self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_name),cv2.COLOR_BGR2GRAY).astype(np.uint8)))))
		elif vflip_flag==1 and hflip_flag==0:
			img = TF.vflip(self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_name),cv2.COLOR_BGR2GRAY).astype(np.uint8))))
		elif vflip_flag==0 and hflip_flag==1:
			img = TF.hflip(self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_name),cv2.COLOR_BGR2GRAY).astype(np.uint8))))
		else:
			# img = [TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle) for i in range(16)]
			img = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(file_name),cv2.COLOR_BGR2GRAY).astype(np.uint8)))
			
		
		return img, target






def load_slot_checkpoint(slot_model,checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
	# transformer_scoring_model.load_state_dict(model_ckp['transformer_scoring_model_state_dict'])
	
	return slot_model





parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_slots', default=6, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--depth', default=24, type=int, help='transformer number of layers')
parser.add_argument('--heads', default=8, type=int, help='transformer number of heads')
parser.add_argument('--mlp_dim', default=512, type=int, help='transformer mlp dimension')
parser.add_argument('--task', type=str, default='svrt', help="{'same_diff', 'RMTS', 'dist3', 'identity_rules'}")


parser.add_argument('--learning_rate', default=0.00004, type=float)
parser.add_argument('--warmup_steps', default=800, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=2000, type=int, help='number of workers for loading data')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--run', type=str, default='1')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=5000)

parser.add_argument('--path', default='svrt/', type=str, help='dataset path')
parser.add_argument('--configuration', type=str, default='results_problem_1')
parser.add_argument('--n', type=int, default=250)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--model_name', type=str, default='slot_attention_augmentations_first_more_pretrained_svrt_500_images_alltasks_frozen_autoencoder_ocra')
parser.add_argument('--model_checkpoint', type=str, default='model saved checkpoint')
parser.add_argument('--apply_context_norm', type=bool, default=True)
# parser.add_argument('--accumulation_steps', type=int, default=8)

args = parser.parse_args()
print(args)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")





log.info('Generating svrt dataset')
train_data = SVRTdataset(args.path, "train", args.img_size,args.configuration,args.n)
val_data = SVRTdataset(args.path, "val", args.img_size,args.configuration,args.n)
test_data = SVRTdataset(args.path, "test", args.img_size,args.configuration,args.n)


# Convert to PyTorch DataLoaders




train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
						shuffle=True, num_workers=args.num_workers)

val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
						shuffle=False, num_workers=args.num_workers)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
						shuffle=False, num_workers=args.num_workers)


print("Number of samples in training set>>",len(train_dataloader))
print("Number of samples in validation set>>",len(val_dataloader))

print("Number of samples in test set>>",len(test_dataloader))









log.info('Building model...')
# slotmask_model = SlotMaskAttention((opt.img_size,opt.img_size), opt.num_slots, opt.hid_dim).to(device)
slot_model = SlotAttentionAutoEncoder((args.img_size,args.img_size), args.num_slots,args.num_iterations, args.hid_dim).to(device)

ocra_model = scoring_model(args,args.hid_dim,args.depth,args.heads,args.mlp_dim,args.num_slots).to(device)
slot_model = load_slot_checkpoint(slot_model,'weights/slot_attention_autoencoder_augmentations_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_grayscale_svrt_alltasks_num_images_250_run_1_more_x3_continuetraining_best.pth.tar')

# slot_model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])
mse_criterion = nn.MSELoss()
bce_criterion = nn.BCELoss()
sigmoid_activation = nn.Sigmoid()


# params = [{'params': list(slot_model.parameters()) + list(correlnet_scoring_model.parameters())}]
params = [{'params': ocra_model.parameters()}]
# model_parameters = filter(lambda p: p.requires_grad, list(slot_model.parameters()) + list(transformer_scoring_model.parameters()))
# model_parameters = filter(lambda p: p.requires_grad, slot_model.parameters())

# print("trainable parameters>>",sum([np.prod(p.size()) for p in model_parameters]))

# params = [{'params': slot_model.parameters()}]

log.info('Setting up optimizer...')

optimizer = optim.Adam(params, lr=args.learning_rate)

log.info('Training begins...')
start = time.time()
i = 0
max_val_acc=0
# optimizer.zero_grad()
for epoch in range(1,args.num_epochs+1):
	slot_model.eval()
	ocra_model.train()

	
	# all_trainloss = []
	all_trainmseloss = []
	all_trainbceloss=[]
	all_trainacc = []
   
	# total_train_images = torch.Tensor().to(device).float()
	# total_train_recons_combined = torch.Tensor().to(device).float()
	# total_train_recons = torch.Tensor().to(device).float()
	# total_train_masks = torch.Tensor().to(device).float()
	
	for batch_idx, (img,target) in enumerate(train_dataloader):
		
		# print(img.shape, torch.max(img),torch.min(img))
		# print("image and masks and target shape>>",img.shape,masks.shape,target.shape)
		# print(torch.max(img), torch.min(img),torch.max(masks),torch.min(masks))
		# i += 1

		# if i < args.warmup_steps:
		# 	learning_rate = args.learning_rate * (i / args.warmup_steps)
		# else:
		learning_rate = args.learning_rate

		# learning_rate = learning_rate * (opt.decay_rate ** (
		# 	i / opt.decay_steps))

		optimizer.param_groups[0]['lr'] = learning_rate
		img = img.to(device).float() #.unsqueeze(1).float()
		# masks = masks.to(device).float()
		target = target.to(device).float()
		
		
			# print(idx)
			# print(img[:,idx].shape,masks[:,idx].shape)
		
			# slots = slotmask_model(img[:,idx],masks[:,idx],device)
		recon_combined, recons, masks, feat_slots,pos_slots,attn = slot_model(img,device)
		# print("mask max and min>>", torch.max(masks), torch.min(masks))
		

		

			
		

		# print("slot reps>>",all_panels.shape)
		# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

		score = ocra_model(feat_slots,pos_slots,device)
		
		# print("scores and target>>",scores,target)
		pred = torch.round(sigmoid_activation(score)).int()

		
		acc = torch.eq(pred,target).float().mean().item() * 100.0
		all_trainacc.append(acc)
# 		
# print(torch.max(recon_combined), torch.min(recon_combined))
# 		# print(img.shape, recon_combined.shape, recons.shape, masks.shape, slots.shape)
		# if batch_idx<10:
		# 	total_train_images = torch.cat((total_train_images,img),dim=0)
		# 	total_train_recons_combined = torch.cat((total_train_recons_combined,torch.stack(recon_combined_seq,dim=1)),dim=0)
		# 	total_train_recons = torch.cat((total_train_recons,torch.stack(recons_seq,dim=1)),dim=0)
		# 	total_train_masks = torch.cat((total_train_masks,torch.stack(masks_seq,dim=1)),dim=0)
			
		
		# print("mse loss>>>",mse_criterion(torch.stack(recon_combined_seq,dim=1).squeeze(4), img))
		# print("ce loss>>",ce_criterion(scores,target))
		# print("recon combined seq shape>>",torch.stack(recon_combined_seq,dim=1).shape)
		# loss = 1000*mse_criterion(torch.stack(recon_combined_seq,dim=1), img) + ce_criterion(scores,target)
		loss = bce_criterion(sigmoid_activation(score),target)
		# loss = ce_criterion(scores,target)
		all_trainmseloss.append(mse_criterion(recon_combined, img).item())
		all_trainbceloss.append(loss.item())
		
# 		all_trainloss.append(loss.item())

# 		del recons, masks, slots
		# loss = loss / opt.accumulation_steps   
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# print("learning rate>>>",learning_rate)
		# for j, para in enumerate(slotmask_model.parameters()):
		#     print(f'{j + 1}th parameter tensor:', para.shape)
		#     # print(para)
		#     print("gradient>>",para.grad)
		# if (batch_idx+1) % opt.accumulation_steps == 0:             # Wait for several backward steps
		# 	optimizer.step()                            # Now we can do an optimizer step
		# 	optimizer.zero_grad()  
		
		if batch_idx % args.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_dataloader)) + '] ' + \
					 '[Total Loss = ' + '{:.4f}'.format(loss.item()) + '] ' +\
					 '[MSE Loss = ' + '{:.4f}'.format(mse_criterion(recon_combined, img).item()) + '] ' +\
					 '[BCE Loss = ' + '{:.4f}'.format(loss.item()) + '] ' +\
					 '[Learning rate = ' + '{:.8f}'.format(learning_rate) + '] ' + \
					 '[Accuracy = ' + '{:.2f}'.format(acc) + ']'

					  )
	print("Average training reconstruction loss>>",np.mean(np.array(all_trainmseloss)))
	print("Average training binary cross entropy loss>>",np.mean(np.array(all_trainbceloss)))
	print("Average training accuracy>>",np.mean(np.array(all_trainacc)))
	# np.savez('predictions/slot_attention_augmentations_pretrained_svrt_images_frozen_autoencoder_eval_{}_train_images_recons_masks.npz'.format(args.configuration), images= img.cpu().detach().numpy() ,recon_combined = recon_combined.cpu().detach().numpy(),recons = recons.cpu().detach().numpy(),masks= masks.cpu().detach().numpy(),attention= attn.cpu().detach().numpy() )
	# np.savez('predictions/{}_tcn_shuffling_augmentation_9slots_nolrdecay_rowcolposemb_raven_allconfigs_train_images_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,masks= masks.cpu().detach().numpy() )

	

	slot_model.eval()
	ocra_model.eval()
	all_valacc = []

	for val_batch_idx, (val_img,val_target) in enumerate(val_dataloader):
			# print("image and masks and target shape>>",img.shape,masks.shape,target.shape)
	# 		# print(torch.max(img), torch.min(img),img.shape)
			
		val_img = val_img.to(device).float() #.unsqueeze(1).float()
			# masks = masks.to(device).float()
		val_target = val_target.to(device).float()
			

			# recon_combined_seq =[]
			# recons_seq=[]
			# masks_seq=[]
			
				# print(idx)
				# print(img[:,idx].shape,masks[:,idx].shape)
			
				# slots = slotmask_model(img[:,idx],masks[:,idx],device)
				# # print("mask max and min>>", torch.max(masks), torch.min(masks))
				# slots_seq.append(slots)
				
				# del slots
		val_recon_combined, val_recons, val_masks, val_feat_slots,val_pos_slots,val_attn = slot_model(val_img,device)
			

				# recon_combined_seq.append(recon_combined)
				# recons_seq.append(recons)
				# masks_seq.append(masks)
				

			
		
			# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

		score = ocra_model(val_feat_slots,val_pos_slots,device)
			# print("scores and target>>",scores,target)
		pred = torch.round(sigmoid_activation(score)).int()
		acc = torch.eq(pred,val_target).float().mean().item() * 100.0
		all_valacc.append(acc)

			
	print("Average validation accuracy>>>",np.mean(np.array(all_valacc)))

		
	if np.mean(np.array(all_valacc)) > max_val_acc:
		print("Validation accuracy increased from %s to %s"%(max_val_acc,np.mean(np.array(all_valacc))))
		max_val_acc = np.mean(np.array(all_valacc))
		
		all_testacc = []
	   
		# total_train_images = torch.Tensor().to(device).float()
		# total_train_recons_combined = torch.Tensor().to(device).float()
		# total_train_recons = torch.Tensor().to(device).float()
		# total_train_masks = torch.Tensor().to(device).float()
		
		for batch_idx, (test_img,test_target) in enumerate(test_dataloader):
			# print("image and masks and target shape>>",img.shape,masks.shape,target.shape)
	# 		# print(torch.max(img), torch.min(img),img.shape)
			
			test_img = test_img.to(device).float() #.unsqueeze(1).float()
			# masks = masks.to(device).float()
			test_target = test_target.to(device).float()
			

			# recon_combined_seq =[]
			# recons_seq=[]
			# masks_seq=[]
			
				# print(idx)
				# print(img[:,idx].shape,masks[:,idx].shape)
			
				# slots = slotmask_model(img[:,idx],masks[:,idx],device)
				# # print("mask max and min>>", torch.max(masks), torch.min(masks))
				# slots_seq.append(slots)
				
				# del slots
			test_recon_combined, test_recons, test_masks, test_feat_slots,test_pos_slots,test_attn = slot_model(test_img,device)
			

				# recon_combined_seq.append(recon_combined)
				# recons_seq.append(recons)
				# masks_seq.append(masks)
				

			
		
			# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

			score = ocra_model(test_feat_slots,test_pos_slots,device)
			# print("scores and target>>",scores,target)
			pred = torch.round(sigmoid_activation(score)).int()
			acc = torch.eq(pred,test_target).float().mean().item() * 100.0
			all_testacc.append(acc)

			
		print("Average test accuracy>>>",np.mean(np.array(all_testacc)))
	else:
		print("Average validation accuracy didn't increase, skipping model saving!!!")
	print("Best validation accuracy till now>>> ",max_val_acc)


test_dir = './test/'
check_path(test_dir)
task_dir = test_dir + args.task + '/'
check_path(task_dir)

model_dir = task_dir + args.model_name + '/'
check_path(model_dir)
test_fname = model_dir + args.configuration + '.txt'
test_f = open(test_fname, 'w')
test_f.write('acc\n')
test_f.write(
			 '{:.2f}'.format(np.mean(np.array(all_testacc))))
test_f.close()		
	