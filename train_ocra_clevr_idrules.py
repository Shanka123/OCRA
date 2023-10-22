import os
import argparse
from ocra_clevr import *
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


class ClevrARTdataset(Dataset):
	def __init__(self,  root_dir,dataset_type, img_size,args):
		self.root_dir = root_dir
		self.transforms = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Lambda(lambda X: 2 * X - 1.0),
					transforms.Resize((img_size,img_size)),
					
					


				]
			)
	


		self.img_size = img_size
		self.allprobs = [os.path.join(root_dir+dataset_type+'_ood',f) for f in os.listdir(root_dir+dataset_type+'_ood')]
		self.len = len(self.allprobs)






	

	
		
		self.alltargets = np.load('identity_rules_ood_{}.npz'.format(dataset_type))['y']
		
		
		self.seq_len = 4
			 
		# print(self.file_names[:100])
		# if dataset_type == 'train':
		# 	self.file_names = self.file_names[:10000]	
		
		# self.embeddings = np.load('./embedding.npy')

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		# data_path = os.path.join(self.root_dir, self.file_names[idx])
		prob_file = self.allprobs[idx]
		y = self.alltargets[int(prob_file.split('/')[-1].split('_')[1])]
		# print("prob_file %s and target %s"%(prob_file,y))

		# x_seq = self.all_imgs[seq_ind]
		# print(x_seq.shape)
		# print(seq_ind.shape,x_seq.shape)

		img=[]
		for m in range(self.seq_len):

		

			
			# x_in1 = np.concatenate([x_seq[0,:,:], x_seq[1,:,:], x_seq[2,:,:]], axis=1)
			# x_in2 = np.concatenate([x_seq[3,:,:], x_seq[4,:,:], x_seq[5+m,:,:]], axis=1)
			# img.append(self.transforms(Image.fromarray(cv2.imread(os.path.join(prob_file,'CLEVR_{}.png'.format(m))).astype(np.uint8))))
			img.append(self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(prob_file,'CLEVR_{}.png'.format(m))),cv2.COLOR_BGR2RGB).astype(np.uint8))))
			# x_in.append(np.concatenate([x_in1, x_in2], axis=0))

		

			# img = [TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle) for i in range(16)]
		# img = [self.transforms(Image.fromarray(x_in[i].astype(np.uint8))) for i in range(len(x_in))]
	
		
		
		# img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]
		
		# img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]
	
			# resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
		resize_image = torch.stack(img,dim=0)
		
		
		
		
		return resize_image, y







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
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--depth', default=24, type=int, help='transformer number of layers')
parser.add_argument('--heads', default=8, type=int, help='transformer number of heads')
parser.add_argument('--mlp_dim', default=512, type=int, help='transformer mlp dimension')

parser.add_argument('--task', type=str, default='CLEVR_identity_rules', help="{'same_diff', 'RMTS', 'dist3', 'identity_rules'}")


parser.add_argument('--learning_rate', default=0.00008, type=float)
parser.add_argument('--warmup_steps', default=15000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=200, type=int, help='number of workers for loading data')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--run', type=str, default='1')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--path', default='output/idrules_images/', type=str, help='dataset path')

parser.add_argument('--img_size', type=int, default=128)
# parser.add_argument('--model_name', type=str, default='slot_attention_random_spatial_heldout_unicodes_resizedcropped_pretrained_frozen_autoencoder_new_correlnet-T_scoring')
parser.add_argument('--model_name', type=str, default='slot_attention_random_spatial_clevrshapes_cv2_rgbcolororder_pretrained_frozen_autoencoder_ocra')

parser.add_argument('--model_checkpoint', type=str, default='model saved checkpoint')
parser.add_argument('--apply_context_norm', type=bool, default=True)

# parser.add_argument('--accumulation_steps', type=int, default=8)

args = parser.parse_args()
print(args)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")





# Convert to PyTorch DataLoaders



train_data = ClevrARTdataset(args.path, 'train', args.img_size,args)
# val_data = Dist3dataset(args.path, val_set, args.img_size,args)
test_data = ClevrARTdataset(args.path, 'test', args.img_size,args)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
						shuffle=True, num_workers=args.num_workers)

# val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
# 						shuffle=False, num_workers=args.num_workers)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
						shuffle=False, num_workers=args.num_workers)


print("Number of samples in training set>>",len(train_dataloader))
# print("Number of samples in validation set>>",len(val_dataloader))

print("Number of samples in test set>>",len(test_dataloader))










log.info('Building model...')
# slotmask_model = SlotMaskAttention((opt.img_size,opt.img_size), opt.num_slots, opt.hid_dim).to(device)
slot_model = SlotAttentionAutoEncoder((args.img_size,args.img_size), args.num_slots,args.num_iterations, args.hid_dim).to(device)

ocra_model = scoring_model(args,args.hid_dim,args.depth,args.heads,args.mlp_dim,args.num_slots).to(device)
# slot_model = load_slot_checkpoint(slot_model,'weights/slot_attention_autoencoder_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_random_spatial_heldout_unicodes_resizedcropped_continuetraining_run_1_best.pth.tar')
slot_model = load_slot_checkpoint(slot_model,'weights/slot_attention_autoencoder_7slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_cv2_rgbcolororder_random_spatial_clevrshapes_continuetraining_run_1_best.pth.tar')
# slot_model = load_slot_checkpoint(slot_model,'weights/slot_attention_autoencoder_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_random_spatial_clevrshapes_continuetraining_run_1_best.pth.tar')

# slot_model = load_slot_checkpoint(slot_model,'weights/slot_attention_autoencoder_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_random_spatial_imgs_resizedcropped_run_1_best.pth.tar')


# slot_model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])
mse_criterion = nn.MSELoss()
ce_criterion = nn.CrossEntropyLoss()

# params = [{'params': list(slot_model.parameters()) + list(correlnet_scoring_model.parameters())}]
params = [{'params': ocra_model.parameters()}]
# model_parameters = filter(lambda p: p.requires_grad, list(slot_model.parameters()) + list(transformer_scoring_model.parameters()))
# model_parameters = filter(lambda p: p.requires_grad, slot_model.parameters())

# print("trainable parameters>>",sum([np.prod(p.size()) for p in model_parameters]))

# params = [{'params': slot_model.parameters()}]

# model_parameters = filter(lambda p: p.requires_grad, list(correlnet_scoring_model.parameters())+ list(slot_model.parameters()))
# model_parameters = filter(lambda p: p.requires_grad, slot_model.parameters())

# print("trainable parameters>>",sum([np.prod(p.size()) for p in model_parameters]))



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
	all_trainceloss=[]
	all_trainacc = []
   
	# total_train_images = torch.Tensor().to(device).float()
	# total_train_recons_combined = torch.Tensor().to(device).float()
	# total_train_recons = torch.Tensor().to(device).float()
	# total_train_masks = torch.Tensor().to(device).float()
	
	for batch_idx, (img,target) in enumerate(train_dataloader):
		# print(img.shape, torch.max(img),torch.min(img),target.shape)
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
		target = target.to(device)
		feat_slots_seq =[]
		pos_slots_seq =[]
		recon_combined_seq =[]
		recons_seq=[]
		masks_seq=[]
		attn_seq=[]
		for idx in range(img.shape[1]):
			# print(idx)
			# print(img[:,idx].shape,masks[:,idx].shape)
		
			# slots = slotmask_model(img[:,idx],masks[:,idx],device)
			recon_combined, recons, masks, feat_slots,pos_slots,attn = slot_model(img[:,idx],device)
			# print(feat_slots.shape, pos_slots.shape)
			# print("mask max and min>>", torch.max(masks), torch.min(masks))
			feat_slots_seq.append(feat_slots)
			pos_slots_seq.append(pos_slots)

			recon_combined_seq.append(recon_combined)
			recons_seq.append(recons)
			masks_seq.append(masks)
			attn_seq.append(attn)
			del recon_combined,recons, masks, feat_slots,pos_slots,attn

		

			
		

		feat_panels = torch.stack(feat_slots_seq,dim=1)
		pos_panels = torch.stack(pos_slots_seq,dim=1)
		# print("slot reps>>",all_panels.shape)
		# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

		scores = ocra_model(feat_panels,pos_panels,device)
		# print("scores>>",scores.shape)
		# print("scores and target>>",scores,target)
		pred = scores.argmax(1)
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
		loss =  ce_criterion(scores,target)
		# loss = ce_criterion(scores,target)
		all_trainmseloss.append(mse_criterion(torch.stack(recon_combined_seq,dim=1), img).item())
		all_trainceloss.append(ce_criterion(scores,target).item())
		
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
					 '[MSE Loss = ' + '{:.4f}'.format(mse_criterion(torch.stack(recon_combined_seq,dim=1), img).item()) + '] ' +\
					 '[CE Loss = ' + '{:.4f}'.format(ce_criterion(scores,target).item()) + '] ' +\
					 '[Learning rate = ' + '{:.8f}'.format(learning_rate) + '] ' + \
					 '[Accuracy = ' + '{:.2f}'.format(acc) + ']'

					  )
	print("Average training reconstruction loss>>",np.mean(np.array(all_trainmseloss)))
	print("Average training cross entropy loss>>",np.mean(np.array(all_trainceloss)))
	print("Average training accuracy>>",np.mean(np.array(all_trainacc)))
	# np.savez('predictions/{}_clevrdecoder_tcn_7slots_nolrdecay_rowcolposemb_idrules_train_images_recons_masks.npz'.format(args.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy(),attention= torch.stack(attn_seq,dim=1).cpu().detach().numpy() )
	# np.savez('predictions/{}_tcn_shuffling_augmentation_9slots_nolrdecay_rowcolposemb_raven_allconfigs_train_images_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,masks= masks.cpu().detach().numpy() )

	if epoch%args.save_interval==0:

		slot_model.eval()
		ocra_model.eval()

		
		
		all_testacc = []
	   
		total_mispred_test_images = torch.Tensor().to(device).float()
		total_mispred_test_recons_combined = torch.Tensor().to(device).float()
		total_mispred_test_recons = torch.Tensor().to(device).float()
		total_mispred_test_masks = torch.Tensor().to(device).float()
		total_mispred_test_attn = torch.Tensor().to(device).float()
		
		for batch_idx, (test_img,test_target) in enumerate(test_dataloader):
			# print("image and masks and target shape>>",img.shape,masks.shape,target.shape)
	# 		# print(torch.max(img), torch.min(img),img.shape)
			
			test_img = test_img.to(device).float() #.unsqueeze(1).float()
			# masks = masks.to(device).float()
			test_target = test_target.to(device)
			test_feat_slots_seq =[]
			test_pos_slots_seq =[]

			test_recon_combined_seq =[]
			test_recons_seq=[]
			test_masks_seq=[]
			test_attn_seq = []
			for idx in range(test_img.shape[1]):
				# print(idx)
				# print(img[:,idx].shape,masks[:,idx].shape)
			
				# slots = slotmask_model(img[:,idx],masks[:,idx],device)
				# # print("mask max and min>>", torch.max(masks), torch.min(masks))
				# slots_seq.append(slots)
				
				# del slots
				test_recon_combined, test_recons, test_masks, test_feat_slots,test_pos_slots,test_attn = slot_model(test_img[:,idx],device)
				test_feat_slots_seq.append(test_feat_slots)
				test_pos_slots_seq.append(test_pos_slots)

				test_recon_combined_seq.append(test_recon_combined)
				test_recons_seq.append(test_recons)
				test_masks_seq.append(test_masks)
				test_attn_seq.append(test_attn)
				del test_recon_combined,test_recons, test_masks, test_feat_slots,test_pos_slots, test_attn

			feat_panels = torch.stack(test_feat_slots_seq,dim=1)
			pos_panels = torch.stack(test_pos_slots_seq,dim=1)
		
			# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

			scores = ocra_model(feat_panels,pos_panels,device)
			# print("scores and target>>",scores,target)
			pred = scores.argmax(1)
			# for i in range(test_img.shape[0]):
			# 	if pred[i]!=test_target[i] and total_mispred_test_images.shape[0]<16:
			# 		# print("mispreds>>",pred[i],test_target[i])
			# 		total_mispred_test_images = torch.cat((total_mispred_test_images,test_img[i].unsqueeze(0)),dim=0)
			# 		total_mispred_test_recons_combined = torch.cat((total_mispred_test_recons_combined,torch.stack(test_recon_combined_seq,dim=1)[i].unsqueeze(0)),dim=0)
			# 		total_mispred_test_recons = torch.cat((total_mispred_test_recons,torch.stack(test_recons_seq,dim=1)[i].unsqueeze(0)),dim=0)
			# 		total_mispred_test_masks = torch.cat((total_mispred_test_masks,torch.stack(test_masks_seq,dim=1)[i].unsqueeze(0)),dim=0)
			# 		total_mispred_test_attn = torch.cat((total_mispred_test_attn,torch.stack(test_attn_seq,dim=1)[i].unsqueeze(0)),dim=0)
					
			acc = torch.eq(pred,test_target).float().mean().item() * 100.0
			all_testacc.append(acc)
		print("Average test accuracy>>>",np.mean(np.array(all_testacc)))
		# np.savez('predictions/{}_clevrdecoder_tcn_7slots_nolrdecay_rowcolposemb_idrules_test_images_recons_masks.npz'.format(args.model_name), images= test_img.cpu().detach().numpy() ,recon_combined = torch.stack(test_recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(test_recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(test_masks_seq,dim=1).cpu().detach().numpy(),attention= torch.stack(test_attn_seq,dim=1).cpu().detach().numpy() )
		# np.savez('predictions/{}_eval_tcn_shuffling_9slots_nolrdecay_rowcolposemb_raven_allconfigs_val_images_recons_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy() )
		# np.savez('predictions/{}_tcn_shuffling_augmentation_9slots_nolrdecay_rowcolposemb_raven_allconfigs_val_images_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,masks= masks.cpu().detach().numpy() )

		# if np.mean(np.array(all_valacc)) > max_val_acc:
		# 	print("Validation accuracy increased from %s to %s"%(max_val_acc,np.mean(np.array(all_valacc))))
		# 	max_val_acc = np.mean(np.array(all_valacc))

test_dir = './test/'
check_path(test_dir)
task_dir = test_dir + args.task + '/'
check_path(task_dir)
gen_dir = task_dir + 'm' + str(args.m_holdout) + '/'
check_path(gen_dir)
model_dir = gen_dir + args.model_name + '/'
check_path(model_dir)
test_fname = model_dir + 'run' + args.run + '.txt'
test_f = open(test_fname, 'w')
test_f.write('acc\n')
test_f.write(
			 '{:.2f}'.format(np.mean(np.array(all_testacc))))
test_f.close()