### This repository contains code for the Object-centric Relational Abstraction (OCRA) model.

## Requirements
* python 3.9.7
* NVIDIA GPU with CUDA 11.0+ capability
* torch==1.11.0
* torchvision==0.12.0
* glob
* PIL==8.4.0
* numpy==1.20.3
* einops==0.4.1


### ART

The pretrained slot attention model for the ART dataset is given under `weights/slot_attention_autoencoder_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_random_spatial_heldout_unicodes_resizedcropped_continuetraining_run_1_best.pth.tar`

To train on Same/different task with $m=95$ run `python train_ocra_sd.py --batch_size 16  --img_size 128 --num_epochs 600 --m_holdout 95 --run '1'` 

To train on Relational match-to-sample task with $m=95$ run `python train_ocra_rmts.py --batch_size 16  --img_size 128 --num_epochs 400 --m_holdout 95 --run '1'` 

To train on Distribution-of-3 task with $m=95$ run `python train_ocra_dist3.py --batch_size 16  --img_size 128 --num_epochs 400 --m_holdout 95 --run '1'` 

To train on Identity rules task with $m=95$ run `python train_ocra_dist3.py --batch_size 16  --img_size 128 --num_epochs 100 --m_holdout 95 --run '1' --task 'identity_rules' --test_gen_method 'subsample'` 

### SVRT

The pretrained slot attention model for the SVRT dataset using 500 samples for each task is given under `weights/slot_attention_autoencoder_augmentations_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_grayscale_svrt_alltasks_num_images_250_run_1_more_x3_continuetraining_best.pth.tar`

The pretrained slot attention model for the SVRT dataset using 1000 samples for each task is given under `weights/slot_attention_autoencoder_augmentations_6slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_grayscale_svrt_alltasks_num_images_500_run_1_more_more_continuetraining_best.pth.tar`

Generate the SVRT dataset using https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=pysvrt.git;a=summary 

Put the images for each of the tasks under 'svrt/' in a folder named `results_problem_1` for task 1 and so on.


Create `train_img_files_allclasses.npy` by randomly sampling 10000 images for each task with equal number of images for each class. Each entry of `train_img_files_allclasses.npy` should refer to the path of an image but formatted like `/////svrt/results_problem_1/sample_0_4102.png` 

Similarly create `val_img_files_allclasses.npy` by randomly sampling a different set of 4000 images for each task, `test_img_files_allclasses.npy` by randomly sampling a different set of 40000 images for each task. 

To run on task 1 with 500 training samples run `python train_ocra_svrt.py --batch_size 32  --img_size 128  --configuration 'results_problem_1' --run '1'`

### CLEVR-ART

The pretrained slot attention model for the CLEVR-ART dataset is given under `weights/slot_attention_autoencoder_7slots_clevrdecoder_morewarmup_lowerlr_nolrdecay_64dim_128res_cv2_rgbcolororder_random_spatial_clevrshapes_continuetraining_run_1_best.pth.tar`

To generate CLEVR-ART dataset, you need to install Blender by following the instructions from https://github.com/facebookresearch/clevr-dataset-gen

Create directory `output/rmts_images/train_ood` and run `render_images_rmts.py` to generate the CLEVR-ART training images for the Relational match-to-sample task.
Create directory `output/rmts_images/test_ood` and run `render_images_rmts.py` to generate the CLEVR-ART test images for the Relational match-to-sample task. On line 161 change to `prob_answer_arr = np.load("RMTS_ood_test.npz")`.

Create directory `output/idrules_images/train_ood` and run `render_images_idrules.py` to generate the CLEVR-ART training images for the Identity rules task.
Create directory `output/idrules_images/test_ood` and run `render_images_idrules.py` to generate the CLEVR-ART test images for the Identity rules task. On line 161 change to `prob_answer_arr = np.load("identity_rules_ood_train.npz")`.

To train on Relational match-to-sample task run `python train_ocra_clevr_rmts.py  --img_size 128 --run '1'`

To train on Identity rules task run `python train_ocra_clevr_idrules.py  --img_size 128 --run '1'`



<!--
**ocraneurips/ocraneurips** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
