USE_CUDA = 'yes'

#import pycaret
#from pycaret.classification import *

import PIL
from PIL import Image, ImageChops
from PIL import ImageDraw, ImageEnhance
from torchvision.transforms import ToPILImage
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy import linalg
from scipy.linalg import sqrtm
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

#import image_similarity_measures
from image_similarity_measures.quality_metrics import *
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import datetime
from time import time

from omegaconf import OmegaConf
from pickle import load
import sys
import os
from pathlib import Path
from taming.models.vqgan_vit import VQModel

#from taming.models.vqgan_carenet import VQModel
from einops import reduce
from sklearn.preprocessing import minmax_scale
import cv2
import pandas as pd
#import sewar
from taming.data.custom import CustomTest_crop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

from skimage import io
from skimage.metrics import structural_similarity as ssim
import cv2


torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

if USE_CUDA == 'yes':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

def load_config(config_path):
	config = OmegaConf.load(config_path)
	return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
	
	model = VQModel(**config.model.params)
	if ckpt_path is not None:
		sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
		missing, unexpected = model.load_state_dict(sd, strict=False)
	return model.eval()

def preprocess_vqgan(x):
	#x = 2.*x - 1.
	return x

def custom_to_pil(x):
	print("x", x.shape)
	x = x.detach().cpu()
	x = torch.clamp(x, -1., 1.)
	#x = (x + 1.)/2.
	x = x.permute(1,2,0).numpy()
	x = x[:, :, 0]
	x = (255*x).astype(np.uint8)
	x = Image.fromarray(x)
	if not x.mode == "RGB":
		x = x.convert("RGB")
	return x
	
def reconstruct_with_vqgan(x, model):
	# could also use model(x) for reconstruction but use explicit encoding and decoding here
	z, emb_loss, [_, _, indices],ze,loss_tensor = model.encode(x)
	# print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
	xrec = model.decode(z)
	xrec_e = model.decode(ze)
	return xrec , xrec_e , emb_loss , loss_tensor

def loss(x, model):
	dict_loss = model.test_step(batch=x,batch_idx=0)
	return dict_loss

def preprocess(img, target_image_size=256):
	s = min(img.size)

	if s < target_image_size:
		raise ValueError(f'min dim for image {s} < {target_image_size}')

	r = target_image_size / s
	s = (round(r * img.size[1]), round(r * img.size[0]))
	img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
	img = TF.center_crop(img, output_size=2 * [target_image_size])
	img = torch.unsqueeze(T.ToTensor()(img), 0)
	return img

def tensor_to_pil(tensor):
	if tensor.dim() == 4:
		tensor = tensor.squeeze(0)
	# Convert a single tensor image to a PIL Image
	tensor = tensor.mul(255).byte()  # Scale to [0, 255] and convert to uint8
	tensor = tensor.cpu().numpy()  # Move tensor to CPU and convert to numpy
	# If the tensor has a channel dimension of size 3 at dim=0 (C, H, W)
	if tensor.shape[0] == 3:
		tensor = tensor.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
	return Image.fromarray(tensor)


def stack_reconstructions(input, reconstruction, diff_morpho, diff_ssim, difference_w, highest_fid_windows_upscaled_all_visu_w, titles=[]):
	
	print(reconstruction)
	assert input.size == reconstruction.size == difference_w.size == highest_fid_windows_upscaled_all_visu_w.size == diff_ssim.size == diff_morpho.size
	w, h = input.size[0], input.size[1]
	img = Image.new("RGB", (6*w, h))
	img.paste(input, (0,0))
	img.paste(reconstruction, (1*w,0))
	img.paste(diff_morpho, (2*w,0))
	img.paste(diff_ssim, (3*w,0))
	img.paste(difference_w, (4*w,0))
	img.paste(Image.blend(input.convert("RGBA"), highest_fid_windows_upscaled_all_visu_w.convert("RGBA"), 0.7), (5*w,0))

	for i, title in enumerate(titles):
		ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255)) # coordinates, text, color
	return img

def homo(img,Homography_path):
		def find_ref(img,ref1,ref2,ref3,ref4):
			image_loaded = PIL.Image.open(img)
			if image_loaded.size[0] == 4500:
				if image_loaded.size[1] != 4340:
					image_loaded=image_loaded.crop((0,0,4500,4340))
			image_loaded=image_loaded.resize((1024,1024), Image.LANCZOS,reducing_gap=10.0)#.convert('L')
			image_loaded = np.asarray(image_loaded)
			def image_processing(template):
				template_img = cv2.imread(template)
				h, w, _ = template_img.shape
				template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
				# res = cv2.matchTemplate(image_loaded, template, cv2.TM_CCOEFF_NORMED)
				# conf = res.max()
				# ([y], [x]), conf = np.where(res == conf), conf
				ho, wo = image_loaded.shape
				indices_img = [[0, int(ho/2), 0, int(wo/2)] if 'top_left' in template else [0, int(ho/2), int(wo/2), wo] if 'top_right' in template else [int(ho/2), ho, 0, int(wo/2)] if 'bottom_left' in template else [int(ho/2), ho, int(wo/2), wo]][0]
				img_quarter = image_loaded[indices_img[0]:indices_img[1],indices_img[2]:indices_img[3]]
				res = cv2.matchTemplate(img_quarter, template_img, cv2.TM_CCOEFF_NORMED)
				conf = res.max()
				([y], [x]), conf = np.where(res == conf), conf
				coord_shift = [[y+0, x+0] if 'top_left' in template else [y+0, x+512] if 'top_right' in template else [y+512, x+0] if 'bottom_left' in template else [y+512, x+512]][0]
				y, x = coord_shift[0], coord_shift[1]
				return x, y, h, w, conf
			def image_write(x, y, h, w, conf):
				cv2.rectangle(image_loaded, (x, y), (x + w, y + h), (255, 255, 255), 2)
				text = f'Conf.: {round(float(conf), 2)}'
				cv2.putText(image_loaded, text, (x, y), 1, cv2.FONT_HERSHEY_PLAIN, (255, 255, 255), 2)
				return image_loaded
			XY = [None,None,None,None]
			i=-1
			for ref in [ref1,ref2,ref3,ref4]:
				i+=1
				x, y, h, w, conf = image_processing(ref)
				XY[i] = [x,y]
			return XY, image_loaded

		#Patch_References
		tl1,tl2,tl3,tl4 = [(220,250,270,300) if 'G270' in model else (220,190,270,240) if 'Step4' in model else (310,145,360,195)][0] #top_left
		tr1,tr2,tr3,tr4 = [(785,165,835,215) if ('G270' in model) or ('Step4' in model) else (785,110,835,160)][0] #top_right
		bl1,bl2,bl3,bl4 = (260,730,310,780) #bottom_left
		br1,br2,br3,br4 = (745,760,795,810) #bottom_right

		if 'G270' in img or 'CCZA' in img: 
			Homography_path_model = 'G270/Master_CaseAssy_G270_CCZA3604128_20230616_1905144999_'
			pts_dst_model=[[tl1,tl2],[tr1,tr2],[bl1,bl2],[br1,br2]]
		elif 'ST4' in img or 'IVZA' in img or 'IWZC' in img or 'UUZA' in img:
			Homography_path_model = 'Step4/Master_CaseAssy_Step4_IVZA_IWZC_UUZA_IVZA3613292_20230622_1021230839_'
			pts_dst_model=[[tl1,tl2],[tr1,tr2],[bl1,bl2],[br1,br2]]
		elif 'VA3' in img or 'VCZA' in img or 'VDZA' in img:
			Homography_path_model = 'VA3/Master_CaseAssy_Va3_VCZA_VDZA_VCZA3603922_20230615_0736153237_'
			pts_dst_model=[[tl1,tl2],[tr1,tr2],[bl1,bl2],[br1,br2]]

		# Crop / Resize keeping geometric proportions
		skew_file = img
		XY, image_loaded = find_ref(
		skew_file,
		Homography_path+Homography_path_model+'top_left.Tif',
		Homography_path+Homography_path_model+'top_right.Tif',
		Homography_path+Homography_path_model+'bottom_left.Tif',
		Homography_path+Homography_path_model+'bottom_right.Tif')
		# Read source image.
		im_src = image_loaded
		pts_src = np.array([XY[0], XY[1], XY[2],XY[3]])
		# Read destination image.
		# print(Homography_path+Homography_path_model+'resized.Tif')
		im_dst = cv2.imread(Homography_path+Homography_path_model+'resized.Tif')
		pts_dst = np.array(pts_dst_model)
		print(pts_dst_model, 'vs', XY[0], XY[1], XY[2],XY[3])
		# Calculate Homography
		h, status = cv2.findHomography(pts_src, pts_dst)
		# Warp source image to destination based on homography
		image_homo = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
		return image_homo

def generate_homography_and_warp(source_img_pil, target_img_pil):
    # Étape 1: Convertir les images PIL en tableaux NumPy
    source_img_np = np.array(source_img_pil)
    target_img_np = np.array(target_img_pil)

    # Utiliser les tableaux NumPy avec SIFT, etc.
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(source_img_np, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(target_img_np, None)

    # Les étapes suivantes restent inchangées...
    # Appariement des descripteurs de points clés
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Filtrage des appariements
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Extraction des positions des bons appariements
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calcul de la matrice d'homographie H
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


    # Étape finale: Application de l'homographie et conversion du résultat en image PIL si nécessaire
    warped_image_np = cv2.warpPerspective(source_img_np, H, (target_img_np.shape[1], target_img_np.shape[0]))
    

    return warped_image_np  # Retourner l'image résultante en tant qu'objet PIL Image

def calculate_ssim(imageA, imageB):
    
    # Calculer le SSIM entre les deux images
    score, diff = ssim(imageA, imageB, win_size=7, channel_axis=2,  full=True)
    print(f"SSIM: {score}")

    # La différence est une image flottante dans la plage [-1, 1], donc nous la transformons en une image en niveaux de gris de 8 bits.
    diff = (diff * 255).astype("uint8")

    return diff, score

def enhance(img):
	def adjust_gamma(image, gamma=1.0):
		# build a lookup table mapping the pixel values [0, 255] to
		# their adjusted gamma values
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
		# apply gamma correction using the lookup table
		return cv2.LUT(image, table)
	img_b_enhancer = ImageEnhance.Brightness(img)
	img_enhanced = img_b_enhancer.enhance(2)
	img_c_enhancer = ImageEnhance.Contrast(img_enhanced)
	img_enhanced = img_c_enhancer.enhance(0.7)
	img_s_enhancer = ImageEnhance.Sharpness(img_enhanced)
	img_enhanced = img_s_enhancer.enhance(2)
	return Image.fromarray(adjust_gamma(np.array(img_enhanced), gamma=2.5))

def test_difference(image_np):
	

	# Augmenter le contraste en utilisant le seuillage
	
	# Appliquer le filtre de Laplace pour détecter les changements rapides d'intensité
	laplacian = cv2.Laplacian(image_np, cv2.CV_64F)

	# Trouver les valeurs absolues pour mettre en évidence à la fois les contours clairs et foncés
	laplacian_abs = np.absolute(laplacian)

	# Convertir en type 'uint8'
	laplacian_uint8 = np.uint8(laplacian_abs)

	# Normaliser l'image
	laplacian_normalized = cv2.normalize(laplacian_uint8, None, alpha=0, beta=255, norm_type=cv2.NORM_L2)

	# Seuiller l'image pour isoler les zones de fort changement
	_, thresholded = cv2.threshold(laplacian_normalized, thresh=0, maxval=255, type=cv2.THRESH_BINARY)


	# Sauvegarder le résultat
	cv2.imwrite('enhanced_image.png', thresholded)
	return thresholded

def custom_to_pil2(x):
	
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()

    # Convertir le tableau NumPy en une image PIL
	x_pil = Image.fromarray(x)
	return x_pil


def reconstruction_pipeline(image, name, filepath_directory_OUT, filepath_config_DL_Model, filepath_DL_Model, patch_size, patch_count, scale_max, scale_step, worst_patch_count,model,size=320):


	global metrics
	global image_to_save
	global current_file
	to_pil = ToPILImage()

	config1024 = load_config(filepath_config_DL_Model)
	model1024 = load_vqgan(config1024, ckpt_path=filepath_DL_Model).to(DEVICE)

	print('Image preprocess')
	x_ori_initial = to_pil(image)#Image.open(filepath_picture_IN)
	#x_ori_initial=x_ori_initial.resize((size,size), Image.LANCZOS,reducing_gap=10.0).convert('L')
	#x_vqgan = preprocess(Image.fromarray(np.array(x_ori_initial)).convert('L'), target_image_size=size)
	
	x_vqgan = preprocess(Image.fromarray(np.array(x_ori_initial)), target_image_size=size)
	
	x_vqgan = x_vqgan.to(DEVICE)

	print(f"input is of size: {x_vqgan.shape}")
	#x_ori = custom_to_pil(preprocess_vqgan(x_vqgan[0])) # x_vqgan torch [B,H,W][0]=[H,W]] => -1,509 => image PIL [H,W] : 0,255

	#reco_prep = np.array(x_ori).astype(np.uint8)
	#reco_prep = (reco_prep/127.5 - 1.0).astype(np.float32)
	#reco_prep = torch.unsqueeze(T.ToTensor()(reco_prep), 0) # x_ori image PIL [H,W] : 0,255 converted in a torch [B,H,W]
	#reco_prep = reco_prep.to(DEVICE)

	print('Raw Reconstruction')
	#print("reco_prep", reco_prep.shape)
	reco , xrec_e , emb_loss , loss_tensor = reconstruct_with_vqgan(x_vqgan.to(DEVICE), model1024) # x_ori image PIL [H,W] : 0,255 => reco torch [H,W] : 0,255 converted in a torch [B,H,W]
	print("reco ok")
	print('reco', reco)
	#x_reco = custom_to_pil(reco) # PIL [B,H,W]

	#dict_loss = (preprocess_vqgan(x_vqgan), model1024)
	dict_loss = model1024.test_step(x_vqgan,batch_idx=0)
	
	metrics = {}
	print("dict_loss", dict_loss)
	metrics['ae_total_loss']=dict_loss[0]['test/total_loss'].item()
	metrics['p_loss']=dict_loss[0]['test/p_loss'].item()
	metrics['rec_loss']=dict_loss[0]['test/rec_loss'].item()
	metrics['quant_loss']=dict_loss[0]['test/quant_loss'].item()

    #x_ori_homo = Image.fromarray(x_ori_initial)
	#print(type(x_ori_initial))
	x_ori_initial = np.array(x_ori_initial)
    
	x_ori_homo = Image.fromarray(x_ori_initial)
    #mask_idiff = np.loadtxt(Mask_HD)

	print("test")
	#image_homo = generate_homography_and_warp(x_ori, x_reco)
	
	

	reco_prep_homo = np.array(x_ori_homo).astype(np.uint8)
	#reco_prep_homo = (reco_prep_homo/127.5 - 1.0).astype(np.float32)
	reco_prep_homo = torch.unsqueeze(T.ToTensor()(reco_prep_homo), 0) # x_ori image PIL [H,W] : 0,255 converted in a torch [B,H,W]
	reco_prep_homo = reco_prep_homo.to(DEVICE)

	print('Preporcess Reconstruction')

	reco_homo, _ , _, _ = reconstruct_with_vqgan(reco_prep_homo, model1024) # x_ori image PIL [H,W] : 0,255 => reco torch [H,W] : 0,255 converted in a torch [B,H,W]
	x_reco_homo = custom_to_pil(reco_homo.squeeze(0)) 

	length = (size / patch_size) ** 2
	max_index = [patch_count if length >= patch_count else length][0]
	difference_visu,highest_diff_patches_mask_windows_upscaled_all_visu,highest_fid_windows_upscaled_all_visu,loss_tensor_mat_upscaled_visu,disc_d_loss_direct_seq_upscale_visu = {},{},{},{},{}
	for weighted in ['weighted']:
		print('Case',weighted)
		weight = 1 # [1 if weighted == 'not_weighted' else (1-mask_idiff)**10][0]
		weight_z = 1

		#x_ori_enhanced = enhance(x_ori_homo)
		
		#x_reco_enhanced = enhance(x_reco_homo)
		#print("x_reco_enhanced", x_reco_enhanced)
		#x_ori_enhanced_masked = hard_mask(x_ori_enhanced)
		
		x_ori_enhanced_masked = x_ori_homo
        
        
		#x_reco_enhanced_masked = hard_mask(x_reco_enhanced)
		x_reco_enhanced_masked = x_reco_homo
        

		print('  Diff image')
		print(x_ori_enhanced_masked.size, x_ori_enhanced_masked.mode)
		print(x_reco_enhanced_masked.size, x_reco_enhanced_masked.mode)

		
		# diff_img = np.array(ImageChops.difference(x_ori_enhanced_masked,x_reco_enhanced_masked).getdata()) #*([weight.flatten() if weighted == 'weighted' else weight][0])
		# print("diff img", diff_img.shape)
		diff_img = np.array(ImageChops.difference(x_ori_enhanced_masked,x_reco_enhanced_masked)) #*([weight.flatten() if weighted == 'weighted' else weight][0])
		print("diff img", diff_img.shape)

		#diff_img = np.reshape(diff_img, (size, size))  # Redimensionnez si nécessaire
		

		diff_img = diff_img.astype(np.uint8)  # Convertissez le type de données en uint8
		# Assurez-vous que diff_img est un array NumPy avec le bon type et la bonne forme
		
		
		diff_ssim, score_ssim = calculate_ssim(np.array(x_ori_enhanced_masked), np.array(x_reco_enhanced_masked))
		diff_ssim = (diff_ssim * 255).astype("uint8")
		
		image_ori_blurred = cv2.GaussianBlur(np.array(x_ori_enhanced_masked), (5, 5), 0)
		image_reco_blurred = cv2.GaussianBlur(np.array(x_reco_enhanced_masked), (5, 5), 0)

		# Calculer la différence
		diff = cv2.absdiff(image_ori_blurred, image_reco_blurred)
		# Définissez une valeur de seuil basée sur votre analyse des images
		seuil = 15  # Cette valeur peut nécessiter un ajustement

		# Appliquer un seuillage
		_, diff_thresh = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)

		# Calculer la différence
		diff = cv2.absdiff(image_ori_blurred, image_reco_blurred)

		# Appliquer un seuillage
		_, diff_thresh = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)

		kernel = np.ones((3,3), np.uint8)
		diff_morph = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)


		# aire_minimale = 10  # exemple de valeur, nécessite ajustement

		# contours, _ = cv2.findContours(diff_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# for cnt in contours:
		# 	if cv2.contourArea(cnt) < aire_minimale:
		# 		cv2.drawContours(diff_morph, [cnt], 0, 0, -1)

		# Extraire les parties du chemin
		path_parts = name.split(os.sep)

		# Construire le nouveau nom de fichier en prenant la dernière partie 'good' et en l'ajoutant au nom du fichier
		new_filename = f"{path_parts[-2]}_{path_parts[-1]}"

		# Sauvegarder l'image résultante
		output_path_edges = new_filename
		#cv2.imwrite(output_path_edges, diff_morph)
		
		test = test_difference(diff_img)
        # Convertissez l'array NumPy en image PIL
        
		difference_visu[weighted] = Image.fromarray(diff_img)

		#difference_visu[weighted] = Image.fromarray(np.reshape(diff_img, (size,size)))
		
		print('  Highest Patches')
		#Construct the highest difference patches
		#diff_patches = reduce(np.reshape(diff_img, (size,size)) * (weight), '(h h2) (w w2)-> h w', 'max', h2=patch_size, w2=patch_size)
		# Assuming diff_img_pil is your PIL Image object with RGB channels
		diff_img_pil = Image.fromarray(diff_img.astype(np.uint8))  # Ensure it's uint8
		gray_img_pil = diff_img_pil.convert('L')  # Convert to grayscale

		# Convert back to numpy array if needed
		gray_img_array = np.array(gray_img_pil)
		diff_patches = reduce((gray_img_array) * (weight), '(h h2) (w w2)-> h w', 'max', h2=patch_size, w2=patch_size)
		
		highest_diff_patches_idx = np.argpartition(diff_patches.flatten(), -max_index)[-max_index:]
		highest_diff_patches = diff_patches.flatten()[highest_diff_patches_idx]
		highest_diff_patches_mask = np.where(diff_patches >= min(highest_diff_patches), diff_patches, 0)
		highest_diff_patches_mask = highest_diff_patches_mask.reshape((size // patch_size, size // patch_size))
		highest_diff_patches_mask_windows_idx = np.where(highest_diff_patches_mask != 0)
		highest_diff_patches_mask_windows_upscaled_all = 0
		
		print('  SSE / Patches')
		patch_metrics_list = ['fid','frechet','mse','directed_hausdorff','ssim','wasserstein','js','cosine','hamming','euclidean','sqeuclidean','minkowski','correlation','cityblock','chebyshev','canberra','braycurtis']
		patch_metrics_value_list = []
		patch_FID_value_list = []

		# highest_patch_time = datetime.datetime.now()
		for scale in range(1,scale_max,scale_step):
			#Construct windows of the highest different patches
			highest_diff_patches_mask_windows = np.zeros(highest_diff_patches_mask.shape)
			for i in range(len(highest_diff_patches_mask_windows_idx[0])):
				for scale_range_x in range(scale,0,-1):
					for scale_range_y in range(scale,0,-1):
						for x_shift in [-scale_range_x,0,scale_range_x]:
							for y_shift in [-scale_range_y,0,scale_range_y]:
								highest_diff_patches_mask_windows[
								[0 if highest_diff_patches_mask_windows_idx[0][i]+x_shift <= 0 else (size // patch_size)-1 if highest_diff_patches_mask_windows_idx[0][i]+x_shift >= (size // patch_size)-1 else highest_diff_patches_mask_windows_idx[0][i]+x_shift][0],
								[0 if highest_diff_patches_mask_windows_idx[1][i]+y_shift <= 0 else (size // patch_size)-1 if highest_diff_patches_mask_windows_idx[1][i]+y_shift >= (size // patch_size)-1 else highest_diff_patches_mask_windows_idx[1][i]+y_shift][0]
								]=255
			highest_diff_patches_mask_windows_upscaled = np.repeat(highest_diff_patches_mask_windows, patch_size, axis=0)
			highest_diff_patches_mask_windows_upscaled = np.repeat(highest_diff_patches_mask_windows_upscaled, patch_size, axis=1)
			highest_diff_patches_mask_windows_upscaled = np.array(highest_diff_patches_mask_windows_upscaled.astype(np.uint8)[:, :])
			
            #Collect all windows of the highest different patches
			highest_diff_patches_mask_windows_upscaled_all += highest_diff_patches_mask_windows_upscaled

			#Zoom out of the windows (scale)
			for i in range(max_index):
				x1 = [0 if (highest_diff_patches_mask_windows_idx[0][i]-scale)*patch_size <= 0 else (size)-1 if (highest_diff_patches_mask_windows_idx[0][i]-scale)*patch_size >= (size)-1 else (highest_diff_patches_mask_windows_idx[0][i]-scale)*patch_size][0]
				x2 = [0 if (highest_diff_patches_mask_windows_idx[0][i]+scale)*patch_size <= 0 else (size)-1 if (highest_diff_patches_mask_windows_idx[0][i]+scale)*patch_size >= (size)-1 else (highest_diff_patches_mask_windows_idx[0][i]+scale)*patch_size][0]
				y1 = [0 if (highest_diff_patches_mask_windows_idx[1][i]-scale)*patch_size <= 0 else (size)-1 if (highest_diff_patches_mask_windows_idx[1][i]-scale)*patch_size >= (size)-1 else (highest_diff_patches_mask_windows_idx[1][i]-scale)*patch_size][0]
				y2 = [0 if (highest_diff_patches_mask_windows_idx[1][i]+scale)*patch_size <= 0 else (size)-1 if (highest_diff_patches_mask_windows_idx[1][i]+scale)*patch_size >= (size)-1 else (highest_diff_patches_mask_windows_idx[1][i]+scale)*patch_size][0]

				shift_x, shift_y = highest_diff_patches_mask_windows_upscaled[x1:x2,y1:y2].shape[0]-1, highest_diff_patches_mask_windows_upscaled[x1:x2,y1:y2].shape[1]-1
				max_shift  = highest_diff_patches_mask_windows_upscaled.shape[0]-1
				x1m, x2p = [x1-shift_x if x1-shift_x>=0 else 0][0], [x2+shift_x if x2+shift_x<=max_shift else max_shift][0]
				y1m, y2p = [y1-shift_y if y1-shift_y>=0 else 0][0], [y2+shift_y if y2+shift_y<=max_shift else max_shift][0]
				coord_list = [(x1m,x1,y1m,y1,1),(x1m,x1,y1,y2,2),(x1m,x1,y2,y2p,3),(x1,x2,y1m,y1,4),(x1,x2,y1,y2,5),(x1,x2,y2,y2p,6),(x2,x2p,y1m,y1,7),(x2,x2p,y1,y2,8),(x2,x2p,y2,y2p,9)]
				if x1m-x1 == 0:
					for ele in [(x1m,x1,y1m,y1,1),(x1m,x1,y1,y2,2),(x1m,x1,y2,y2p,3)]:
						try:
							coord_list.remove(ele)
						except:
							pass
				if y1m-y1 == 0:
					for ele in [(x1m,x1,y1m,y1,1),(x1,x2,y1m,y1,4),(x2,x2p,y1m,y1,7)]:
						try:
							coord_list.remove(ele)
						except:
							pass
				if x2-x2p == 0:
					for ele in [(x2,x2p,y1m,y1,7),(x2,x2p,y1,y2,8),(x2,x2p,y2,y2p,9)]:
						try:
							coord_list.remove(ele)
						except:
							pass
				if y2-y2p == 0:
					for ele in [(x1m,x1,y2,y2p,3),(x1,x2,y2,y2p,6),(x2,x2p,y2,y2p,9)]:
						try:
							coord_list.remove(ele)
						except:
							pass

				# FID_time = datetime.datetime.now()
				#Shift around the windows (+1 and -1; x and y)
				coord_i = 0
				for coord in coord_list:
					coord_i +=1
					bbx1,bbx2,bby1,bby2 = coord[0],coord[1],coord[2],coord[3]
					xc, yc = bbx2-bbx1, bby2-bby1
					input_sel = np.array(x_ori_enhanced_masked)[bbx1:bbx2,bby1:bby2]
					reco_sel = np.array(x_reco_enhanced_masked)[bbx1:bbx2,bby1:bby2]

					weight_patch = 1 #[1 if weighted == 'not_weighted' else (np.mean(mask_idiff[bbx1:bbx2,bby1:bby2]))][0]
					#Compute input vs reco FID distances, among the Zoom/Shift windows
					# print('exec before FID_time :', i, coord_i, datetime.datetime.now()-FID_time)
					FID = np.sum(ImageChops.difference(Image.fromarray(input_sel),Image.fromarray(reco_sel)).getdata())
					# print('exec after FID_time :', i, coord_i, datetime.datetime.now()-FID_time)
					patch_FID_value=[FID,input_sel,reco_sel,bbx1,bbx2,bby1,bby2,xc,yc,i,scale]
					patch_FID_value_list.append(patch_FID_value)
				# print('exec FID_time :', i, datetime.datetime.now()-FID_time)

		highest_diff_patches_mask_windows_upscaled_all_visu[weighted] = Image.fromarray(highest_diff_patches_mask_windows_upscaled_all)
		
		print('  Highest Zoom&Shift SSE Patches')

		# print('exec all highest_patch_time :', datetime.datetime.now()-highest_patch_time)
		highest_fid_windows_upscaled_all = 0
		disc_logits_real_patchGAN = []
		disc_logits_fake_patchGAN = []
		disc_d_loss_patchGAN = []
		loss_tensor_mat_list = []
		patch_coord_list = []
		loss_tensor_mat_centered,disc_logits_real_patchGAN_centered,disc_logits_fake_patchGAN_centered,disc_d_loss_patchGAN_centered = [],[],[],[]

		#Keep the 'worst_patch_count' FID windows among all the zoomed and shifted windows
		patch_FID_value_list_sorted = sorted(patch_FID_value_list, key=lambda x: x[0], reverse=True)
		min_size_patch = 7
		for patch_id in range(worst_patch_count):
			try:
				fid_patch_list = patch_FID_value_list_sorted[patch_id]
			except:
				break
			fid_patch_sel = fid_patch_list[0]
			input_sel = fid_patch_list[1]
			reco_sel = fid_patch_list[2]
			bbx1 = fid_patch_list[3]
			bbx2 = fid_patch_list[4]
			bby1 = fid_patch_list[5]
			bby2 = fid_patch_list[6]

			x_patch = fid_patch_list[7]
			y_patch = fid_patch_list[8]
			max_fid_patch_patch_count = fid_patch_list[9]
			max_fid_patch_scale = fid_patch_list[10]

			highest_fid_windows = np.zeros((size//patch_size, size//patch_size), dtype=np.uint8)
			for scale_range_x in range(max_fid_patch_scale,0,-1):
				for scale_range_y in range(max_fid_patch_scale,0,-1):
					for x_shift in [-scale_range_x,0,scale_range_x]:
						for y_shift in [-scale_range_y,0,scale_range_y]:
							highest_fid_windows[
							[0 if (highest_diff_patches_mask_windows_idx[0][max_fid_patch_patch_count]+x_shift) <= 0 else (size // patch_size)-1 if (highest_diff_patches_mask_windows_idx[0][max_fid_patch_patch_count]+x_shift) >= (size // patch_size)-1 else (highest_diff_patches_mask_windows_idx[0][max_fid_patch_patch_count]+x_shift)][0],
							[0 if (highest_diff_patches_mask_windows_idx[1][max_fid_patch_patch_count]+y_shift) <= 0 else (size // patch_size)-1 if (highest_diff_patches_mask_windows_idx[1][max_fid_patch_patch_count]+y_shift) >= (size // patch_size)-1 else (highest_diff_patches_mask_windows_idx[1][max_fid_patch_patch_count]+y_shift)][0]
							]=255-(patch_id*2)

			highest_fid_windows_upscaled = np.repeat(highest_fid_windows, patch_size, axis=0)
			highest_fid_windows_upscaled = np.repeat(highest_fid_windows_upscaled, patch_size, axis=1)
			highest_fid_windows_upscaled_all += highest_fid_windows_upscaled

		highest_fid_windows_upscaled_all_visu[weighted] = Image.fromarray(highest_fid_windows_upscaled_all)
		#cv2.imwrite('image_test.png',np.array(difference_visu['weighted']))

		
		
		

		diff_morph = custom_to_pil2(torch.from_numpy(diff_morph))
		diff_ssim = custom_to_pil2(torch.from_numpy(diff_ssim))

		#cv2.imwrite('image_test.png', diff_ssim)
		#diff_ssim.save('image_test.png')

		threshold_value = 150  # Exemple de valeur seuil
		diff_ssim_ = np.array(diff_ssim)

		# Appliquer un seuillage binaire

		_, mask = cv2.threshold(diff_ssim_, threshold_value, 255, cv2.THRESH_BINARY)

		# mask = diff_ssim.point(lambda p: p > 200 and 255)
		# mask = mask.convert('1')

		aire_minimale = 50  # exemple de valeur, nécessite ajustement

		gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		# Convert to 8-bit if not already
		mask = gray.astype('uint8')


		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			if cv2.contourArea(cnt) < aire_minimale:
				cv2.drawContours(mask, [cnt], 0, 0, -1)

		# Convertir le tableau Numpy modifié en image PIL
		mask_pil = Image.fromarray(mask)

		white_pixels = np.sum(mask != 0)
		print("white_pixels", white_pixels)
		# Calculer la proportion de pixels blancs par rapport au nombre total de pixels dans l'image
		total_pixels = mask_pil.size[0]*mask_pil.size[1]
		print(total_pixels)
		proportion_white = white_pixels / total_pixels
		print(proportion_white)


		image_to_save = stack_reconstructions(x_ori_homo, x_reco_homo, diff_morph , mask_pil , difference_visu['weighted'], highest_fid_windows_upscaled_all_visu['weighted'], titles=["Input", "Reconstruction", "Diff morpho", "Diff ssim", "Difference Weighted", "FID Input Patched Weighted"])

	return metrics, image_to_save, score_ssim, proportion_white


#def main(i_loop,root,current_file,filepath_picture_IN,filepath_directory_OUT,filepath_config_DL_Model,filepath_DL_Model,filepath_config_ML_Model,filepath_scaler_ML_Model,filepath_ML_Model,Mask_HD,Mask_LD,Homography_path,model):
def main(current_file, data_test ,filepath_directory_OUT,filepath_config_DL_Model,filepath_DL_Model, model):
	start_time = datetime.datetime.now()
	
	str_time = str(round(time()))
	i = 0
	patch_size = 2
	patch_count = 100 #25 #5 for debug (speedup) #100
	worst_patch_count = 15 #50 #25 #3 for debug (speedup) #250
	scale_step = 3 #4
	scale_max = 16 #12 #16

	status = 'TO_INFER'
	img_size=64 #736 #1024
	print("data_test", data_test)
	#try : 
	print("coucou")
	for i, data in enumerate(data_test):
		print("coucou2")
		print(i)
		print("data ", data)
        
		image = data['image'].squeeze(0)
		filename = data['path']

        
        # Diviser la chaîne de caractères en utilisant le slash comme séparateur
		chemin = filename[0]
		parties = chemin.split('/')

        # Sélectionner l'élément qui contient "good"
        # Dans votre cas, "good" se trouve deux positions avant le nom du fichier, donc nous utilisons -3
		categorie = parties[-2]
        
		filepath_picture_IN = filename
		i = 0
		for i, crop in enumerate(image):
			print("Crop", crop)
			name = str(filepath_picture_IN[0]) +'_crop_'+str(i) 
			# metrics, image_to_save = reconstruction_pipeline(filepath_picture_IN, filepath_directory_OUT, filepath_config_DL_Model, filepath_DL_Model, filepath_config_ML_Model, filepath_scaler_ML_Model, filepath_ML_Model, Mask_HD, Mask_LD, Homography_path, patch_size, patch_count, scale_max, scale_step, worst_patch_count, model, size=img_size)
			metrics, image_to_save, diff_img_2, proportion_white = reconstruction_pipeline(crop, name, filepath_directory_OUT, filepath_config_DL_Model, filepath_DL_Model, patch_size, patch_count, scale_max, scale_step, worst_patch_count, model, size=img_size) 
			test_file = os.path.basename(name)


            
			with open(current_file, "w") as f:
				f.write('idx,test_file,status,img_size,patch_size,patch_count,scale_max,scale_step,'+str([key for key in metrics.keys()]).replace('[','').replace(']','').replace("'","")+'\n')

			with open(current_file, "a") as f:
				f.write(f"{str(i)},"
				f"{test_file},"
				f"{status},"
				f"{str(img_size)},"
				f"{str(patch_size)},"
				f"{str(patch_count)},"
				f"{str(scale_max)},"
				f"{str(scale_step)},"
				f"{str([value for value in metrics.values()]).replace('[','').replace(']','')},\n")

			anomaly_score_ae_total_loss = np.round((metrics['ae_total_loss'])*10000,2)
			anomaly_score_quant_loss = np.round((metrics['quant_loss'])*10000,2)
			anomaly_score_rec_loss = np.round((metrics['rec_loss'])*10000,2)
			anomaly_score_p_loss = np.round((metrics['p_loss'])*10000,2)


			print("anomaly_score_ae_total_loss :", anomaly_score_ae_total_loss)
			print("anomaly_score_quant_loss :", anomaly_score_quant_loss)
			print("anomaly_score_rec_loss :", anomaly_score_rec_loss)
			print("anomaly_score_p_loss :", anomaly_score_p_loss)
            
            
			# if model == 'G270' : 
			if anomaly_score_ae_total_loss > 1600 :
			#(np.round((0.098361447*10000),2) + np.round((0.088987052*10000),2))/2 : # threshold = mean(AS_min_NG,AS_max_OK) for G270 test set
				status = 'NG'
				print('Image NG')
                
			else :
				if proportion_white < 0.01 :
					status = 'OK'
					print('Image OK')
                    
				else:
					status = 'NG'
					print('Image NG')
                
                

            # elif model == 'Step4' : 
            # 	if anomaly_score > (np.round((0.096601196*10000),2) + np.round((0.09114553*10000),2))/2 : # threshold = mean(AS_min_NG,AS_max_OK) for Step4 test set
            # 		status = 'NG'
            # 		print('Image NG')
            # 	else:
            # 		status = 'OK'
            # 		print('Image OK')

            # else :
            # 	if anomaly_score > (np.round((0.116251886*10000),2) + np.round((0.115176544*10000),2))/2 : # threshold = mean(AS_min_NG,AS_max_OK) for VA3 test set
            # 		status = 'NG'
            # 		print('Image NG')
            # 	else:
            # 		status = 'OK'
            # 		print('Image OK')
            #status = 'INFER'

			last_component = os.path.basename(os.path.normpath(name))[:-4]

            #fp_src = [v[:-1] for v in ['NG/','OK/'] if v in filepath_picture_IN][0]  #for v in ['NG/','NG_new/','OK/','OK_new/','OK_new_complement/','XP_MXP', 'OK - Copy','OK_Masking_idff_G270','OK_Masking_idff_Step4','OK_Masking_idff_VA3'] if v in filepath_picture_IN][0]

			image_to_save.save(filepath_directory_OUT+categorie+'_'+status+'_'+last_component+'_'+str(anomaly_score_ae_total_loss)+'_test'+'.png')
            #print(filepath_directory_OUT+status+'_'+test_file+'_'+str(anomaly_score_ae_total_loss)+'_test'+'.png')

            # print('exec time :', datetime.datetime.now()-start_time)
			exec_time = datetime.datetime.now()-start_time
			print('exec time :', exec_time)
			print('\n')
			#return (image_to_save, exec_time, status, str(round(anomaly_score_ae_total_loss*100,2)))

	print(f"Traitement du batch {i+1}/{len(dataloader)}")

	# except Exception as e:
	# 	print(f"Erreur lors du traitement du batch {i+1}: {e}")

if __name__ == "__main__":
	root_test = '/scratch/coolsa/datasets/mvtec_anomaly/screw/test'
	

	dataset_Test = CustomTest_crop(1024, root_test, transform=None, random_crops=5)

	data_test = DataLoader(dataset_Test, batch_size=1, shuffle=False)

	current_file = '/scratch/coolsa/THESE2024/VQGanoDIP_benchmark_propre/Test/log_result_crop_'+str(round(time()))+'.txt'
					
	output_dir = "/scratch/coolsa/THESE2024/VQGanoDIP_benchmark_propre/test/2024-04-16T08-31-27_custom_vqgan_1CH_crop_test2/"			
					
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	main(current_file, data_test, 
						filepath_directory_OUT=output_dir, 
						filepath_config_DL_Model="/scratch/coolsa/THESE2024/VQGanoDIP_benchmark_propre/logs/2024-04-13T01-19-04_custom_vqgan_1CH_screw_classique/configs/2024-04-13T01-19-04-project.yaml", 
						filepath_DL_Model="/scratch/coolsa/THESE2024/VQGanoDIP_benchmark_propre/logs/2024-04-13T01-19-04_custom_vqgan_1CH_screw_classique/checkpoints/last.ckpt", 
						model='VQGan')
