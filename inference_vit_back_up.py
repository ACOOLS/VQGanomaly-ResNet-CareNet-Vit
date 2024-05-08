USE_CUDA = 'yes'

#import pycaret
#from pycaret.classification import *
import torchvision.utils as vutils
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

def numpy_to_pil(np_array):
    # Assurez-vous que le tableau numpy est en uint8 et en RGB
    if np_array.dtype != np.uint8:
        np_array = np.clip(np_array, 0, 255).astype('uint8')
    if np_array.ndim == 3 and np_array.shape[2] == 3:
        mode = 'RGB'
    else:
        mode = 'L'  # Mode niveaux de gris pour les images à un canal
    return Image.fromarray(np_array, mode)

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
	z, emb_loss, _ ,ze,loss_tensor = model.encode(x)
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


def stack_reconstructions(input, reconstruction, diff_morpho, titles=[]):
	
	print(reconstruction)
	#assert input.size == reconstruction.size == difference_w.size == highest_fid_windows_upscaled_all_visu_w.size == diff_ssim.size == diff_morpho.size
	w, h = input.size[0], input.size[1]
	img = Image.new("RGB", (3*w, h))
	img.paste(input, (0,0))
	img.paste(reconstruction, (1*w,0))
	img.paste(diff_morpho, (2*w,0))
	#img.paste(diff_ssim, (3*w,0))
	#img.paste(difference_w, (4*w,0))
	#img.paste(Image.blend(input.convert("RGBA"), highest_fid_windows_upscaled_all_visu_w.convert("RGBA"), 0.7), (5*w,0))

	for i, title in enumerate(titles):
		ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255)) # coordinates, text, color
	return img

def calculate_ssim(imageA, imageB):
    # Ensure images are on CPU and detached from the graph
    print("imageA", imageA.shape)
    print("imageB", imageB.shape)
    imageA = imageA.detach().cpu().numpy()
    imageB = imageB.detach().cpu().numpy()

    # Convert the data type to float32 if not already
    imageA = imageA.astype(np.float32).squeeze(0)
    imageB = imageB.astype(np.float32).squeeze(0)

    # Calculate SSIM using scikit-image

    score, diff = ssim(imageA, imageB, win_size=7, channel_axis=0, data_range=255, full=True)
    
    return score, diff


def tensor_to_pil(tensor):
    # Assurez-vous que le tensor est sur CPU et non sur GPU
    tensor = tensor.cpu()
    
    # Convertir de [C, H, W] à [H, W, C] si nécessaire
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Si le tensor a été normalisé, dénormalisez-le
    # Remarque: adaptez ces valeurs à votre cas spécifique, ici supposant [0, 1]
    tensor = tensor * 255
    
    # Convertir en numpy array et en entiers uint8
    numpy_image = tensor.numpy().astype('uint8')
    
    # Créer une image PIL
    pil_image = Image.fromarray(numpy_image)
    return pil_image

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



def reconstruct_image_from_crops(crops, full_image_shape, crop_size):
    num_crops_vertical = full_image_shape[0] // crop_size[0]  # Utilisation de crop_size[0] pour la hauteur
    num_crops_horizontal = full_image_shape[1] // crop_size[1]  # Utilisation de crop_size[1] pour la largeur

    reconstructed_image = np.zeros(full_image_shape, dtype=np.float32)  # Assurez-vous que la forme et le type sont corrects
    
    idx = 0
    for i in range(num_crops_vertical):
        for j in range(num_crops_horizontal):
            start_row = i * crop_size[0]  # Utilisation de crop_size[0] pour la hauteur
            start_col = j * crop_size[1]  # Utilisation de crop_size[1] pour la largeur

            # Transposer les crops si nécessaire (dépend de la disposition de vos données)
            crop = crops[idx].transpose(1, 2, 0)  # Supposé déjà dans le bon format HWC ici

            # Assignation du crop à la bonne position
            reconstructed_image[start_row:start_row + crop_size[0], start_col:start_col + crop_size[1]] = crop
            idx += 1

    return reconstructed_image


def reconstruct_image_from_diff(crops, full_diff_shape, crop_size):
    num_crops_vertical = full_diff_shape[1] // crop_size[1]
    num_crops_horizontal = full_diff_shape[2] // crop_size[2]

    reconstructed_image = np.zeros(full_diff_shape, dtype=np.uint8)  # Assurer le type de données correct

    idx = 0
    for i in range(num_crops_vertical):
        for j in range(num_crops_horizontal):
            start_row = i * crop_size[1]
            start_col = j * crop_size[2]
            crop = crops[idx]
            if crop.ndim == 2:  # Assurer que le crop a une troisième dimension
                crop = crop[:, :, np.newaxis]
            reconstructed_image[start_row:start_row + crop_size[1], start_col:start_col + crop_size[2]] = crop
            idx += 1

    return reconstructed_image




def ensure_correct_dimensions(image_array):
    if image_array.shape[0] == 3:  # Supposant que le premier axe est pour les canaux
        return image_array.transpose(1, 2, 0)  # Convertir de (C, H, W) à (H, W, C)
    return image_array


def save_image_np(np_array, file_path):
    if np_array.ndim == 3 and np_array.shape[0] in [1, 3]:  # Pour les formats de canaux 1 ou 3
        if np_array.shape[0] == 3:
            np_array = np_array.transpose(1, 2, 0)  # CHW à HWC pour RGB
        else:
            print("np_array", np_array.shape)
            np_array = np_array.squeeze(0)  # Enlever le canal s'il est seul
	


    # Convertir de float32 [0,1] à uint8 [0,255]
    np_array = (np.clip(np_array, 0, 1) * 255).astype(np.uint8)
    
    print("np_array", np_array)
    # Créer et sauvegarder l'image
    image = Image.fromarray(np_array)
    image.save(file_path)

def save_image_tensor(tensor, filename):
	if tensor.ndim == 5 and tensor.shape[:2] == (1, 1):
		tensor = tensor[0, 0]  # This selects the (3, 64, 64) part

	tensor = tensor.cpu()

	# Convert PyTorch tensor to numpy array
	if tensor.requires_grad:
		# If the tensor is part of a computation graph
		tensor = tensor.detach()
	np_array = tensor.numpy()

    # The numpy array should be in the format (H, W, C) for RGB images
    # If the channels are first (C, H, W), you need to transpose
	if np_array.ndim == 3 and np_array.shape[0] in {1, 3}:
		# Transpose the array to (H, W, C)
		np_array = np_array.transpose(1, 2, 0)

	# Handle single-channel (grayscale) images
	if np_array.ndim == 3 and np_array.shape[2] == 1:
        # Remove the channel dimension for grayscale images
		np_array = np_array.squeeze(axis=2)

    # Convert numpy array to PIL Image
	img = Image.fromarray(np_array)

    # Save the image
	img.save(filename)


def reconstruction_pipeline(image, name, filepath_directory_OUT, filepath_config_DL_Model, filepath_DL_Model, patch_size, patch_count, scale_max, scale_step, worst_patch_count,model,size=320):

	global metrics
	global image_to_save
	global current_file
	to_pil = ToPILImage()

	config1024 = load_config(filepath_config_DL_Model)
	model1024 = load_vqgan(config1024, ckpt_path=filepath_DL_Model).to(DEVICE)

	# Traiter les crops
	with torch.no_grad():  # Désactiver le calcul de gradient pour l'évaluation
		original_crops = []
		reconstructed_crops = []
		diff_crops_ssim = []
		diff_crops_morpho = []

		#print("image", image.shape)
		for i, crop in enumerate(image):
			#print(f"Traitement du crop numéro {i} avec dimensions {crop.shape}")
			#print("batch", batch.shape)
			original_crops.extend(crop.unsqueeze(0).numpy())  # Stocker les crops originaux
			reconstructed_crop , xrec_e , emb_loss , loss_tensor = reconstruct_with_vqgan(crop.unsqueeze(0).to(DEVICE), model1024) 
			#print("reconstructed_crop", reconstructed_crop.shape)
			dict_loss = model1024.test_step(crop.unsqueeze(0).to(DEVICE),batch_idx=0)
		
			#econstructed_crops.append(reconstructed_crop.squeeze(0).cpu().numpy())
			reconstructed_crops.extend(reconstructed_crop.cpu().numpy())  # Stocker les reconstructions
			
			#TEST 
			#print("reconstructed_crops", reconstructed_crops)
			score_ssim, diff_ssim = calculate_ssim(crop.unsqueeze(0), reconstructed_crop)
			print("score_ssim", score_ssim)
			#diff_ssim = (diff_ssim * 255).astype("uint8")
			print("crop type", crop.dtype)
			print("reconstructed_crop type ", reconstructed_crop.dtype)
			
			# image_ori_blurred = cv2.GaussianBlur(np.array(crop.unsqueeze(0)), (5, 5), 0)
			# image_reco_blurred = cv2.GaussianBlur(np.array(reconstructed_crop), (5, 5), 0)

			crop = crop.unsqueeze(0)  # Supposons que crop est (H, W)
			reconstructed_crop = reconstructed_crop.unsqueeze(0)

			# Calcul du SSIM
			#ssim_val = ssim(crop, reconstructed_crop, data_range=1.0, size_average=True)  # Assurez-vous que data_range correspond à l'échelle de vos images

			# Pour obtenir la carte de différence SSIM
			
			
	
			diff_crops_ssim.extend(diff_ssim)
			print(f"Élément ajouté à l'indice {i}: {diff_ssim}")
			print(f"Taille actuelle de la liste: {len(diff_crops_ssim)}")
			# # Calculer la différence
			diff_morph = torch.abs(crop.cpu() - reconstructed_crop.cpu())
			diff_crops_morpho.extend(diff_morph.cpu())

			
			vutils.save_image(diff_morph.squeeze(0).squeeze(0), 'Test/diff_morph.png')
			save_image_np(diff_ssim, 'Test/diff_ssim.png')
			# # Définissez une valeur de seuil basée sur votre analyse des images
			seuil = 15  # Cette valeur peut nécessiter un ajustement

			# # Appliquer un seuillage
			# _, diff_thresh = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)

			# # Calculer la différence
			# diff = cv2.absdiff(image_ori_blurred, image_reco_blurred)

			# # Appliquer un seuillage
			# _, diff_thresh = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)

			# kernel = np.ones((3,3), np.uint8)
			# diff_morph = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
			#print("diff_ssim", diff_ssim.shape)
			#diff_crops.extend(diff_ssim)  # Stocker les reconstructions
			
	



	print('Preprocess Reconstruction')
	# Reconstruire les images de 16x16 crops
	full_image_shape = (1024, 1024, 3)  # Définir la forme complète de l'image désirée
	full_diff_shape = (3, 1024, 1024)
	crop_size = (64, 64, 3)  # Taille de chaque crop
	crops_size_diff = (3, 64, 64)
	reconstructed_image = reconstruct_image_from_crops(reconstructed_crops, full_image_shape, crop_size)
	input_image = reconstruct_image_from_crops(original_crops, full_image_shape, crop_size)
	#diff_image_morpho = reconstruct_image_from_diff(diff_crops_morpho, full_diff_shape, crops_size_diff)
	
	patches_tensor_morpho = torch.stack(diff_crops_morpho, dim=0)

	full_image = torch.zeros(3, 1024, 1024)
	index = 0
	for i in range(16):  # 16 lignes de patches
		for j in range(16):  # 16 colonnes de patches
			full_image[:, i*64:(i+1)*64, j*64:(j+1)*64] = patches_tensor_morpho[index]
			index += 1

	# Continue with saving the image

	torchvision.utils.save_image(full_image, 'Test/diff_image_morpho.png')
	

	print("Nombre d'éléments dans diff_crops_ssim:", len(diff_crops_ssim))

	diff_image_ssim = reconstruct_image_from_crops(diff_crops_ssim, full_image_shape, crop_size)
	
	#print("diff_image", diff_image.dtype)
	
	
	# # Sauvegarder l'image reconstruite
	save_image_np(reconstructed_image, 'Test/reconstructed_image.png')

	# # Sauvegarder l'image originale pour comparaison
	save_image_np(input_image, 'Test/original_image.png')
	# # Sauvegarder l'image originale pour comparaison
 
	#vutils.save_image(diff_image_morpho, 'Test/diff_image_morpho.png')
	save_image_tensor(diff_image_ssim, 'Test/diff_image_ssim.png')

	

	
	# size = 1024
	# length = (size / patch_size) ** 2
	# max_index = [patch_count if length >= patch_count else length][0]
	# difference_visu,highest_diff_patches_mask_windows_upscaled_all_visu,highest_fid_windows_upscaled_all_visu,loss_tensor_mat_upscaled_visu,disc_d_loss_direct_seq_upscale_visu = {},{},{},{},{}
	# for weighted in ['weighted']:
	# 	print('Case',weighted)
	# 	weight = 1 # [1 if weighted == 'not_weighted' else (1-mask_idiff)**10][0]
	# 	weight_z = 1

	# 	#x_ori_enhanced_masked = x_ori_homo
        
        
	# 	#x_reco_enhanced_masked = hard_mask(x_reco_enhanced)
	# 	#x_reco_enhanced_masked = x_reco_homo
        

	# 	print('  Diff image')
		
	# 	# Supposons que `input_image` est votre tableau numpy
	# 	# Vérifiez d'abord les dimensions
	# 	print("Dimensions originales:", input_image.shape)

	# 	# Si les dimensions ne sont pas correctes, réorganisez-les
	# 	if len(input_image.shape) == 3 and input_image.shape[0] == 3:  # Supposant format CHW (Canaux, Hauteur, Largeur)
	# 		# Convertir de CHW à HWC
	# 		input_image = input_image.transpose(1, 2, 0)

	# 	if len(reconstructed_image.shape) == 3 and reconstructed_image.shape[0] == 3:  # Supposant format CHW (Canaux, Hauteur, Largeur)
	# 		# Convertir de CHW à HWC
	# 		reconstructed_image = reconstructed_image.transpose(1, 2, 0)


	# 	print("Dimensions après transposition:", input_image.shape)


	# 	# Supposons que input_image et reconstructed_image sont des numpy ndarrays
	# 	input_tensor = torch.from_numpy(input_image)
	# 	reconstructed_tensor = torch.from_numpy(reconstructed_image)

	# 	# Application de la fonction abs() de PyTorch
	# 	diff_img = torch.abs(input_tensor - reconstructed_tensor)

	# 	# diff_img = np.array(ImageChops.difference(x_ori_enhanced_masked,x_reco_enhanced_masked).getdata()) #*([weight.flatten() if weighted == 'weighted' else weight][0])
	# 	# print("diff img", diff_img.shape)
	# 	#diff_img = torch.abs(input_image - reconstructed_image)
	# 	#diff_img = np.array(ImageChops.difference(numpy_to_pil(input_image), numpy_to_pil(reconstructed_image)).getdata()) #*([weight.flatten() if weighted == 'weighted' else weight][0])
	# 	print("diff img", diff_img)

	# 	diff_img = np.reshape(diff_img, (size, size, 3))  # Redimensionnez si nécessaire
	# 	print("diff img", diff_img)

	# 	#diff_img = diff_img.astype(np.uint8)  # Convertissez le type de données en uint8
	# 	# Assurez-vous que diff_img est un array NumPy avec le bon type et la bonne forme
		
		
	# 	diff_ssim, score_ssim = calculate_ssim( input_image, reconstructed_image)
	# 	diff_ssim = (diff_ssim * 255).astype("uint8")
		
	# 	image_ori_blurred = cv2.GaussianBlur(np.array(input_image), (5, 5), 0)
	# 	image_reco_blurred = cv2.GaussianBlur(np.array(reconstructed_image), (5, 5), 0)

	# 	# Calculer la différence
	# 	diff = cv2.absdiff(image_ori_blurred, image_reco_blurred)
	# 	# Définissez une valeur de seuil basée sur votre analyse des images
	# 	seuil = 15  # Cette valeur peut nécessiter un ajustement

	# 	# Appliquer un seuillage
	# 	_, diff_thresh = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)

	# 	# Calculer la différence
	# 	diff = cv2.absdiff(image_ori_blurred, image_reco_blurred)

	# 	# Appliquer un seuillage
	# 	_, diff_thresh = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)

	# 	kernel = np.ones((3,3), np.uint8)
	# 	diff_morph = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)


	# 	# aire_minimale = 10  # exemple de valeur, nécessite ajustement

	# 	# contours, _ = cv2.findContours(diff_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# 	# for cnt in contours:
	# 	# 	if cv2.contourArea(cnt) < aire_minimale:
	# 	# 		cv2.drawContours(diff_morph, [cnt], 0, 0, -1)

	# 	# Extraire les parties du chemin
	# 	path_parts = name.split(os.sep)

	# 	# Construire le nouveau nom de fichier en prenant la dernière partie 'good' et en l'ajoutant au nom du fichier
	# 	new_filename = f"{path_parts[-2]}_{path_parts[-1]}"

	# 	# Sauvegarder l'image résultante
	# 	output_path_edges = new_filename
	# 	#cv2.imwrite(output_path_edges, diff_morph)
		
	# 	#test = test_difference(diff_img)
    #     # Convertissez l'array NumPy en image PIL
        
	# 	#difference_visu[weighted] = tensor_to_pil(diff_img)
	# 	#difference_visu[weighted] = Image.fromarray(np.reshape(diff_img, (size,size)))
		
	# 	print('  Highest Patches')
	# 	#Construct the highest difference patches
	# 	#diff_patches = reduce(np.reshape(diff_img, (size,size)) * (weight), '(h h2) (w w2)-> h w', 'max', h2=patch_size, w2=patch_size)
	# 	# Assuming diff_img_pil is your PIL Image object with RGB channels
	# 	#diff_img_pil = tensor_to_pil(diff_img)  # Ensure it's uint8
	# 	#gray_img_pil = diff_img_pil.convert('L')  # Convert to grayscale
	# 	#size = 1024 

	# 	# # Convert back to numpy array if needed
	# 	# gray_img_array = np.array(gray_img_pil)
	# 	# print("gray_img_array", gray_img_array.shape)
	# 	# diff_patches = reduce((gray_img_array) * (weight), '(h h2) (w w2)-> h w', 'max', h2=patch_size, w2=patch_size)
	# 	# print("diff_patches", diff_patches.shape)
	# 	# highest_diff_patches_idx = np.argpartition(diff_patches.flatten(), -max_index)[-max_index:]
	# 	# print("highest_diff_patches_idx", highest_diff_patches_idx.shape)
	# 	# highest_diff_patches = diff_patches.flatten()[highest_diff_patches_idx]
	# 	# print("highest_diff_patches", highest_diff_patches.shape)
	# 	# highest_diff_patches_mask = np.where(diff_patches >= min(highest_diff_patches), diff_patches, 0)
	# 	# print("highest_diff_patches_mask", highest_diff_patches_mask.shape)
	# 	# print("size", size)
	# 	# highest_diff_patches_mask = highest_diff_patches_mask.reshape((size // patch_size, size // patch_size))
	# 	# highest_diff_patches_mask_windows_idx = np.where(highest_diff_patches_mask != 0)
	# 	# highest_diff_patches_mask_windows_upscaled_all = 0
		
	# 	# print('  SSE / Patches')
	# 	# patch_metrics_list = ['fid','frechet','mse','directed_hausdorff','ssim','wasserstein','js','cosine','hamming','euclidean','sqeuclidean','minkowski','correlation','cityblock','chebyshev','canberra','braycurtis']
	# 	# patch_metrics_value_list = []
	# 	# patch_FID_value_list = []

	# 	# # highest_patch_time = datetime.datetime.now()
	# 	# for scale in range(1,scale_max,scale_step):
	# 	# 	#Construct windows of the highest different patches
	# 	# 	highest_diff_patches_mask_windows = np.zeros(highest_diff_patches_mask.shape)
	# 	# 	print("ici : ", len(highest_diff_patches_mask_windows_idx[0]))
	# 	# 	for i in range(len(highest_diff_patches_mask_windows_idx[0])):
	# 	# 		for scale_range_x in range(scale,0,-1):
	# 	# 			for scale_range_y in range(scale,0,-1):
	# 	# 				for x_shift in [-scale_range_x,0,scale_range_x]:
	# 	# 					for y_shift in [-scale_range_y,0,scale_range_y]:
	# 	# 						highest_diff_patches_mask_windows[
	# 	# 						[0 if highest_diff_patches_mask_windows_idx[0][i]+x_shift <= 0 else (size // patch_size)-1 if highest_diff_patches_mask_windows_idx[0][i]+x_shift >= (size // patch_size)-1 else highest_diff_patches_mask_windows_idx[0][i]+x_shift][0],
	# 	# 						[0 if highest_diff_patches_mask_windows_idx[1][i]+y_shift <= 0 else (size // patch_size)-1 if highest_diff_patches_mask_windows_idx[1][i]+y_shift >= (size // patch_size)-1 else highest_diff_patches_mask_windows_idx[1][i]+y_shift][0]
	# 	# 						]=255
	# 	# 	highest_diff_patches_mask_windows_upscaled = np.repeat(highest_diff_patches_mask_windows, patch_size, axis=0)
	# 	# 	highest_diff_patches_mask_windows_upscaled = np.repeat(highest_diff_patches_mask_windows_upscaled, patch_size, axis=1)
	# 	# 	highest_diff_patches_mask_windows_upscaled = np.array(highest_diff_patches_mask_windows_upscaled.astype(np.uint8)[:, :])
			
    #     #     #Collect all windows of the highest different patches
	# 	# 	highest_diff_patches_mask_windows_upscaled_all += highest_diff_patches_mask_windows_upscaled

	# 	# 	#Zoom out of the windows (scale)
	# 	# 	for i in range(max_index):
	# 	# 		x1 = [0 if (highest_diff_patches_mask_windows_idx[0][i]-scale)*patch_size <= 0 else (size)-1 if (highest_diff_patches_mask_windows_idx[0][i]-scale)*patch_size >= (size)-1 else (highest_diff_patches_mask_windows_idx[0][i]-scale)*patch_size][0]
	# 	# 		x2 = [0 if (highest_diff_patches_mask_windows_idx[0][i]+scale)*patch_size <= 0 else (size)-1 if (highest_diff_patches_mask_windows_idx[0][i]+scale)*patch_size >= (size)-1 else (highest_diff_patches_mask_windows_idx[0][i]+scale)*patch_size][0]
	# 	# 		y1 = [0 if (highest_diff_patches_mask_windows_idx[1][i]-scale)*patch_size <= 0 else (size)-1 if (highest_diff_patches_mask_windows_idx[1][i]-scale)*patch_size >= (size)-1 else (highest_diff_patches_mask_windows_idx[1][i]-scale)*patch_size][0]
	# 	# 		y2 = [0 if (highest_diff_patches_mask_windows_idx[1][i]+scale)*patch_size <= 0 else (size)-1 if (highest_diff_patches_mask_windows_idx[1][i]+scale)*patch_size >= (size)-1 else (highest_diff_patches_mask_windows_idx[1][i]+scale)*patch_size][0]

	# 	# 		shift_x, shift_y = highest_diff_patches_mask_windows_upscaled[x1:x2,y1:y2].shape[0]-1, highest_diff_patches_mask_windows_upscaled[x1:x2,y1:y2].shape[1]-1
	# 	# 		max_shift  = highest_diff_patches_mask_windows_upscaled.shape[0]-1
	# 	# 		x1m, x2p = [x1-shift_x if x1-shift_x>=0 else 0][0], [x2+shift_x if x2+shift_x<=max_shift else max_shift][0]
	# 	# 		y1m, y2p = [y1-shift_y if y1-shift_y>=0 else 0][0], [y2+shift_y if y2+shift_y<=max_shift else max_shift][0]
	# 	# 		coord_list = [(x1m,x1,y1m,y1,1),(x1m,x1,y1,y2,2),(x1m,x1,y2,y2p,3),(x1,x2,y1m,y1,4),(x1,x2,y1,y2,5),(x1,x2,y2,y2p,6),(x2,x2p,y1m,y1,7),(x2,x2p,y1,y2,8),(x2,x2p,y2,y2p,9)]
	# 	# 		if x1m-x1 == 0:
	# 	# 			for ele in [(x1m,x1,y1m,y1,1),(x1m,x1,y1,y2,2),(x1m,x1,y2,y2p,3)]:
	# 	# 				try:
	# 	# 					coord_list.remove(ele)
	# 	# 				except:
	# 	# 					pass
	# 	# 		if y1m-y1 == 0:
	# 	# 			for ele in [(x1m,x1,y1m,y1,1),(x1,x2,y1m,y1,4),(x2,x2p,y1m,y1,7)]:
	# 	# 				try:
	# 	# 					coord_list.remove(ele)
	# 	# 				except:
	# 	# 					pass
	# 	# 		if x2-x2p == 0:
	# 	# 			for ele in [(x2,x2p,y1m,y1,7),(x2,x2p,y1,y2,8),(x2,x2p,y2,y2p,9)]:
	# 	# 				try:
	# 	# 					coord_list.remove(ele)
	# 	# 				except:
	# 	# 					pass
	# 	# 		if y2-y2p == 0:
	# 	# 			for ele in [(x1m,x1,y2,y2p,3),(x1,x2,y2,y2p,6),(x2,x2p,y2,y2p,9)]:
	# 	# 				try:
	# 	# 					coord_list.remove(ele)
	# 	# 				except:
	# 	# 					pass

	# 	# 		# FID_time = datetime.datetime.now()
	# 	# 		#Shift around the windows (+1 and -1; x and y)
	# 	# 		coord_i = 0
	# 	# 		for coord in coord_list:
	# 	# 			coord_i +=1
	# 	# 			bbx1,bbx2,bby1,bby2 = coord[0],coord[1],coord[2],coord[3]
	# 	# 			xc, yc = bbx2-bbx1, bby2-bby1
	# 	# 			input_sel = np.array(input_image)[bbx1:bbx2,bby1:bby2]
	# 	# 			reco_sel = np.array(reconstructed_image)[bbx1:bbx2,bby1:bby2]

	# 	# 			weight_patch = 1 #[1 if weighted == 'not_weighted' else (np.mean(mask_idiff[bbx1:bbx2,bby1:bby2]))][0]
	# 	# 			#Compute input vs reco FID distances, among the Zoom/Shift windows
	# 	# 			# print('exec before FID_time :', i, coord_i, datetime.datetime.now()-FID_time)
	# 	# 			FID = np.sum(ImageChops.difference(Image.fromarray(input_sel),Image.fromarray(reco_sel)).getdata())
	# 	# 			# print('exec after FID_time :', i, coord_i, datetime.datetime.now()-FID_time)
	# 	# 			patch_FID_value=[FID,input_sel,reco_sel,bbx1,bbx2,bby1,bby2,xc,yc,i,scale]
	# 	# 			patch_FID_value_list.append(patch_FID_value)
	# 	# 		# print('exec FID_time :', i, datetime.datetime.now()-FID_time)

	# 	# highest_diff_patches_mask_windows_upscaled_all_visu[weighted] = Image.fromarray(highest_diff_patches_mask_windows_upscaled_all)
		
	# 	# print('  Highest Zoom&Shift SSE Patches')

	# 	# # print('exec all highest_patch_time :', datetime.datetime.now()-highest_patch_time)
	# 	# highest_fid_windows_upscaled_all = 0
	# 	# disc_logits_real_patchGAN = []
	# 	# disc_logits_fake_patchGAN = []
	# 	# disc_d_loss_patchGAN = []
	# 	# loss_tensor_mat_list = []
	# 	# patch_coord_list = []
	# 	# loss_tensor_mat_centered,disc_logits_real_patchGAN_centered,disc_logits_fake_patchGAN_centered,disc_d_loss_patchGAN_centered = [],[],[],[]

	# 	# #Keep the 'worst_patch_count' FID windows among all the zoomed and shifted windows
	# 	# patch_FID_value_list_sorted = sorted(patch_FID_value_list, key=lambda x: x[0], reverse=True)
	# 	# min_size_patch = 7
	# 	# for patch_id in range(worst_patch_count):
	# 	# 	try:
	# 	# 		fid_patch_list = patch_FID_value_list_sorted[patch_id]
	# 	# 	except:
	# 	# 		break
	# 	# 	fid_patch_sel = fid_patch_list[0]
	# 	# 	input_sel = fid_patch_list[1]
	# 	# 	reco_sel = fid_patch_list[2]
	# 	# 	bbx1 = fid_patch_list[3]
	# 	# 	bbx2 = fid_patch_list[4]
	# 	# 	bby1 = fid_patch_list[5]
	# 	# 	bby2 = fid_patch_list[6]

	# 	# 	x_patch = fid_patch_list[7]
	# 	# 	y_patch = fid_patch_list[8]
	# 	# 	max_fid_patch_patch_count = fid_patch_list[9]
	# 	# 	max_fid_patch_scale = fid_patch_list[10]

	# 	# 	highest_fid_windows = np.zeros((size//patch_size, size//patch_size), dtype=np.uint8)
	# 	# 	for scale_range_x in range(max_fid_patch_scale,0,-1):
	# 	# 		for scale_range_y in range(max_fid_patch_scale,0,-1):
	# 	# 			for x_shift in [-scale_range_x,0,scale_range_x]:
	# 	# 				for y_shift in [-scale_range_y,0,scale_range_y]:
	# 	# 					highest_fid_windows[
	# 	# 					[0 if (highest_diff_patches_mask_windows_idx[0][max_fid_patch_patch_count]+x_shift) <= 0 else (size // patch_size)-1 if (highest_diff_patches_mask_windows_idx[0][max_fid_patch_patch_count]+x_shift) >= (size // patch_size)-1 else (highest_diff_patches_mask_windows_idx[0][max_fid_patch_patch_count]+x_shift)][0],
	# 	# 					[0 if (highest_diff_patches_mask_windows_idx[1][max_fid_patch_patch_count]+y_shift) <= 0 else (size // patch_size)-1 if (highest_diff_patches_mask_windows_idx[1][max_fid_patch_patch_count]+y_shift) >= (size // patch_size)-1 else (highest_diff_patches_mask_windows_idx[1][max_fid_patch_patch_count]+y_shift)][0]
	# 	# 					]=255-(patch_id*2)

	# 	# 	highest_fid_windows_upscaled = np.repeat(highest_fid_windows, patch_size, axis=0)
	# 	# 	highest_fid_windows_upscaled = np.repeat(highest_fid_windows_upscaled, patch_size, axis=1)
	# 	# 	highest_fid_windows_upscaled_all += highest_fid_windows_upscaled

	# 	# highest_fid_windows_upscaled_all_visu[weighted] = Image.fromarray(highest_fid_windows_upscaled_all)
	# 	# #cv2.imwrite('image_test.png',np.array(difference_visu['weighted']))

		
		
		

	# 	# diff_morph = custom_to_pil2(torch.from_numpy(diff_morph))
	# 	# diff_ssim = custom_to_pil2(torch.from_numpy(diff_ssim))

	# 	# #cv2.imwrite('image_test.png', diff_ssim)
	# 	# #diff_ssim.save('image_test.png')

	# 	# threshold_value = 150  # Exemple de valeur seuil
	# 	# diff_ssim_ = np.array(diff_ssim)

	# 	# # Appliquer un seuillage binaire

	# 	# _, mask = cv2.threshold(diff_ssim_, threshold_value, 255, cv2.THRESH_BINARY)

	# 	# # mask = diff_ssim.point(lambda p: p > 200 and 255)
	# 	# # mask = mask.convert('1')

	# 	# aire_minimale = 50  # exemple de valeur, nécessite ajustement

	# 	# gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	# 	# # Convert to 8-bit if not already
	# 	# mask = gray.astype('uint8')


	# 	# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# 	# for cnt in contours:
	# 	# 	if cv2.contourArea(cnt) < aire_minimale:
	# 	# 		cv2.drawContours(mask, [cnt], 0, 0, -1)

	# 	# # Convertir le tableau Numpy modifié en image PIL
	# 	# mask_pil = Image.fromarray(mask)

	# 	# white_pixels = np.sum(mask != 0)
	# 	# print("white_pixels", white_pixels)
	# 	# # Calculer la proportion de pixels blancs par rapport au nombre total de pixels dans l'image
	# 	# total_pixels = mask_pil.size[0]*mask_pil.size[1]
	# 	# print(total_pixels)
	# 	# proportion_white = white_pixels / total_pixels
	# 	# print(proportion_white)


	# 	image_to_save = stack_reconstructions( input_image , reconstructed_image, diff_morph , titles=["Input", "Reconstruction", "Diff morpho"])


	return input_image, reconstructed_image, diff_image


#def main(i_loop,root,current_file,filepath_picture_IN,filepath_directory_OUT,filepath_config_DL_Model,filepath_DL_Model,filepath_config_ML_Model,filepath_scaler_ML_Model,filepath_ML_Model,Mask_HD,Mask_LD,Homography_path,model):
#def main(current_file, data_test ,filepath_directory_OUT,filepath_config_DL_Model,filepath_DL_Model, model):
	

if __name__ == "__main__":
	root_test = '/home/aurelie/datasets/mvtec_anomaly/screw/test'
	

	dataset_Test = CustomTest_crop(1024, root_test, transform=None, random_crops=0)

	data_test = DataLoader(dataset_Test, batch_size=1, shuffle=False,  drop_last=False)

	current_file = '/home/aurelie/THESE/VQGanomaly-ResNet-CareNet-Vit/Test/log_result_crop_'+str(round(time()))+'.txt'
					
	output_dir = "/home/aurelie/THESE/VQGanomaly-ResNet-CareNet-Vit/Test"			
	filepath_directory_OUT=output_dir
	filepath_config_DL_Model="/home/aurelie/THESE/VQGanomaly-ResNet-CareNet-Vit/logs/2024-04-13T00-34-54_custom_vqgan_1CH_screw_vit/configs/2024-04-13T00-34-54-project.yaml"
	filepath_DL_Model="/home/aurelie/THESE/VQGanomaly-ResNet-CareNet-Vit/logs/2024-04-13T00-34-54_custom_vqgan_1CH_screw_vit/checkpoints/last.ckpt"
	model='VQGan'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# main(current_file, data_test, 
	# 					filepath_directory_OUT=output_dir, 
	# 					filepath_config_DL_Model="/home/aurelie/THESE/VQGanomaly-ResNet-CareNet-Vit/logs/2024-04-13T00-34-54_custom_vqgan_1CH_screw_vit/configs/2024-04-13T00-34-54-project.yaml", 
	# 					filepath_DL_Model="/home/aurelie/THESE/VQGanomaly-ResNet-CareNet-Vit/logs/2024-04-13T00-34-54_custom_vqgan_1CH_screw_vit/checkpoints/last.ckpt", 
	# 					model='VQGan')

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

	#for i, data in enumerate(data_test):
	for batch_idx, batch in enumerate(data_test):
		print(f"Batch {batch_idx}: {batch['image'].shape}")	
		
		image = batch['image'].squeeze(0)
		filename = batch['path']

		# Diviser la chaîne de caractères en utilisant le slash comme séparateur
		chemin = filename[0]
		parties = chemin.split('/')
		print("parties", parties)
		# Sélectionner l'élément qui contient "good"
		# Dans votre cas, "good" se trouve deux positions avant le nom du fichier, donc nous utilisons -3
		categorie = parties[-2]
		print("categorie", categorie)
		filepath_picture_IN = filename
		print("filepath_picture_IN", filepath_picture_IN)
		#exit(0)
		i = 0
		#for i, crop in enumerate(image):
		#	print("Crop", crop)
		name = str(filepath_picture_IN[0])
		input_image, reconstructed_image, diff_image = reconstruction_pipeline(image, name, filepath_directory_OUT, filepath_config_DL_Model, filepath_DL_Model, patch_size, patch_count, scale_max, scale_step, worst_patch_count, model, size=img_size)

		big_folder_input = "Input_model"
		bug_folder_output = "output_model"
		nom_image = f"{categorie}_{os.path.basename(chemin)}"
		print("nom image", nom_image)
		
		# Déterminer le dossier de sortie en fonction de la catégorie
		sub_folder = "normal" if categorie == "good" else "anormal"
		folder_input_path = os.path.join(big_folder_input, sub_folder)
		folder_output_path = os.path.join(bug_folder_output, sub_folder)

		# Assurez-vous que les dossiers existent
		os.makedirs(folder_input_path, exist_ok=True)
		os.makedirs(folder_output_path, exist_ok=True)

		# Sauvegarder les images
		input_image_path = os.path.join(folder_input_path, nom_image)
		output_image_path = os.path.join(folder_output_path, nom_image)

		# Utilisez votre fonction de sauvegarde d'image ici, par exemple cv2.imwrite ou une autre librairie
		save_image(input_image, input_image_path)
		save_image(reconstructed_image, output_image_path)


		print(f"Image sauvegardée: {input_image_path} et {output_image_path}")





		#metrics, image_to_save, diff_img_2, proportion_white = reconstruction_pipeline(crop, name, filepath_directory_OUT, filepath_config_DL_Model, filepath_DL_Model, patch_size, patch_count, scale_max, scale_step, worst_patch_count, model, size=img_size) 
		# test_file = os.path.basename(name)


            
		# with open(current_file, "w") as f:
		# 	f.write('idx,test_file,status,img_size,patch_size,patch_count,scale_max,scale_step,'+str([key for key in metrics.keys()]).replace('[','').replace(']','').replace("'","")+'\n')

		# with open(current_file, "a") as f:
		# 	f.write(f"{str(i)},"
		# 	f"{test_file},"
		# 	f"{status},"
		# 	f"{str(img_size)},"
		# 	f"{str(patch_size)},"
		# 	f"{str(patch_count)},"
		# 	f"{str(scale_max)},"
		# 	f"{str(scale_step)},"
		# 	f"{str([value for value in metrics.values()]).replace('[','').replace(']','')},\n")

		# anomaly_score_ae_total_loss = np.round((metrics['ae_total_loss'])*10000,2)
		# anomaly_score_quant_loss = np.round((metrics['quant_loss'])*10000,2)
		# anomaly_score_rec_loss = np.round((metrics['rec_loss'])*10000,2)
		# anomaly_score_p_loss = np.round((metrics['p_loss'])*10000,2)


		# print("anomaly_score_ae_total_loss :", anomaly_score_ae_total_loss)
		# print("anomaly_score_quant_loss :", anomaly_score_quant_loss)
		# print("anomaly_score_rec_loss :", anomaly_score_rec_loss)
		# print("anomaly_score_p_loss :", anomaly_score_p_loss)
		
		
		# # if model == 'G270' : 
		# if anomaly_score_ae_total_loss > 1600 :
		# #(np.round((0.098361447*10000),2) + np.round((0.088987052*10000),2))/2 : # threshold = mean(AS_min_NG,AS_max_OK) for G270 test set
		# 	status = 'NG'
		# 	print('Image NG')
			
		# else :
		# 	if proportion_white < 0.01 :
		# 		status = 'OK'
		# 		print('Image OK')
				
		# 	else:
		# 		status = 'NG'
		# 		print('Image NG')
			
			

		# 	last_component = os.path.basename(os.path.normpath(name))[:-4]

        #     #fp_src = [v[:-1] for v in ['NG/','OK/'] if v in filepath_picture_IN][0]  #for v in ['NG/','NG_new/','OK/','OK_new/','OK_new_complement/','XP_MXP', 'OK - Copy','OK_Masking_idff_G270','OK_Masking_idff_Step4','OK_Masking_idff_VA3'] if v in filepath_picture_IN][0]

		# 	image_to_save.save(filepath_directory_OUT+categorie+'_'+status+'_'+last_component+'_'+str(anomaly_score_ae_total_loss)+'_test'+'.png')
        #     #print(filepath_directory_OUT+status+'_'+test_file+'_'+str(anomaly_score_ae_total_loss)+'_test'+'.png')

        #     # print('exec time :', datetime.datetime.now()-start_time)
		# 	exec_time = datetime.datetime.now()-start_time
		# 	print('exec time :', exec_time)
		# 	print('\n')
		# 	#return (image_to_save, exec_time, status, str(round(anomaly_score_ae_total_loss*100,2)))

