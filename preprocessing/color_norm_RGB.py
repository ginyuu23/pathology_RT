# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'c:/Users/ariken/Desktop/openslidewin/bin'
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


import matplotlib.pyplot as plt
import numpy as np
import cv2
import openslide

from pathml.core import HESlide
from pathml.preprocessing import StainNormalizationHE
#import staintools

# color normalization and RGB channel for patch
path = "D:/pathological/pathology_image/lusc_left/"
slidename = os.listdir(path)

for wsi in slidename:
    tilefile = os.path.splitext(wsi)
    TILE_path = os.getcwd() + '/' + tilefile[0] + '/' + 'scored'
    print("start color normalization:", TILE_path)
    tilefile = os.path.splitext(wsi)
    # print(tilefile)

    new_norm = os.getcwd() + '/' + tilefile[0] + '/' + 'norm'
    if not os.path.exists(new_norm):
        os.makedirs(new_norm)
    norm_path = new_norm

    new_norm_R = os.getcwd() + '/' + tilefile[0] + '/' + 'norm' + '/' + 'R'
    if not os.path.exists(new_norm_R):
        os.makedirs(new_norm_R)
    norm_R_path = new_norm_R

    new_norm_G = os.getcwd() + '/' + tilefile[0] + '/' + 'norm' + '/' + 'G'
    if not os.path.exists(new_norm_G):
        os.makedirs(new_norm_G)
    norm_G_path = new_norm_G

    new_norm_B = os.getcwd() + '/' + tilefile[0] + '/' + 'norm' + '/' + 'B'
    if not os.path.exists(new_norm_B):
        os.makedirs(new_norm_B)
    norm_B_path = new_norm_B

    new_hematoxylin = os.getcwd() + '/' + tilefile[0] + '/' + 'hematoxylin'
    if not os.path.exists(new_hematoxylin):
        os.makedirs(new_hematoxylin)
    hematoxylin_path = new_hematoxylin

    new_hematoxylin_R = os.getcwd() + '/' + tilefile[0] + '/' + 'hematoxylin' + '/' + 'R'
    if not os.path.exists(new_hematoxylin_R):
        os.makedirs(new_hematoxylin_R)
    hematoxylin_R_path = new_hematoxylin_R

    new_hematoxylin_G = os.getcwd() + '/' + tilefile[0] + '/' + 'hematoxylin' + '/' + 'G'
    if not os.path.exists(new_hematoxylin_G):
        os.makedirs(new_hematoxylin_G)
    hematoxylin_G_path = new_hematoxylin_G

    new_hematoxylin_B = os.getcwd() + '/' + tilefile[0] + '/' + 'hematoxylin' + '/' + 'B'
    if not os.path.exists(new_hematoxylin_B):
        os.makedirs(new_hematoxylin_B)
    hematoxylin_B_path = new_hematoxylin_B

    new_eosin = os.getcwd() + '/' + tilefile[0] + '/' + 'eosin'
    if not os.path.exists(new_eosin):
        os.makedirs(new_eosin)
    eosin_path = new_eosin

    new_eosin_R = os.getcwd() + '/' + tilefile[0] + '/' + 'eosin' + '/' + 'R'
    if not os.path.exists(new_eosin_R):
        os.makedirs(new_eosin_R)
    eosin_R_path = new_eosin_R

    new_eosin_G = os.getcwd() + '/' + tilefile[0] + '/' + 'eosin' + '/' + 'G'
    if not os.path.exists(new_eosin_G):
        os.makedirs(new_eosin_G)
    eosin_G_path = new_eosin_G

    new_eosin_B = os.getcwd() + '/' + tilefile[0] + '/' + 'eosin' + '/' + 'B'
    if not os.path.exists(new_eosin_B):
        os.makedirs(new_eosin_B)
    eosin_B_path = new_eosin_B

    patchname = os.listdir(TILE_path)
    # print(patchname)

    n = 0
    for patch in patchname:
        img_patch = os.path.join(TILE_path, patch)
        patch_name = os.path.splitext(patch)
        img = cv2.imread(img_patch)
        region = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        normalizer1 = StainNormalizationHE(target="normalize", stain_estimation_method="macenko")  # vahadane, macenko
        im1 = normalizer1.F(region)
        im1_name = patch_name[0] + '_norm' + '.png'
        # Split the RGB channel
        red1 = im1[:, :, 0]
        green1 = im1[:, :, 1]
        blue1 = im1[:, :, 2]
        im1_R_name = patch_name[0] + '_norm_R' + '.png'
        im1_G_name = patch_name[0] + '_norm_G' + '.png'
        im1_B_name = patch_name[0] + '_norm_B' + '.png'

        normalizer2 = StainNormalizationHE(target="hematoxylin", stain_estimation_method="macenko")
        im2 = normalizer2.F(region)
        im2_name = patch_name[0] + '_hematoxylin' + '.png'
        # Split the RGB channel
        red2 = im2[:, :, 0]
        green2 = im2[:, :, 1]
        blue2 = im2[:, :, 2]
        im2_R_name = patch_name[0] + '_hematoxylin_R' + '.png'
        im2_G_name = patch_name[0] + '_hematoxylin_G' + '.png'
        im2_B_name = patch_name[0] + '_hematoxylin_B' + '.png'

        normalizer3 = StainNormalizationHE(target="eosin", stain_estimation_method="macenko")
        im3 = normalizer3.F(region)
        im3_name = patch_name[0] + '_eosin' + '.png'
        # Split the RGB channel
        red3 = im3[:, :, 0]
        green3 = im3[:, :, 1]
        blue3 = im3[:, :, 2]
        im3_R_name = patch_name[0] + '_eosin_R' + '.png'
        im3_G_name = patch_name[0] + '_eosin_G' + '.png'
        im3_B_name = patch_name[0] + '_eosin_B' + '.png'

        # save the H&E normalized image
        cv2.imwrite(os.path.join(norm_path, im1_name), cv2.cvtColor(im1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(hematoxylin_path, im2_name), cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(eosin_path, im3_name), cv2.cvtColor(im3, cv2.COLOR_RGB2BGR))
        # save the RGB channel image,normalized image
        cv2.imwrite(os.path.join(norm_R_path, im1_R_name), red1)
        cv2.imwrite(os.path.join(norm_G_path, im1_G_name), green1)
        cv2.imwrite(os.path.join(norm_B_path, im1_B_name), blue1)
        # save the RGB channel image, hematoxylin
        cv2.imwrite(os.path.join(hematoxylin_R_path, im2_R_name), red2)
        cv2.imwrite(os.path.join(hematoxylin_G_path, im2_G_name), green2)
        cv2.imwrite(os.path.join(hematoxylin_B_path, im2_B_name), blue2)
        # save the RGB channel image, eosin
        cv2.imwrite(os.path.join(eosin_R_path, im3_R_name), red3)
        cv2.imwrite(os.path.join(eosin_G_path, im3_G_name), green3)
        cv2.imwrite(os.path.join(eosin_B_path, im3_B_name), blue3)

        n += 1
    print("finish normalized and RGB channel separation:", wsi)
print("-------------------------Mission Complete------------------------")

