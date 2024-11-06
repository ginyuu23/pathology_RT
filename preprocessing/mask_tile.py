import SimpleITK as sitk
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re #regular expression
import sys #System
import cv2 #OpenCv
import PIL #Pillow

path = "D:/pathological/pathology_image/lusc_left/"
base_path = "D:/pathological/pathology_image/"
slidename = os.listdir(path)

for wsi in slidename:
    tilefile = os.path.splitext(wsi)
    print("patient case:", tilefile[0])

    norm_R_path = os.path.join(base_path, tilefile[0], 'norm', 'R')
    norm_R_name = os.listdir(norm_R_path)
    norm_G_path = os.path.join(base_path, tilefile[0], 'norm', 'G')
    norm_G_name = os.listdir(norm_G_path)
    norm_B_path = os.path.join(base_path, tilefile[0], 'norm', 'B')
    norm_B_name = os.listdir(norm_B_path)

    hematoxylin_R_path = os.path.join(base_path, tilefile[0], 'hematoxylin', 'R')
    hematoxylin_R_name = os.listdir(hematoxylin_R_path)
    hematoxylin_G_path = os.path.join(base_path, tilefile[0], 'hematoxylin', 'G')
    hematoxylin_G_name = os.listdir(hematoxylin_G_path)
    hematoxylin_B_path = os.path.join(base_path, tilefile[0], 'hematoxylin', 'B')
    hematoxylin_B_name = os.listdir(hematoxylin_B_path)

    eosin_R_path = os.path.join(base_path, tilefile[0], 'eosin', 'R')
    eosin_R_name = os.listdir(eosin_R_path)
    eosin_G_path = os.path.join(base_path, tilefile[0], 'eosin', 'G')
    eosin_G_name = os.listdir(eosin_G_path)
    eosin_B_path = os.path.join(base_path, tilefile[0], 'eosin', 'B')
    eosin_B_name = os.listdir(eosin_B_path)

    for normRtile in norm_R_name:
        # print("start processing:",normtile)
        tilename_normR = os.path.join(norm_R_path, normRtile)
        # print(tilename_norm)
        im_normR = sitk.ReadImage(tilename_normR)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_normR.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        # Compute lower and upper bound of the ROI
        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        # ensure the ROI stays within the image bounds
        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_normR)

        patch_name = os.path.splitext(normRtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(norm_R_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)

    for normGtile in norm_G_name:
        # print("start processing:",normtile)
        tilename_normG = os.path.join(norm_G_path, normGtile)
        # print(tilename_norm)
        im_normG = sitk.ReadImage(tilename_normG)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_normG.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_normG)

        patch_name = os.path.splitext(normGtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(norm_G_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)

    for normBtile in norm_B_name:
        # print("start processing:",normtile)
        tilename_normB = os.path.join(norm_B_path, normBtile)
        # print(tilename_norm)
        im_normB = sitk.ReadImage(tilename_normB)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_normB.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        # Compute lower and upper bound of the ROI
        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        # ensure the ROI stays within the image bounds
        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_normB)

        patch_name = os.path.splitext(normBtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(norm_B_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)

    for hematoxylinRtile in hematoxylin_R_name:
        # print("start processing:",normtile)
        tilename_hematoxylinR = os.path.join(hematoxylin_R_path, hematoxylinRtile)
        # print(tilename_norm)
        im_hematoxylinR = sitk.ReadImage(tilename_hematoxylinR)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_hematoxylinR.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        # Compute lower and upper bound of the ROI
        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        # ensure the ROI stays within the image bounds
        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_hematoxylinR)

        patch_name = os.path.splitext(hematoxylinRtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(hematoxylin_R_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)

    for hematoxylinGtile in hematoxylin_G_name:
        # print("start processing:",normtile)
        tilename_hematoxylinG = os.path.join(hematoxylin_G_path, hematoxylinGtile)
        # print(tilename_norm)
        im_hematoxylinG = sitk.ReadImage(tilename_hematoxylinG)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_hematoxylinG.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_hematoxylinG)

        patch_name = os.path.splitext(hematoxylinGtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(hematoxylin_G_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)

    for hematoxylinBtile in hematoxylin_B_name:
        # print("start processing:",normtile)
        tilename_hematoxylinB = os.path.join(hematoxylin_B_path, hematoxylinBtile)
        # print(tilename_norm)
        im_hematoxylinB = sitk.ReadImage(tilename_hematoxylinB)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_hematoxylinB.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        # Compute lower and upper bound of the ROI
        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        # ensure the ROI stays within the image bounds
        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_hematoxylinB)

        patch_name = os.path.splitext(hematoxylinBtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(hematoxylin_B_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)

    for eosinRtile in eosin_R_name:
        # print("start processing:",normtile)
        tilename_eosinR = os.path.join(eosin_R_path, eosinRtile)
        # print(tilename_norm)
        im_eosinR = sitk.ReadImage(tilename_eosinR)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_eosinR.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        # Compute lower and upper bound of the ROI
        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        # ensure the ROI stays within the image bounds
        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_eosinR)

        patch_name = os.path.splitext(eosinRtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(eosin_R_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)

    for eosinGtile in eosin_G_name:
        # print("start processing:",normtile)
        tilename_eosinG = os.path.join(eosin_G_path, eosinGtile)
        # print(tilename_norm)
        im_eosinG = sitk.ReadImage(tilename_eosinG)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_eosinG.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_eosinG)

        patch_name = os.path.splitext(eosinGtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(eosin_G_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)

    for eosinBtile in eosin_B_name:
        # print("start processing:",normtile)
        tilename_eosinB = os.path.join(eosin_B_path, eosinBtile)
        # print(tilename_norm)
        im_eosinB = sitk.ReadImage(tilename_eosinB)
        point = (500, 500)  # fill in the index of your point here
        roi_size = (990, 990)  # x, y, z; uneven to ensure the point is really the center of your ROI
        im_size = im_eosinB.GetSize()[
                  ::-1]  # size in z, y, x, needed because the arrays obtained from the image are oriented in z, y, x
        ma_arr = np.zeros(im_size, dtype='uint8')

        # Compute lower and upper bound of the ROI
        L_x = point[0] - int((roi_size[0] - 1) / 2)
        L_y = point[1] - int((roi_size[1] - 1) / 2)

        U_x = point[0] + int((roi_size[0] - 1) / 2)
        U_y = point[1] + int((roi_size[1] - 1) / 2)

        # ensure the ROI stays within the image bounds
        L_x = max(0, L_x)
        L_y = max(0, L_y)

        U_x = min(im_size[0] - 1, U_x)
        U_y = min(im_size[0] - 1, U_y)
        # 'segment' the mask
        ma_arr[L_y:U_y + 1, L_x:U_x + 1] = 1

        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(im_eosinB)

        patch_name = os.path.splitext(eosinBtile)
        mask_filename = patch_name[0] + '_mask' + '.nrrd'
        MASK_PATH = os.path.join(eosin_B_path, mask_filename)

        sitk.WriteImage(ma, MASK_PATH, True)
    print("finished")
    print("------------------------------------------------")
print("-------------------------Mission Complete------------------------")