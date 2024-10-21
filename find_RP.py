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

import re #regular expression
import sys #System
import cv2 #OpenCv
import PIL #Pillow
import glob #Global variable
import math
import datetime
import numpy as np # Numpy
import pandas as pd #Pandas
import multiprocessing
import matplotlib.pyplot as plt
import skimage.io

import openslide
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ImageDraw, ImageFont

from histolab.slide import Slide
from histolab.tiler import RandomTiler, GridTiler, ScoreTiler
from histolab.scorer import NucleiScorer
from histolab.filters.image_filters import ApplyMaskImage, GreenPenFilter, Invert, OtsuThreshold, RgbToGrayscale
from histolab.filters.morphological_filters import RemoveSmallHoles, RemoveSmallObjects
from histolab.masks import TissueMask



# ------------------------------score tile------------------------------
path = "D:/pathological/pathology_image/example/"   #/lusc_dia/
slidename = os.listdir(path)

for wsi in slidename:
    LUSC_path = os.path.join(path, wsi)
    print("start processing:", LUSC_path)
    tilefile = os.path.splitext(wsi)
    print(tilefile)
    new = os.getcwd() + '/' + tilefile[0]
    # print(new)
    if not os.path.exists(new):
        os.makedirs(new)
    pro_path = new
    print("processed files will be saved to:", pro_path)
    LUSC_slide = Slide(LUSC_path, processed_path=pro_path)
    print(f"selecting patches for:{LUSC_slide.name}")  # 幻灯片名称
    LUSC_slide.thumbnail

    scored_tiles_extractor = ScoreTiler(scorer=NucleiScorer(),
                                        tile_size=(1000, 1000),
                                        n_tiles=5,
                                        level=0,
                                        check_tissue=True,
                                        tissue_percent=80,
                                        pixel_overlap=0,
                                        prefix="scored/",
                                        suffix=".png")

    scored_tiles_extractor.locate_tiles(slide=LUSC_slide, outline="red")
    summary_filename = "tile_score.csv"
    SUMMARY_PATH = os.path.join(LUSC_slide.processed_path, summary_filename)
    scored_tiles_extractor.extract(LUSC_slide, report_path=SUMMARY_PATH)
    print("processed successfully", SUMMARY_PATH)
    print("------------------------------------------------------------------------")


"""
# ------------------------------grid tile------------------------------s
path = "D:/pathological/pathology_image/lusc_left/"
slidename = os.listdir(path)

for wsi in slidename:
    LUSC_path = os.path.join(path, wsi)
    print("start processing:", LUSC_path)
    tilefile = os.path.splitext(wsi)
    print(tilefile)
    new = os.getcwd() + '/' + tilefile[0]
    # print(new)
    if not os.path.exists(new):
        os.makedirs(new)
    pro_path = new
    print("processed files will be saved to:", pro_path)
    LUSC_slide = Slide(LUSC_path, processed_path=pro_path)
    print(f"selecting patches for:{LUSC_slide.name}")  # 幻灯片名称
    LUSC_slide.thumbnail

    grid_tiles_extractor = GridTiler(tile_size=(1000, 1000),
                                     level=0,
                                     check_tissue=True,
                                     tissue_percent=60,
                                     pixel_overlap=0,  # default
                                     prefix="grid/",
                                     suffix=".png")

    grid_tiles_extractor.locate_tiles(slide=LUSC_slide, outline="red")

    grid_tiles_extractor.extract(LUSC_slide)

    print("processed successfully", tilefile)
    print("------------------------------------------------------------------------")
"""
