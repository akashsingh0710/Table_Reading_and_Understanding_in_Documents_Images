# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 21:54:10 2018

@author: Akash Singh
"""
import DOC_2_PDF
from DOC_2_PDF import total_word_file
import shutil
import os,  os.path
import glob
from pdf2image import convert_from_path
import cv2
import numpy as np 


input_direc = r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES"
total_input_files = len(os.listdir(input_direc))

total_effective_files=total_input_files - total_word_file

total_input_files = len(os.listdir(input_direc))



    
#name_lists_new = set(name_lists)    
            

direc_1= r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1"
if os.path.exists(direc_1):
    shutil.rmtree(direc_1)

os.makedirs(direc_1)

for i in range(0, total_effective_files):
   pocessing_files_1_direc = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}".format(i+1)        
   os.makedirs(pocessing_files_1_direc) 

    
   
   
direc_2 = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2"
if os.path.exists(direc_2):
    shutil.rmtree(direc_2)

os.makedirs(direc_2)    

for i in range(0, total_effective_files):
   pocessing_files_2_direc = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}".format(i+1)        
   os.makedirs(pocessing_files_2_direc) 
   direc_3= r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables".format(i+1)
   os.makedirs(direc_3)

direc_3= r"C:\Axis AI Challenge @ Akash_Abhishek\OUTPUT FILES"
if os.path.exists(direc_3):
    shutil.rmtree(direc_3) 

os.makedirs(direc_3) 

#for i in range(0, total_effective_files):
#   outpu_files_direc = r"C:\Axis AI Challenge @ Akash_Abhishek\OUTPUT FILES\{}".format(i+1)        
#   os.makedirs(outpu_files_direc) 




file_name= []

results_pdf = [each for each in os.listdir(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES") if each.endswith('.pdf')]
results_jpg = [each for each in os.listdir(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES") if each.endswith('.jpg')]
results_png = [each for each in os.listdir(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES") if each.endswith('.png')]
results_jpeg = [each for each in os.listdir(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES") if each.endswith('.jpeg')]
results_tiff = [each for each in os.listdir(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES") if each.endswith('.tiff')]
results_tif = [each for each in os.listdir(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES") if each.endswith('.tif')]






pdf = [convert_from_path(in_file, 500) for in_file in glob.glob(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.pdf")]
total_no_pdf = len(pdf)        

for i in range(0, len(pdf)):
    pages = pdf[i]
    name = results_pdf[i]
    name_list = os.path.splitext(name)[0]
    file_name.append(name_list)
    
    for j in range(0, len(pages)):
        page = pages[j]   
        page.save(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\PDF_{} page_{}.jpg" .format(i+1,i+1,j+1))



#input_files_name=os.listdir(input_direc)
#
#for i in range (0,total_input_files):
#    name = input_files_name[i]
#    name_list = os.path.splitext(name)[0]
#    name_lists.append(name_list)

        
jpgs = [cv2.imread(file) for file in glob.glob(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.jpg") ]        
total_no_jpgs = len(jpgs)        

for i in range(0, len(jpgs)):
    jpg_page=jpgs[i] 
    cv2.imwrite(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\jpg_{}.jpg" .format(i+1+total_no_pdf,i+1)   , jpg_page)
    name = results_jpg[i]
    name_list = os.path.splitext(name)[0]
    file_name.append(name_list)

pngs = [cv2.imread(file) for file in glob.glob(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.png") ]        
total_no_pngs = len(pngs)        

for i in range(0, len(pngs)):
    png_page=pngs[i] 
    cv2.imwrite(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\png_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs,i+1)   , png_page)
    name = results_png[i]
    name_list = os.path.splitext(name)[0]
    file_name.append(name_list)

jpegs = [cv2.imread(file) for file in glob.glob(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.jpeg") ]        
total_no_jpegs = len(jpegs)        

for i in range(0, len(jpegs)):
    jpeg_page=jpegs[i] 
    cv2.imwrite(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\jpeg_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs+total_no_pngs,i+1))
    name = results_jpeg[i]
    name_list = os.path.splitext(name)[0]
    file_name.append(name_list)

    
tiffs = [cv2.imread(file) for file in glob.glob(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.tiff") ]        
total_no_tiffs = len(tiffs)        

for i in range(0, len(tiffs)):
    tiff_page=tiffs[i]
    cv2.imwrite(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\tiff_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs+total_no_pngs+total_no_jpegs, i+1), tiff_page)    
    name = results_tiff[i]
    name_list = os.path.splitext(name)[0]
    file_name.append(name_list)
    #tiff_page.save("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\tiff_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs, i+1))


tifs = [cv2.imread(file) for file in glob.glob(r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.tif") ]        
total_no_tiffs = len(tifs)        

for i in range(0, len(tifs)):
    tif_page=tifs[i]
    cv2.imwrite(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\tiff_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs+total_no_pngs+total_no_jpegs+total_no_tiffs, i+1), tif_page)    
    name = results_tif[i]
    name_list = os.path.splitext(name)[0]
    file_name.append(name_list)
    #tiff_page.save("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\tiff_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs, i+1))

    



print(file_name)

