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


input_direc = "C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES"
total_input_files = len(os.listdir(input_direc))

total_effective_files = total_input_files - total_word_file




direc_1= "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1"
if os.path.exists(direc_1):
    shutil.rmtree(direc_1)

os.makedirs(direc_1)

for i in range(0, total_effective_files):
   pocessing_files_1_direc = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}".format(i+1)        
   os.makedirs(pocessing_files_1_direc) 

    
   
   
direc_2 = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2"
if os.path.exists(direc_2):
    shutil.rmtree(direc_2)

os.makedirs(direc_2)    

for i in range(0, total_effective_files):
   pocessing_files_2_direc = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}".format(i+1)        
   os.makedirs(pocessing_files_2_direc) 
   direc_3= "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables".format(i+1)
   os.makedirs(direc_3)

direc_3= "C:\Axis AI Challenge @ Akash_Abhishek\OUTPUT FILES"
if os.path.exists(direc_3):
    shutil.rmtree(direc_3) 

os.makedirs(direc_3) 

for i in range(0, total_effective_files):
   outpu_files_direc = "C:\Axis AI Challenge @ Akash_Abhishek\OUTPUT FILES\{}".format(i+1)        
   os.makedirs(outpu_files_direc) 






pdf = [convert_from_path(in_file, 500) for in_file in glob.glob("C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.pdf")]
total_no_pdf = len(pdf)        

for i in range(0, len(pdf)):
    pages = pdf[i]
    
    for j in range(0, len(pages)):
        page = pages[j]   
        page.save("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\PDF_{} page_{}.jpg" .format(i+1,i+1,j+1))



        
jpgs = [cv2.imread(file) for file in glob.glob("C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.jpg") ]        
total_no_jpgs = len(jpgs)        

for i in range(0, len(jpgs)):
    jpg_page=jpgs[i] 
    cv2.imwrite("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\jpg_{}.jpg" .format(i+1+total_no_pdf,i+1)   , jpg_page)
    
    
    
tiffs = [cv2.imread(file) for file in glob.glob("C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.tiff") ]        
total_no_tiffs = len(tiffs)        

for i in range(0, len(tiffs)):
    tiff_page=tiffs[i]
    cv2.imwrite("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\tiff_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs, i+1), tiff_page)    
    #tiff_page.save("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\tiff_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs, i+1))


pngs = [cv2.imread(file) for file in glob.glob("C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.png") ]        
total_no_pngs = len(pngs)        

for i in range(0, len(pngs)):
    png_page=pngs[i] 
    cv2.imwrite("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\png_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs+total_no_tiffs,i+1)   , png_page)    


jpegs = [cv2.imread(file) for file in glob.glob("C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.jpeg") ]        
total_no_jpegs = len(jpegs)        

for i in range(0, len(jpegs)):
    jpeg_page=jpegs[i] 
    cv2.imwrite("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\jpeg_{}.jpg" .format(i+1+total_no_pdf+total_no_jpgs+total_no_tiffs+total_no_pngs,i+1))

