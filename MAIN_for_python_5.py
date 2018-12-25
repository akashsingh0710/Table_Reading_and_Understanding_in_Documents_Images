# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:23:24 2018

@author: Akash Singh
"""

import PDF_2_JPG
import numpy as np
from PIL import Image as im
from scipy import ndimage
from scipy.ndimage import interpolation as inter
#import hickle as hkl
import cv2
import glob
import collections
import os
import shutil
#from skew_correction import skew_correction_func
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def skew_correction(img_orig):
    img = im.fromarray(img_orig)
    a,b = img.size

    if b>=a:
        size =  720 , int((b/a)*720)  
    else:
        size =   int((a/b)*720)   ,720  
    
    img = img.resize(size, im.ANTIALIAS)

    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

    delta = 0.1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    if best_score>300000:
        best_angle = angles[scores.index(best_score)]
    else: 
        best_angle=0 
    

    if abs(best_angle)<0.2:
        best_angle = 0
#print('Best angle: {}'.format(best_angle))

# correct skew
    img_skew_corrected = ndimage.rotate(img, best_angle)
    if len(img_skew_corrected.shape) == 3:
        img_skew_corrected=cv2.cvtColor(img_skew_corrected,cv2.COLOR_BGR2GRAY)
#hkl.dump([img_skew_corrected], "img_skew_corrected")
#cv2.imwrite('skew_corrected.jpg', img_skew_corrected)
    return img_skew_corrected

def text_inversion(img,  vert_mode ,img_skew_corrected):
    
    img_skew_corrected_reverse=cv2.bitwise_not(img_skew_corrected)
    original_img_reverse = cv2.bitwise_not(img)
    #blur = cv2.GaussianBlur(img, (1,1) , 0)
    ret,thresh_img = cv2.threshold(np.uint8(img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    inv_img = cv2.bitwise_not(thresh_img)
    #ret,img_line_removed = cv2.threshold(img_line_removed,170,255,cv2.THRESH_BINARY)
    # im2, contours, hierarchy = cv2.findContours(img_line_removed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    a,b = thresh_img.shape
    
    ret_outer_bound, markers_outer_bound,stats_outer_bound,centroids_outer_bound= cv2.connectedComponentsWithStats(inv_img)
    
    
    area_outer_bound = np.zeros((stats_outer_bound.shape[0] ,1))    
     
    for i in range (0,stats_outer_bound.shape[0]):
            area_outer_bound[i] = (stats_outer_bound[i,3] * stats_outer_bound[i,2])
            
            if area_outer_bound[i] < ((vert_mode)*(vert_mode*6)) or area_outer_bound[i]>a*b/1.1  or stats_outer_bound[i,2]<vert_mode*3:
                area_outer_bound[i] = 0
                
    for k in range (0,area_outer_bound.shape[0]):
            if area_outer_bound[k]!=0:
                left = int(stats_outer_bound[k,0])
                top = int(stats_outer_bound[k,1])
                
                for i in range (top, stats_outer_bound[k,3]+top):
                    for j in range (left, stats_outer_bound[k,2]+left): 
                        img[i,j] = 255
                
                
                for i in range (top, stats_outer_bound[k,3]+top):
                    for j in range (left, stats_outer_bound[k,2]+left): 
                        img[i,j] = img_skew_corrected_reverse[i,j]
                        
    new_img = img
    
    return new_img

def line_removal(img_skew_corrected):
    img = img_skew_corrected.copy()
    if len(img.shape) == 3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    ret,thresh = cv2.threshold(img,70,255,cv2.THRESH_BINARY)

    th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,10)
    #cv2.imwrite("th2.jpg" , th2)

    #blur2 = cv2.GaussianBlur(img, (3,3) , 0)
    #ret42,th42 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    blur = cv2.bilateralFilter(th2,9,75,75)
    #blur = cv2.GaussianBlur(th2, (1,1) , 0)
    ret4,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imwrite("th4.jpg" , th4)
    #th4 = cv2.bilateralFilter(th4,9,75,75)      #bahiashdakjdsakjaskj
    
    vertical_line = cv2.bitwise_not(th4.copy())    
    cv2.imwrite("hello.jpg" , vertical_line)
    ret_bound, markers_img,stats,centroids = cv2.connectedComponentsWithStats(vertical_line)
    
    freq = collections.Counter(stats[:,3])
    for i  in range (0,6):
        freq[i]=0
    
    counter_list = sorted(freq.most_common())
    print(counter_list)
    counter_index = 0
    
    for i in range (len(counter_list)-1 , 6 ,-1):    
        vert_mode , value = counter_list[i]
        if value>30 and vert_mode<18:
            counter_index = i
            break    
    
    if counter_index > 0:
        vert_mode , value = counter_list[counter_index]
    else:
        vert_mode=1
    print(vert_mode)    
    for i in range (0,1):
        kernel = np.ones((vert_mode+2,1),np.uint8)
        vertical_line = cv2.erode(vertical_line , kernel, iterations=1)
        vertical_line = cv2.dilate(vertical_line , kernel, iterations=1)
        kernel = np.ones((vert_mode,1),np.uint8)
        vertical_line = cv2.dilate(vertical_line , kernel, iterations=1)
        
    vertical_line = cv2.blur(vertical_line , (3,3))

    horizontal_line = cv2.bitwise_not(th4.copy())
    
    for i in range (0,1):
        kernel = np.ones((1,2*vert_mode),np.uint8)
        horizontal_line = cv2.erode(horizontal_line , kernel, iterations=1)
        horizontal_line = cv2.dilate(horizontal_line , kernel, iterations=1)
        kernel = np.ones((1,vert_mode),np.uint8)
        horizontal_line = cv2.dilate(horizontal_line , kernel, iterations=1)
        
    horizontal_line = cv2.blur(horizontal_line , (3,3))
    
    ret,horizontal_line = cv2.threshold(horizontal_line,70,255,cv2.THRESH_BINARY)
    ret,vertical_line = cv2.threshold(vertical_line,70,255,cv2.THRESH_BINARY)
    
    
    #final_img1 = th42+horizontal_line+vertical_line
    img_line_only = horizontal_line.astype(np.float64) + vertical_line.astype(np.float64)
    for i in range (0,img_line_only.shape[0]):
        for j in range (0,img_line_only.shape[1]):    
            if img_line_only[i,j]> 255:
                img_line_only[i,j]=255
    cv2.imwrite("lpplpl.jpg" , img_line_only)            
    ret, markers,stats,centroids =cv2.connectedComponentsWithStats(np.uint8(img_line_only))
    cv2.imwrite("wwe.jpg" , markers) 
    area = np.zeros((stats.shape[0] ,1)) 
    for i in range (0,stats.shape[0]):
        area[i] = (stats[i,3] * stats[i,2])
        print(area[i])
        if area[i] < (vert_mode*2)*(vert_mode*2):
            area[i] = 0
    
    
    for k in range (0,area.shape[0]):
        if area[k]==0:
            left = int(stats[k,0])
            top = int(stats[k,1])    
            for i in range (0, stats[k,3]):
                for j in range (0, stats[k,2]):
                    vertical_line[int(top+i) , int(left+j)] = 0
                    horizontal_line[int(top+i) , int(left+j)] = 0
    
    cv2.imwrite("peepoo.jpg" , vertical_line) 
    cv2.imwrite("poopoo.jpg" , horizontal_line)
    
    final_img2 = img.astype(np.float64) + horizontal_line.astype(np.float64) + vertical_line.astype(np.float64)
    for i in range (0,final_img2.shape[0]):
        for j in range (0,final_img2.shape[1]):    
            if final_img2[i,j]> 255:
                final_img2[i,j]=255

    #ret5,th5 = cv2.threshold(np.uint8(final_img2),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)            

    #cv2.imwrite("final_img2.jpg" , final_img2)
    #cv2.imwrite("final_img1.jpg" , final_img1)

    
    img_line_removed = final_img2
    
    return img_line_only, img_line_removed , vertical_line , horizontal_line , vert_mode

def table_seperator(vertical, horizontal, img_line_removed , img_line_only , img_skew_corrected , image_no , folder):
    #img_line_only = cv2.imread("img_line_only.jpg" , 0)
   # img_line_removed=cv2.imread("img_line_removed.jpg" ,0)
    
    #img_skew_corrected = cv2.imread("skew_corrected_0.jpg" , 0)


    lines_img = img_line_only.copy()
#lines_img = cv2.imread("zzx.jpg" , 0)
#lines_bwimg =cv2.adaptiveThreshold(lines_img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,225,-2)
    ret, thresh = cv2.threshold(np.uint8(lines_img),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    a,b = lines_img.shape


    ret_outer_bound, markers_outer_bound,stats_outer_bound,centroids_outer_bound = cv2.connectedComponentsWithStats(~thresh)      
        
    area_outer_bound = np.zeros((stats_outer_bound.shape[0] ,1))      

    for i in range (0,stats_outer_bound.shape[0]):
        area_outer_bound[i] = (stats_outer_bound[i,3] * stats_outer_bound[i,2])
        
        if area_outer_bound[i] < (a*b)/32 or area_outer_bound[i]>=a*b/1.2 or stats_outer_bound[i,2]<80:
            area_outer_bound[i] = 0
     
        #table_no = np.count_nonzero(area_outer_bound)
        temp=1
        

    for k in range (0,area_outer_bound.shape[0]):
        
        if area_outer_bound[k]!=0:
            left = int(stats_outer_bound[k,0])
            top = int(stats_outer_bound[k,1])
            detected_tables= np.zeros(( stats_outer_bound[k,3]-8,stats_outer_bound[k,2]-8) )
            detected_lines_horizontal = np.zeros(( stats_outer_bound[k,3]-8,stats_outer_bound[k,2]-8) )
            detected_lines_vertical = np.zeros(( stats_outer_bound[k,3]-8,stats_outer_bound[k,2]-8) )
            
            for i in range (0, stats_outer_bound[k,3]-8):
                for j in range (0, stats_outer_bound[k,2]-8):
                    #if i!=-1 or j!=-1 or i!=0 or j!=0 or i!=stats_outer_bound[k,3]-3 or i!=stats_outer_bound[k,3]-2 or i!=stats_outer_bound[k,3]-1 or j!=stats_outer_bound[k,2]-3 or j!=stats_outer_bound[k,2]-2 or j!=stats_outer_bound[k,2]-1:
                    detected_lines_vertical[ int(i) , int(j)] = vertical[int(top+i+4) , int(left+j+4)]
                    detected_lines_horizontal[ int(i) , int(j)] = horizontal[int(top+i+4) , int(left+j+4)]
                        #detected_tables[i , j] = img_skew_corrected[int(top+i) , int(left+j)]
                    detected_tables[i , j] = img_line_removed[int(top+i+4) , int(left+j+4)]
                    #img_skew_corrected[int(top+i) , int(left+j)] = 255
        #new_size = (detected_tables.shape[0]+10, detected_tables.shape[1]+10)
        #new_im = IMAGE.new("RGB" , new_size)
        #new_im.paste(old_im, ((new_size[0]-old_size[0])/2,(new_size[1]-old_size[1])/2))
        
        #color0 = [0]
        #color1 = [255]
        #top, bottom, left, right = [3]*4
        #detected_tables_lines = cv2.copyMakeBorder(detected_tables_lines, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color0)
        #detected_tables = cv2.copyMakeBorder(detected_tables, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color1) #1
            #pair =(3**image_no)*(2**temp)
            #filename_tables_lines = "detected_table_lines_%d.jpg"%pair
           
            
            
            #vertical_lines=detected_lines_vertical.copy()
            #horizontal_lines = detected_lines_horizontal.copy()
            
            ht,wd=detected_lines_horizontal.shape
            
            ret, markers,statsh,centroids =cv2.connectedComponentsWithStats(np.uint8(detected_lines_horizontal))
            max_statsh = wd/2
            area = np.zeros((statsh.shape[0] ,1)) 
            for i in range (0,statsh.shape[0]):
                if statsh[i,2] < max_statsh:
                    area[i] = 1
                        
            for p in range (0,statsh.shape[0]):
                if area[p]==1:
                    lefth = int(statsh[p,0])
                    toph = int(statsh[p,1])    
                    for i in range (0, statsh[p,3]):
                        for j in range (0, statsh[p,2]):
                            detected_lines_horizontal[int(toph+i) , int(lefth+j)] = 0
                            
            
             
            ret, markers,statsv,centroids =cv2.connectedComponentsWithStats(np.uint8(detected_lines_vertical))
            max_statsv = ht/3
            area = np.zeros((statsv.shape[0] ,1))
            for i in range (0,statsv.shape[0]):
                if statsv[i,3] < max_statsv:
                    area[i] = 1
            
            
            for q in range (0,statsv.shape[0]):
                if area[q]==1:
                    leftv = int(statsv[q,0])
                    topv = int(statsv[q,1])    
                    for i in range (0, statsv[q,3]):
                        for j in range (0, statsv[q,2]):
                            detected_lines_vertical[int(topv+i) , int(leftv+j)] = 0    
            
            
            
            
            
            pocessing_files_2_direc = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables\{}\{}".format(folder+1,image_no , temp)        
            if not (os.path.exists(pocessing_files_2_direc)):
                os.makedirs(pocessing_files_2_direc) 
            #row_print, worksheet = combined_3.main_table_1(img_line_removed, row_print , vertical, horizontal, worksheet)
            filename_tables_1 = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables\{}\{}\image.jpg".format(folder+1 , image_no , temp)
            filename_tables_vertical = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables\{}\{}\vertical.jpg".format(folder+1 , image_no , temp)
            filename_tables_horizontal = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables\{}\{}\horizontal.jpg".format(folder+1 , image_no , temp)
            #cv2.imwrite(filename_tables_lines , detected_tables_lines)
            cv2.imwrite(filename_tables_1 , detected_tables)
            cv2.imwrite(filename_tables_vertical , detected_lines_vertical)
            cv2.imwrite(filename_tables_horizontal , detected_lines_horizontal)
            temp=temp+1
            cv2.imwrite("klv.jpg",vertical)
            cv2.imwrite("klh.jpg",horizontal)
            
            
            left = int(stats_outer_bound[k,0])
            top = int(stats_outer_bound[k,1])
            for i in range (0, stats_outer_bound[k,3]):
                for j in range (0, stats_outer_bound[k,2]):
                    img_line_removed[int(top+i) , int(left+j)] = 255
                    vertical[int(top+i) , int(left+j)]=0
                    horizontal[int(top+i) , int(left+j)]=0
                
        
        
        #row_print , worksheet = combined_3.main_table(img_line_removed, row_print , vertical , horizontal, worksheet)
        processing_files_2_direc = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\{}".format(folder+1,image_no)        
        if not(os.path.exists(processing_files_2_direc)):
            os.makedirs(processing_files_2_direc)
        
        #os.makedirs("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES\11")    
        filename_part_image = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\{}\image.jpg".format(folder+1 , image_no)
        filename_part_image_horizontal = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\{}\horizontal.jpg".format(folder+1 , image_no)
        filename_part_image_vertical = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\{}\vertical.jpg".format(folder+1 , image_no)
        
        
        cv2.imwrite(filename_part_image , img_line_removed)
        cv2.imwrite(filename_part_image_horizontal , horizontal)
        cv2.imwrite(filename_part_image_vertical , vertical)
    return 0







processed_2_direc = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2"
total_input_files = len(os.listdir(processed_2_direc))



for j in range (0,total_input_files):
    images = [cv2.imread(file) for file in glob.glob("C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 1\{}\*.jpg" .format(j+1)) ]
        
    
#    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range (0 , len(images)):
        image_no = i+1
        img_orig = images[i]    
        
        #img_skew_corrected = img_orig
        img_skew_corrected = skew_correction(img_orig)
        img_line_only, img_line_removed,vertical_line , horizontal_line , vert_mode = line_removal(img_skew_corrected)
        cv2.imwrite("img_line_removed.jpg",img_line_removed)
        cv2.imwrite("vertical_line.jpg",vertical_line)
        cv2.imwrite("horizontal_line.jpg",horizontal_line)
        img_line_removed = text_inversion(np.uint8(img_line_removed) , vert_mode , img_skew_corrected)
        #cv2.imwrite("img_line_removed_after_inversion.jpg",np.uint8(img_line_removed))
        table_seperator(vertical_line,horizontal_line, img_line_removed , img_line_only , img_skew_corrected , image_no , j )
    
  
    
    
import combined_5     
    