import os
import glob
import cv2
import numpy as np
from collections import Counter
import collections
import matplotlib.pyplot as plt
import copy
import pytesseract
from PIL import Image
import xlsxwriter

def vertical_mode(th4):
    th4 = np.uint8(th4)
    cv2.imwrite("th4.jpg" , th4)
    vertical_line = cv2.bitwise_not(th4.copy())    
    ret_bound, markers_img,stats,centroids = cv2.connectedComponentsWithStats(vertical_line)
    
    freq = Counter(stats[:,3])
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
        vert_mode=12
    
    print(vert_mode)    
    return vert_mode        



def comp_line(labels,i,j):
    if i !=0 and j != 0: 
        if top(labels,i) <= bottom(labels,j) :
            if bottom(labels,i) >= top(labels,j):
                line = 1
            else:
                line = 0
        else:
            line = 0
    else:
        line = 0
        
    return line

def block_line(labels,i,j):
#    if i !=0 and j != 0: 
    if block_top(labels,i) <= block_bottom(labels,j) :
        if block_bottom(labels,i) >= block_top(labels,j):
            line = 1
        else:
            line = 0
    else:
            line = 0
#    else:
 #       line = 0
        
    return line

def left(stats,i):
    return stats[i,0]

def right(stats,i):
    return stats[i,0] + stats[i,2] - 1

def top(stats,i):
    return stats[i,1]

def bottom(stats,i):
    return stats[i,1] + stats[i,3] - 1

def block_left(stats,i):
    return stats[i,0]

def block_right(stats,i):
    return stats[i,1]

def block_top(stats,i):
    return stats[i,2]

def block_bottom(stats,i):
    return stats[i,3]

def bounds(stats,i):
    t = top(stats,i)
    b = bottom(stats,i)
    r = right(stats,i)
    l = left(stats,i)
    
    return l,r,t,b

def consc_comp(labels,i):
    line_c = []
    for j in range(0,labels.shape[0]):
        if comp_line(labels,i,j) == 1 :
            if left(labels,j) >= right(labels,i) and i != j :
                line_c.append(np.insert(labels[j][:],0,j))
        
    if not line_c:
        return -1,-1
    line_c = np.array(line_c)
    diff = line_c[:,1]-right(labels,i)
    #print line_c
    arg = np.argmin(diff)
    dist = np.min(diff)
    arg_c = line_c[arg,0]
    return arg_c, dist

def consc_block(labels,i):
    line_c = []
    for j in range(0,labels.shape[0]):
        if block_line(labels,i,j) == 1 :
            if block_left(labels,j) >= block_right(labels,i) and i != j:
                line_c.append(np.insert(labels[j][:],0,j))
        
    if not line_c:
        return -1,-1
    line_c = np.array(line_c)
    diff = line_c[:,1]-block_right(labels,i)
    #print line_c
    arg = np.argmin(diff)
    dist = np.min(diff)
    arg_c = line_c[arg,0]
    return arg_c, dist

tolerance = 7

def vertical_align(stats,i,j):
    area_l = stats[j,0]
    area_r = stats[j,1]
    test_l = stats[i,0]
    test_r = stats[i,1]
    area_span = area_r - area_l
    test_span = test_r - test_l
    align = 0
    if test_span <= area_span:
        if test_l < area_l :
            align = (abs(area_l - test_l) <= tolerance)
        elif test_r > area_r:
            align = (abs(area_r - test_r) <= tolerance)
        elif test_l == area_l or test_r == area_r:
            align = 1
        elif test_l > area_l and test_r < area_r:
            align = 1
    else:
        if test_l > area_l :
            align = (abs(area_l - test_l) <= tolerance)
        elif test_r < area_r:
            align = (abs(area_r - test_r) <= tolerance)
        elif test_l == area_l or test_r == area_r:
            align = 1
        elif test_l < area_l and test_r > area_r:
            align = 1
    
    return align

def sort_lineb(label,bounds,i):
    blocks = np.argwhere(label == i)[:,0]
    b_bounds = bounds[blocks]
    b_bounds_l = b_bounds[:,0]
    sort = np.argsort(b_bounds_l)
    sort_b = []
    for i in range(0,blocks.shape[0]):
        sort_b.append(blocks[sort[i]])
    return sort_b

def line_align(label,bounds,i,j):
    blocks_i = sort_lineb(label,bounds,i)
    blocks_j = sort_lineb(label,bounds,j)
    len_i = len(blocks_i)
    len_j = len(blocks_j)
    counter = 0
    for i in range(0,len_j):
        block_j = blocks_j[i]
        sum_c = 0
        for j in range(0,len_i):
            block_i = blocks_i[j]
            sum_c += vertical_align(bounds,block_j,block_i)
        if sum_c > 1:
            break
        counter += sum_c 
        
    return (counter == len_j)
            
def table_column(table_lines,columns,textb_line_label,block_count, block_bounds):
    lines = table_lines.shape[0]
    tb_blocks = []
    for i in range(0,lines):
        blocks = np.argwhere(textb_line_label == table_lines[i] + 1)[:,0]
        tb_blocks.extend(list(blocks))
    n_blocks = len(tb_blocks)
    block_column = -1*np.ones((n_blocks,2))
    block_column[:,0] = np.array(tb_blocks)
    line_left = []
    line_done = []
    for i in range(0,lines):
        line = table_lines[i]
        if block_count[line] == columns:
            blocks_s = sort_lineb(textb_line_label, block_bounds, line + 1)
            for j in range(0,columns):
                block_j = blocks_s[j]
                index = np.argwhere(block_column[:,0] == block_j)[0][0]
                block_column[index,1] = j
            line_done.append(table_lines[i])
        else:
            line_left.append(table_lines[i])
    line_done = np.array(line_done)
    for i in range(0,len(line_left)):
        dist = np.abs(line_done - line_left[i])
        min_d = min(np.abs(line_done - line_left[i]))
        index = np.argwhere(dist == min_d)[0][0]
        l_index = line_done[index]
        block_l = np.argwhere(textb_line_label == line_left[i] + 1)[:,0]
        block_d = np.argwhere(textb_line_label == l_index + 1)[:,0]
        for j in range(0,block_l.shape[0]):
            bound_l = block_bounds[block_l[j],0]
            bounds_d = block_bounds[block_d,0]
            diff = np.abs(bounds_d - bound_l)
    
            b_index = np.argwhere(diff == min(diff))[0][0]
            b_index = block_d[b_index]
#            print b_index
            #if b_index.shape[0] == 1:
            index_1 = np.argwhere(block_column[:,0] == block_l[j])[0][0]
            index_2 = np.argwhere(block_column[:,0] == b_index)[0][0]
            block_column[index_1,1] = block_column[index_2,1]
    block_column = block_column.astype(int)
    return block_column

def align_table(table_lines,n_columns,block_count,textb_line_label, block_bounds):
    block_column = table_column(table_lines, n_columns ,textb_line_label,block_count, block_bounds)
    textb_table_line = -1 * np.ones_like(textb_line_label)
    table_lines_new = [] 
    for i in range(0,table_lines.shape[0]):
        line = table_lines[i]
        if block_count[line] >= n_columns:
            blocks_l = np.argwhere(textb_line_label == line + 1)[:,0]
            for j in range(0,blocks_l.shape[0]):
                block = blocks_l[j]
                textb_table_line[block] = line + 1
            table_lines_new.append(line)
    table_lines_new = np.array(table_lines_new)
#    print table_lines_new
    table_lines_2 = []
    for i in range(0,table_lines.shape[0]):
        line = table_lines[i]
#        print line
        if block_count[line] < n_columns and (line not in table_lines_new) :
            blocks = np.argwhere(textb_line_label == line + 1)[:,0]
#            print blocks
            col_index = []
#            dist = table_lines_new - line
#            index = np.argwhere(dist == min(dist))[0][0]                
            for j in range(0, blocks.shape[0]):
#                print blocks[j]
                index = np.argwhere(block_column[:,0] == blocks[j])[0][0]
                col_index.append(block_column[index,1])
#            print col_index
            if 0 in col_index :
#                print 'w row header'
                if block_count[line] == 1:
                    dist = table_lines_new - line
                    dist_1 = dist[dist < 0]
                    if dist_1.shape[0] :
                        min_index = max(dist_1)
                        check = 1 * min_index
                        m_line_i = np.argwhere(dist == check)[0][0]
                        m_line = table_lines_new[m_line_i]
                        block = blocks[0]
    ########            more checks here  ####################
                        textb_table_line[block] = m_line + 1       #line changed
                        table_lines_2.append(line)
                    else:
                        for k in range(0,blocks.shape[0]):
                            block = blocks[k]
                            textb_table_line[block] = line + 1
                        table_lines_2.append(line)
                else:
                    for k in range(0,blocks.shape[0]):
                        block = blocks[k]
                        textb_table_line[block] = line + 1
                    table_lines_2.append(line)
            else:
#                print 'no row header'
                dist = table_lines_new - line
                #print(dist)
                dist_1 = dist[dist < 0]
                #print(dist_1)
                if dist_1.shape[0]:
                    min_index = np.max(dist_1)
                    check = 1 * min_index
                    m_line_i = np.argwhere(dist == check)[0][0]
                    m_line = table_lines_new[m_line_i]
#                    print m_line
                    for k in range(0, blocks.shape[0]):
                        block = blocks[k]
                        textb_table_line[block] = m_line + 1        #line changed
                    table_lines_2.append(line)
                else:
                    for k in range(0, blocks.shape[0]):
                        block = blocks[k]
                        textb_table_line[block] = line + 1        #line changed
                    table_lines_2.append(line)
        #print('\n')
    new_label = 0
#    textb_table_line_1 = -1* np.ones_like(textb_table_line)
#    print textb_line_label
#    print textb_table_line
    block_table_line = -1* np.ones_like(block_column)
    block_table_line[:,0] = block_column[:,0]
    
    for i in range(0 , block_table_line.shape[0]):
        block = int(block_table_line[i,0])
#        print block
        line = textb_table_line[block]
        index_1 = np.argwhere(block_table_line[:,0] == block)[0][0]
        block_table_line[index_1,1] = line
    max_index = int(np.max(block_table_line[:,1]))
    
    for i in range(0, max_index + 1):
        counter = i
        check = np.argwhere(block_table_line[:,1] == counter)[:,0]
        if check.shape[0]:
            for j in range(0,check.shape[0]):
                block = check[j]
                block_table_line[block,1] = new_label
            new_label += 1
    table_lines_new = list(table_lines_new)
    t_lines = table_lines_new.extend(table_lines_2)
    block_table_line = block_table_line.astype(int)
    return block_table_line 

l_extra =5
r_extra =5
t_extra =2
b_extra =5

def col_bounds(block_column,block_bounds):
    n_col = np.unique(block_column[:,1]).shape[0]
    col_i = np.unique(block_column[:,1])
    col_i = col_i.astype(int)
    column_bounds = -1 * np.ones((n_col, 4))
    for i in range(0,n_col):
        index = col_i[i]
        blocks = block_column[block_column[:,1] == index][:,0]
        blocks = blocks.astype(int)
        bounds_mat = block_bounds[blocks]
        l = 0
        r = 0
        t = 0
        b = 0
        l = max(np.min(bounds_mat[:,0]) - l_extra,0)
        r = np.max(bounds_mat[:,1]) + r_extra
        t = max(np.min(bounds_mat[:,2]) - t_extra,0)
        b = np.max(bounds_mat[:,3]) + b_extra
        column_bounds[i] = l,r,t,b
    column_bounds = column_bounds.astype(int)
    return column_bounds


line_thresh = 5
def col_bounds_2(column_bounds, image, line_ver):
    a,b = image.shape
    column_bounds_1 = np.zeros((column_bounds.shape[0],4))
    column_bounds_1[:,0] = column_bounds[:,0]
    column_bounds_1[:,2] = column_bounds[:,2]
    column_bounds_1[:,3] = column_bounds[:,3]
    print (column_bounds)
    left = column_bounds[1,0]
    for i in range(0, column_bounds.shape[0] - 1):
        left = column_bounds[i + 1, 0]
        column_bounds_1[i,1] = left
        check = col_ln_chck(column_bounds_1[i], line_ver)
        if check:
            right = column_bounds[i,1]
            dist_l = []
            index_l = []
            for j in range(0, line_ver.shape[0]):
                left_j = line_ver[j,0]
                dist = left_j - right
                print (dist)
                if dist >= -1*line_thresh:
                    dist_l.append(dist)
                    index_l.append(j)
            dist_l = np.array(dist_l)
            index_l = np.array(index_l)
            print(dist_l)
            print(index_l)
            if dist_l.shape[0]:
                index_c = np.argwhere(dist_l == min(dist_l))[0][0]
                index_j = index_l[index_c]
            left_new = line_ver[index_j, 0]
            column_bounds_1[i,1] = left_new    
    column_bounds_1[i + 1, 1] = b        
    print (column_bounds_1)
    for j in range(1,column_bounds.shape[0]):
        right = column_bounds_1[j - 1, 1]
        column_bounds_1[j,0] = right
        
    print (column_bounds_1)
    return column_bounds_1

gap_t = 0

def col_ln_chck(bounds, line_ver):
    l,r,t,b = bounds
    l1 = np.argwhere(line_ver[:,1] > l)[:,0]
    l2 = np.argwhere(line_ver[:,0] < r)[:,0]
    l3 = np.intersect1d(l1,l2)
    l4 = np.argwhere(line_ver[:,3] > t)[:,0]
    l5 = np.intersect1d(l4,l3)
    l6 = np.argwhere(line_ver[:,2] < b)[:,0]
    l7 = np.intersect1d(l5,l6)
    lefts = []
    for i in range(0, l7.shape[0]):
        bounds_i = line_ver[l7[i]]
        if abs(bounds_i[0] - l) > gap_t  :
            lefts.append(bounds_i[0])
    if len(lefts):
        exists = 1
    else:
        exists = 0
        
    return exists

def vertical_align_2(bounds1, bounds2):
    area_l = bounds1[0]
    area_r = bounds1[1]
    test_l = bounds2[0]
    test_r = bounds2[1]
    area_span = area_r - area_l
    test_span = test_r - test_l
    align = 0
    if test_span <= area_span:
        if test_l < area_l :
            align = (abs(area_l - test_l) <= tolerance)
        elif test_r > area_r:
            align = (abs(area_r - test_r) <= tolerance)
        elif test_l == area_l or test_r == area_r:
            align = 1
        elif test_l > area_l and test_r < area_r:
            align = 1
    else:
        if test_l > area_l :
            align = (abs(area_l - test_l) <= tolerance)
        elif test_r < area_r:
            align = (abs(area_r - test_r) <= tolerance)
        elif test_l == area_l or test_r == area_r:
            align = 1
        elif test_l < area_l and test_r > area_r:
            align = 1
    
    return align

def row_cells(block_column, block_table_line , index):
    blocks_i = block_table_line[block_table_line[:,1] == index][:,0]
    col_i = -1 * np.ones(blocks_i.shape[0])
    for j in range(0,blocks_i.shape[0]):
        block = blocks_i[j]
        col_i[j] = block_column[block_column[:,0] == block][:,1][0]
    n_cells = np.unique(block_column[:,1]).shape[0]
    cells_i = np.unique(block_column[:,1])
    cells = []
    for i in range(0, n_cells):
        index = cells_i[i]
        blocks_ci = []
        indices = np.argwhere(col_i == index)[:,0]
        if indices.shape[0]:
            for j in range(0, indices.shape[0]):
                indx = indices[j]
                blck = blocks_i[indx]
                blocks_ci.append(blck)
        cells.append(blocks_ci)
    return cells
    
def table_det(table_lines, textb_line_label, columns, block_count, block_bounds):
    block_column = table_column(table_lines, columns, textb_line_label,block_count, block_bounds)
    block_table_line = align_table(table_lines, columns, block_count, textb_line_label, block_bounds)
    n_columns = np.unique(block_column[:,1]).shape[0]
    n_rows = np.unique(block_table_line[:,1]).shape[0]
    rows = np.unique(block_table_line[:,1])
    rows = rows.astype(int)
#    shape_1 = int(n_columns * n_rows)
#    shape_2 = 5    #depends on the presence of sub headers
#    table_details = -1 *  np.ones((shape_1,shape_2))
#    header_row = block_table_line[block_table_line[:,1] == 0][:,0]
#    for i in range(0,int(n_rows)):
#        blocks_i = block_table_line[block_table_line[:,1] == i][:,0]
#        col_i = -1 * np.ones(blocks_i.shape[0])
#        for j in range(0,blocks_i.shape[0]):
#            block = blocks_i[j]
#            col_i[j] = block_column[block_column[:,0] == block][:,1][0]
#        n_cells = np.unique(col_i).shape[0]
#        cell_i = np.unique(col_i)
#        for j in range(0,int(n_cells)):
#            cell = cell_i[j]
#            cells = np.argwhere(col_i == cell)[:,0]
##            table_det[]
##            for k in range(0,cells.shape[0]):
    n_headers = 1
    table_detail =[]
    headers = []
    for i in range(0, n_headers):
        row = rows[i]
        h_cells = row_cells(block_column, block_table_line, row)
        headers.append(h_cells)
        
    for i in range(n_headers, int(n_rows)):
        row = rows[i]
        cells = row_cells(block_column, block_table_line, row)
        n_cells = len(cells)
        for j in range(1, n_cells):
            row_det = []
            cell = cells[j]
            row_det.append(cell)
            row_det.append([])
            for k in range(n_headers, 0 , -1):      #column headers
                h_index = k - 1
                cell_index = j
                h_cell = headers[h_index][cell_index]
                row_det.append(h_cell)
            row_det.append([])
            row_det.append(cells[0])       #row header
########## append row sub headers here##############
            for k in range(n_headers, 0 , -1):      #column headers of hrow header
                h_index = k - 1
                cell_index = 0
                h_cell = headers[h_index][cell_index]
                row_det.append(h_cell)
            table_detail.append(row_det)
    return table_detail

def col_image(column_bounds, image):
    n_imgs = column_bounds.shape[0]
    images = []
    for i in range(0, n_imgs):
        l,r,t,b = column_bounds[i]
        shape_1 = r - l + 1
        shape_2 = b - t + 1
        img_i = np.zeros((shape_1,shape_2) , dtype = np.uint8)
        img_i = image[t : b + 1,l : r + 1]
        images.append(img_i)
    return images

def col_image_2(column_bounds, image, table_2, line_bounds):
    t = []
    for i in range(0, table_2.shape[0]):
        t.append(line_bounds[table_2[i],2])
    if len(t):
        top = int(min(t))
    else:
        top = 0
    n_imgs = column_bounds.shape[0]
    images = []
    print(column_bounds)
    for i in range(0, n_imgs):
        l,r,t,b = column_bounds[i]
        shape_1 = r - l + 1
        shape_2 = b - t + 1
        img_i = np.zeros((shape_1,shape_2) , dtype = np.uint8)
        img_i = image[top : b + 1,l : r + 1]
        images.append(img_i)
        #print img_i.shape
    return images
        

bl_extra = -5
br_extra = 5
bt_extra = -2
bb_extra = 2

def blck_image(bounds, image):
    l,r,t,b = bounds
    l = max(int(l) + bl_extra, 0) 
    r = int(r) + br_extra 
    t = max(int(t) + bt_extra, 0) 
    b = int(b) + bb_extra 
    shape_1 = int(r - l + 1)
    shape_2 = int(b - t + 1)
    img_i = np.zeros((shape_1,shape_2) , dtype = np.uint8)
    img_i = image[t : b + 1,l : r + 1]
    return img_i

def text_it(arr):
    #config = ('-l eng --oem 1 --psm 6 ')
    config = ('-l eng --oem 2 --psm 6 ')
    arr = np.uint8(arr)
    im_pil = Image.fromarray(arr)
    a,b = arr.shape
    text = ''
    if a and b:
        size =  b * 4 , a * 4
    
        im_resized = im_pil.resize(size, Image.ANTIALIAS)
        im_np = np.asarray(im_resized)
        im_bw = cv2.bitwise_not(im_np)
    
        kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
        sharpened = cv2.filter2D(im_bw, -1, kernel_sharpening)
        text = pytesseract.image_to_string(sharpened , config=config)
        print(text)
    return text


def block_to_text(block_column,strings,index, block_bounds, image, image2):
    blocks = block_column[block_column[:,1] == index][:,0]
    #print(blocks)
    strings_d = strings.split('\n')
    strings_f = []
    for i in range(0, len(strings_d)):
        strr = strings_d[i]
        if len(strr) and strr != ' ' and strr != '  ' and strr != '   ':
            strings_f.append(strr)
    blcks = list(blocks)
    if len(blcks) != len(strings_f):
#        print 'failedddddddddddddddddddddddddddddddddd'
        strings_f = blck_txt(block_column, block_bounds, index, image2)
#        print(strings_f)
    b_2_s = dict(zip(blcks,strings_f))
    return b_2_s

def blck_txt(block_column, block_bounds, index, image):
    blocks = block_column[block_column[:,1] == index][:,0]
    strings_g = []
    
    for i in range(0, blocks.shape[0]):
#        print(i)
        blck = blocks[i]
        bounds = np.zeros(4)
        bounds = block_bounds[blck]
#        bounds_1 = bounds.astype(int)
        img = blck_image(bounds, image)
        cv2.imwrite('blck_img.jpg', img)
        if img.shape[0] and img.shape[1]:
            strr = text_it(img)
            strr = strr
        else:
            strr = ' ' 
#        print strr
        strings_g.append(strr)
    return strings_g
        
    
def bl_bounds(comp_textb_label,stats):
    n_blocks = np.unique(comp_textb_label).shape[0]
#    print(n_blocks)
    block_bounds = np.zeros((n_blocks-1,4))
    for i in range(1,n_blocks):
        text_b = np.argwhere(comp_textb_label == i)[:,0]
        l = 0
        r = 0
        t = 0
        b = 0
        bounds_mat = np.zeros((text_b.shape[0],4))
        for j in range(0,text_b.shape[0]):
            bounds_mat[j] = bounds(stats,text_b[j])
        if text_b.shape[0] :
            l = np.min(bounds_mat[:,0])
            r = np.max(bounds_mat[:,1])
            t = np.min(bounds_mat[:,2])
            b = np.max(bounds_mat[:,3])
            block_bounds[i-1] = l,r,t,b
    return block_bounds

def re_arrange(array,label):
    new_array = -1 * np.ones_like(array)
    max_value = int(np.max(array))
    new_label = label
    for i in range(0, max_value + 1):
        counter = i
        components = np.argwhere(array == counter)[:,0]
        if components.shape[0]:
            for j in range(0, components.shape[0]):
                index = components[j]
                new_array[index] = new_label
            new_label += 1
    return new_array

def block_check(block_bounds, comp_textb_label,thresh, line_bnds):
    comp_textb_label_1 = comp_textb_label
    n_blocks = np.unique(comp_textb_label).shape[0]
    blocks_1 = np.unique(comp_textb_label)
    blocks_1 = blocks_1.astype(int)
    block_label = np.zeros((n_blocks,2))
    block_label[:,0] = blocks_1
    block_label[:,1] = blocks_1
    block_mat = -1 * np.ones((n_blocks - 1,n_blocks - 1))
#    print(n_blocks)
    for i in range(0, n_blocks - 1):
        block_i = blocks_1[i]
        index_i = int(np.argwhere(blocks_1 == block_i)[0][0]) 
        for j in range(0, i):
            block_j = blocks_1[j]
            index_j = int(np.argwhere(blocks_1 == block_j)[0][0]) 
            if block_line(block_bounds, index_i, index_j):
                dist_1 = np.abs(block_left(block_bounds, index_j) - block_right(block_bounds,index_i))
                dist_2 = np.abs(block_right(block_bounds, index_j) - block_left(block_bounds,index_i))
                dist = min(dist_1,dist_2)
                if dist <= thresh and not(blck_ln_chck_ver(block_bounds, index_i, index_j, line_bnds)):
#                    print('yes')
#                    print(dist_1)
#                    print(dist_2)
                    block_mat[index_i,index_j] = 1
                    block_mat[index_j,index_i] = 1
#    print(block_mat)
    for i in range(0, n_blocks - 1):
        block_i = blocks_1[i + 1] 
        index_i = int(np.argwhere(blocks_1 == block_i)[0][0]) 
        bl_row = block_mat[index_i - 1] 
        cand = np.argwhere(bl_row == 1)[:,0]
        if cand.shape[0]:
            label = block_i
            for j in range(0, cand.shape[0]):
                index_j = cand[j] + 1
                block_label[index_j,1] = label
                block_mat[index_i - 1,index_j - 1] = -1
                block_mat[index_j - 1,index_i - 1] = -1
    new_label = np.argwhere(block_label[:,0] != block_label[:,1])[:,0]
    if new_label.shape[0]:
        for i in range(0, new_label.shape[0]):
            block_i = new_label[i]
            block = blocks_1[block_i]
            new_lbl = block_label[block_i,1]
            comp = np.argwhere(comp_textb_label == block)[:,0]
            if comp.shape[0]:
                for j in range(0,comp.shape[0]):
                    comp_i = comp[j]
                    comp_textb_label_1[comp_i] = new_lbl
                    
    return comp_textb_label_1,block_label

def n_columns(table_lines, block_count):
    blck_count = []
    for i in range(0, table_lines.shape[0]):
        index = table_lines[i]
        blck_count.append(block_count[index])
    stats = Counter(blck_count)
    columns = stats.most_common(1)[0][0]
    columns = max(blck_count)
    return int(columns)

#def n_tables(line_table_label, block_count):
#    tables = []
#    columns = []
#    derv = np.diff(line_table_label)
#    derv_check = np.abs(derv)
#    if np.sum(derv_check) > 0:
#        ones = np.argwhere(derv == 1)[:,0]
#        ones = ones.astype(int)
#        nv_ones = np.argwhere(derv == -1)[:,0]
#        nv_ones = nv_ones.astype(int)
#        if ones.shape[0] == nv_ones.shape[0]:
#            for i in range(0, ones.shape[0]):
#                index_start = int(ones[i] + 1)
#                index_end = int(nv_ones[i])
#                lines = range(index_start , index_end + 1)
#                lines = np.array(lines)
#                tables.append(lines)
#                clmns = n_columns(lines , block_count)
#                columns.append(clmns)
#        else:
#             if ones.shape[0] > nv_ones.shape[0]:
#                 nv_ones = np.concatenate((nv_ones, int(int(line_table_label.shape[0] - 1))))
#                 for i in range(0, ones.shape[0]):
#                    index_start = int(ones[i] + 1)
#                    index_end = int(nv_ones[i])
#                    lines = range(index_start , index_end + 1)
#                    lines = np.array(lines)
#                    tables.append(lines)
#                    clmns = n_columns(lines , block_count)
#                    columns.append(clmns)
#             else:
#                 ones = np.concatenate(([-1], ones))
#                 for i in range(0, ones.shape[0]):
#                    index_start = int(ones[i] + 1)
#                    index_end = int(nv_ones[i])
#                    lines = range(index_start , index_end + 1)
#                    lines = np.array(lines)
#                    tables.append(lines)
#                    clmns = n_columns(lines , block_count)
#                    columns.append(clmns)
#    else:
#        tables.append(line_table_label)
#        clmns = n_columns(line_table_label)
#        columns.append(clmns)
#        
#    return tables,columns

def n_tables(line_table_label, block_count):
    tables = []
    columns = []
    tabl_lines = np.argwhere(line_table_label == 1)[:,0]
    if tabl_lines.shape[0]:
        start = 0
        start_l = tabl_lines[start]
        while start_l != tabl_lines[-1]:
            
            table_i = []
            table_i.append(start_l)
            while tabl_lines[start + 1] - tabl_lines[start] == 1:
                table_i.append(tabl_lines[start + 1])
                start += 1
#                print start
#                print table_i
#                print 'TABLEEE'
                if tabl_lines[start] == tabl_lines[-1]:
                    break
            table_i = np.array(table_i)
            tables.append(table_i)
            clmns = n_columns(table_i, block_count)
            columns.append(clmns)
            end_l = tabl_lines[start]
            if end_l == tabl_lines[-1]:
                break
            start_l = tabl_lines[start + 1]
            start += 1
            
    return tables,columns



def blck_ln_chck_ver(block_bounds, block1, block2, line_bounds):
    bounds1 = block_bounds[block1]
    bounds2 = block_bounds[block2]
#    b1 = bounds1[3] 
#    t2 = bounds2[2]
#    b = t2 - 1
#    t = b1 + 1
    l = min(bounds1[1], bounds2[1])
    r = max(bounds1[0], bounds2[0])
    b = max(bounds1[3], bounds2[3])
    t = min(bounds1[2], bounds2[2])
    bounds_w = np.zeros(4)
    bounds_w = l,r,t,b
    list_1 = np.argwhere(line_bounds[:,0] < r)[:,0]
    list_2 = np.argwhere(line_bounds[:,1] > l)[:,0]
    list_3 = np.intersect1d(list_1,list_2)
    list_4 = np.argwhere(line_bounds[:,2] < b)[:,0]
    list_5 = np.intersect1d(list_3,list_4)
    list_6 = np.argwhere(line_bounds[:,3] > t)[:,0]
    list_7 = np.intersect1d(list_5, list_6)
    if list_7.shape[0]:
        exist = 1
    else:
        exist = 0
    return exist



def blck_ln_chck(block_bounds, block1, block2, line_bounds):
    bounds1 = block_bounds[block1]
    bounds2 = block_bounds[block2]
#    b1 = bounds1[3] 
#    t2 = bounds2[2]
#    b = t2 - 1
#    t = b1 + 1
    l = max(bounds1[0], bounds2[0])
    r = min(bounds1[1], bounds2[1])
    b = max(bounds1[2], bounds2[2])
    t = min(bounds1[3], bounds2[3])
    bounds_w = np.zeros(4)
    bounds_w = l,r,t,b
    list_1 = np.argwhere(line_bounds[:,2] < b)[:,0]
    list_2 = np.argwhere(line_bounds[:,3] > t)[:,0]
    list_3 = np.intersect1d(list_1,list_2)
    list_4 = np.argwhere(line_bounds[:,0] < r)[:,0]
    list_5 = np.intersect1d(list_3,list_4)
    if list_5.shape[0]:
        exist = 1
    else:
        exist = 0
    return exist

def line_bnds(stats):
    line_bounds = np.zeros((stats.shape[0],4))
    line_bounds[:,0] = stats[:,0]
    line_bounds[:,1] = stats[:,0] + stats[:,2] - 1
    line_bounds[:,2] = stats[:,1]
    line_bounds[:,3] = stats[:,1] + stats[:,3] - 1
    return line_bounds

def cmp_ln_chck(stats, comp1, comp2, line_bounds):
    if left(stats,comp1) < left(stats,comp2):
        l = right(stats,comp1)
        r = left(stats,comp2)
    else:
        r = left(stats,comp1)
        l = right(stats,comp2)
    t = max(top(stats, comp1), top(stats, comp2))
    b = min(bottom(stats,comp1), bottom(stats,comp1))
    list_1 = np.argwhere(line_bounds[:,0] < r)[:,0]
    list_2 = np.argwhere(line_bounds[:,1] > l)[:,0]
    list_3 = np.intersect1d(list_1,list_2)
    list_4 = np.argwhere(line_bounds[:,2] < b)[:,0]
    list_5 = np.intersect1d(list_3,list_4)
    list_6 = np.argwhere(line_bounds[:,3] > t)[:,0]
    list_7 = np.intersect1d(list_5, list_6)
    
    if list_7.shape[0]:
        exist = 1
    else:
        exist = 0
    return exist

def overlap(blocks,block,block_bounds):
    overlap_arr = []
    for i in range(0, blocks.shape[0]):
        blck_i = blocks[i]
        if block_top(block_bounds, blck_i) > block_top(block_bounds, block):
            overlap = block_top(block_bounds,blck_i) - block_bottom(block_bounds, block)
        else:
            overlap = block_bottom(block_bounds,blck_i) - block_top(block_bounds,block)
        overlap_arr.append(overlap)
    overlap_arr = np.array(overlap_arr)
    min_overlap = max(overlap_arr)
    index = np.argwhere(overlap_arr == min_overlap)[0][0]
    blck_ret = blocks[index]
    block_arr = np.delete(blocks, index)
    return blck_ret, block_arr
    
def blck_ln_align(blocks, block, block_bounds):
    in_line = 0
    for i in range(0, blocks.shape[0]):
        blck = blocks[i]
        in_line = block_line(block_bounds, blck, block)
        if in_line == 0:
            break
    
    return in_line

def noise_reduction(input_cc,input_bw):
    nlabels_1,labels,stats_1,centroids_1 = cv2.connectedComponentsWithStats(input_cc)
    factor = 3.
    x_median = np.median(stats_1[:,2])
    y_median = np.median(stats_1[:,3])
    x_thresh = np.ceil(x_median/factor)
    y_thresh = np.ceil(y_median/factor)
    x_thresh_f = 2
    y_thresh_f = 2
    l1 = np.argwhere(stats_1[:,2] <= x_thresh)[:,0]
    l2 = np.argwhere(stats_1[:,3] <= y_thresh)[:,0]
    remove = np.intersect1d(l1,l2)
    for i in range(0,remove.shape[0]):
        r_list = np.argwhere(labels == remove[i])
        for j in range(0, r_list.shape[0]):
            tupp = tuple(r_list[j])
            input_bw[tupp] = 255
            
    l3 = np.argwhere(stats_1[:,3] == 1)[:,0]
    for i in range(0,l3.shape[0]):
        r_list = np.argwhere(labels == l3[i])
        for j in range(0, r_list.shape[0]):
            tupp = tuple(r_list[j])
            input_bw[tupp] = 255
    cv2.imwrite('after.jpg',input_bw)
    l4 = np.argwhere(stats_1[:,2] <= x_thresh_f)[:,0]
    l5 = np.argwhere(stats_1[:,3] <= y_thresh_f)[:,0]
    remove = np.intersect1d(l4,l5)
    for i in range(0,remove.shape[0]):
        r_list = np.argwhere(labels == remove[i])
        for j in range(0, r_list.shape[0]):
            tupp = tuple(r_list[j])
            input_bw[tupp] = 255
    cv2.imwrite('after_1.jpg',input_bw)
    l6 = np.argwhere(stats_1[:,3] < 3)[:,0]
    for i in range(0,l6.shape[0]):
        r_list = np.argwhere(labels == l6[i])
        for j in range(0, r_list.shape[0]):
            tupp = tuple(r_list[j])
            input_bw[tupp] = 255
    return input_bw

def consc_comp_det(stats):
    consc_det = np.zeros((stats.shape[0],2),int)
    consc_dist = []
    for i in range(0,stats.shape[0]):
        if stats[i,4] >= 2:
            consc_det[i] = consc_comp(stats,i)
        else:
            consc_det[i] = -1,-1
        if consc_det[i,1] != -1:
            consc_dist.append(consc_det[i,1])
            
    consc_dist = np.array(consc_dist)
    return consc_det, consc_dist

def block_form(nlabels,consc_det,consc_dist_hist, stats, line_bnds):
    n_blocks1 = 0
    comp_textb_label_1 = -1 * np.ones(nlabels)
    non_textb_l = np.argwhere(comp_textb_label_1 == -1)[:,0] 
    for i in range(0,non_textb_l.shape[0]):
        start = non_textb_l[i]
        if comp_textb_label_1[start] == -1:
            consc = consc_det[start,0]
            list_conn = [start]
            label = comp_textb_label_1[consc]
            
            while consc_det[start,0] != -1 and comp_textb_label_1[consc_det[start,0]] == -1:
                if consc_det[start,1] < consc_dist_hist and not(cmp_ln_chck(stats, start, consc_det[start,0], line_bnds)):
                    list_conn.append(consc_det[start,0])
                    start = consc_det[start,0]
    
                else:
                    break
    #        print list_conn
            if consc_det[start,0] != -1:
                label = comp_textb_label_1[consc_det[start,0]]
            else:
                label = -1
            if label == -1 :
                #print list_conn
                for j in range(0,len(list_conn)):
                    comp_textb_label_1[list_conn[j]] = n_blocks1
                n_blocks1 += 1
            elif label > -1 and consc_det[start,1] < consc_dist_hist and not(cmp_ln_chck(stats, start, consc_det[start,0], line_bnds)):
    #            print list_conn
    #            print label
                for j in range(0,len(list_conn)):
                    comp_textb_label_1[list_conn[j]] = label
            elif label > -1 and consc_det[start,1] > consc_dist_hist:
                for j in range(0,len(list_conn)):
                    comp_textb_label_1[list_conn[j]] = n_blocks1
                n_blocks1 += 1
        else:
            continue
        
        if comp_textb_label_1[non_textb_l[i]] == -1:
            start = non_textb_l[i]
            consc = consc_det[start,0]
            list_consc = np.argwhere(consc_det[:,0] == consc)[:,0]
            #print list_consc
            list_consc = list(list_consc)
            if len(list_consc) > 1:
                list_consc.remove(start)
                for j in range(0,len(list_consc)):
                    if comp_textb_label_1[list_consc[j]] > -1:
                        comp_textb_label_1[start] = comp_textb_label_1[list_consc[j]]
                       # print start
                        #print comp_textb_label_1[list_consc[j]]
                        break
    n_blocks1 += -1
    return comp_textb_label_1, n_blocks1

def blck_noise(block_bounds, comp_textb_label):
    h_fac = 1/3.
    w_fac = 1/3.
    h_thresh = np.median(block_bounds[:,3] - block_bounds[:,2] + 1) * h_fac
    w_thresh = np.median(block_bounds[:,1] - block_bounds[:,0] + 1) * w_fac
    for i in range(0, block_bounds.shape[0]):
        height = block_bounds[i,3] - block_bounds[i,2] + 1
        width = block_bounds[i,1] - block_bounds[i,0] + 1
        if height <= h_thresh:
#            print('deletedddddd')
            b_index = i + 1
            comp = np.argwhere(comp_textb_label == b_index)[:,0]
            for j in range(0,comp.shape[0]):
                index = comp[j]
                comp_textb_label[index] = 0
    return comp_textb_label

def consc_blck_det(block_bounds):
    conscB_det = np.zeros((block_bounds.shape[0],2),int)
    conscB_dist = []
    for i in range(0,block_bounds.shape[0]):
        #if stats[i,4] >= 2:
        conscB_det[i] = consc_block(block_bounds,i)
        #else:
         #   conscB_det[i] = -1,-1
        if conscB_det[i,1] != -1:
            conscB_dist.append(conscB_det[i,1])
    return conscB_det, conscB_dist

def line_form(comp_textb_label, conscB_det,block_bounds):
    n_lines = 1
    n_blocks = np.unique(comp_textb_label).shape[0] - 1
    textb_line_label = -1*np.ones(conscB_det.shape[0])
    EOL_1 = np.argwhere(conscB_det[:,1] == -1)[:,0]
    block_count = []
    for i in range(0,EOL_1.shape[0]):
        count = 0
        start_block = EOL_1[i]
        line_i = [start_block]
        textb_line_label[start_block] = n_lines
        count += 1
        extra = []
        while 1:
            line_block = np.argwhere(conscB_det[:,0] == start_block)[:,0]
            if line_block.shape[0]:
                if line_block.shape[0] == 1:
                    textb_line_label[line_block] = n_lines
                    start_block = line_block
                    count += 1
                else:
                    l_block,arr = overlap(line_block, start_block, block_bounds)
                    textb_line_label[int(l_block)] = n_lines
                    start_block = l_block
                    count += 1
                    for j in range(0, arr.shape[0]):
                        extra.append(int(arr[j]))
            else:
                break
#                if 0:       #len(extra)
#                    print extra
#                    print 'extra'
#                    for j in range(0, len(extra)):
#                        block = extra[j]
#                        line_j = [block]
#                        line_blck = np.argwhere(conscB_det[:,0] == block)[:,0]
#                        if line_blck.shape[0]:
#                            label = textb_line_label[line_blck]
#                            if label == -1:
#                                n_lines += 1
#                                textb_line_label[block] = n_lines
#                                textb_line_label[line_blck] = n_lines
#                                block = line_blck
#                                extra.remove(block)
#                                while 1:
#                                    line_blck = np.argwhere(conscB_det[:,0] == block)[:,0]
#                                    if line_blck.shape[0]:
#                                        textb_line_label[line_blck] = n_lines
#                                        block = line_blck
#                                        extra.remove(block)
#                                    else:
#                                        break
#                            else:
#                                 blocks = np.argwhere(textb_line_label == label)[:,0]
#                                 line_check = blck_ln_align(blocks, block, block_bounds)
#                                 if line_check == 1:
#                                     textb_line_label[block] = label
#                                     extra.remove(block)
#                                 else:
#                                     n_lines += 1
#                                     textb_line_label[block] = n_lines
#                                     extra.remove(block)
#                        else:
#                            break
#                else: 
#                    break
        block_count.append(count)
        n_lines += 1
        
    n_lines += -1
    return textb_line_label, n_lines, block_count

def line_form_2(textb_line_label, block_bounds, conscB_det, n_lines,block_count):
    blocks = np.argwhere(textb_line_label == -1)[:,0]
    blocks_2 = copy.copy(list(blocks))
    new_lines = n_lines + 1
    while len(blocks_2) > 0 :
        blck = [blocks_2[0]]
        start = blocks_2[0]
        for i in range(1, len(blocks_2)):
            consc = conscB_det[start,0]
            if consc in blocks_2:
                blck.append(consc)
                if conscB_det[consc,0] != -1:
                    start = consc
                else:
                    break
            else:
                break
        for j in range(0 , len(blck)):
            block = blck[j]
            textb_line_label[block] = new_lines
            blocks_2.remove(block)
        new_lines += 1
        block_count.append(len(blck))
    
    new_lines += -1
    return textb_line_label, new_lines, block_count

def line_sort(textb_line_label, line_bounds):
    t = line_bounds[:,2]
    blck_count = []
    sorted_t = np.sort(t)
    textb_line_label_1 =  np.zeros_like(textb_line_label)
    for i in range(0, t.shape[0]):
        label = i + 1
        new_i = np.argwhere(line_bounds[:,2] == sorted_t[i])[:,0]
        for k in range(0, new_i.shape[0]):
            block_old_i = np.argwhere(textb_line_label == new_i[k] + 1)[:,0]
            for j in range(0, block_old_i.shape[0]):
                block = block_old_i[j]
                if textb_line_label_1[block] == 0:
                    textb_line_label_1[block] = label
            blck_count.append(block_old_i.shape[0])
    return textb_line_label_1, blck_count
        
def ln_bnds(n_lines, textb_line_label,block_bounds):
    line_bounds = np.zeros((n_lines,4))
    for i in range(0,n_lines):
        line_block = np.argwhere(textb_line_label == i+1)[:,0]
        l = 0
        r = 0
        t = 0
        b = 0
        bounds_mat = np.zeros((line_block.shape[0],4))
        for j in range(0,line_block.shape[0]):
            bounds_mat[j] = block_bounds[line_block[j]]
        if line_block.shape[0]:
            l = np.min(bounds_mat[:,0])
            r = np.max(bounds_mat[:,1])
            t = np.min(bounds_mat[:,2])
            b = np.max(bounds_mat[:,3])
            line_bounds[i] = l,r,t,b    
    return line_bounds

def hor_sep_in(textb_line_label, block_bounds,n_lines,input_bw):
    hor_sep_inline = []
    for i in range(0,n_lines):
        line_blocks = np.argwhere(textb_line_label == i+1)[:,0]
       # line_blocks += 1
        bound_t = block_bounds[line_blocks,2]
        bound_b = block_bounds[line_blocks,3]
        bound_l = block_bounds[line_blocks,0]
        bound_r = block_bounds[line_blocks,1]
        t = int(min(bound_t))
        b = int(max(bound_b))
        l = 0
        for j in range(0,line_blocks.shape[0]):
            r = bound_l[j] - 1
            bounds_b = (l,r,t,b,i+1)
            hor_sep_inline.append(bounds_b)
            l = bound_r[j] + 1
        img_bounds = int(input_bw.shape[1])
        r = img_bounds
        bounds_b = (l,r,t,b,i+1)
        hor_sep_inline.append(bounds_b)
    hor_sep_inline = np.array(hor_sep_inline,dtype = int)
    return hor_sep_inline
        
def hor_sep_out(n_lines, line_bounds, input_bw):
    hor_sep_outline = []
    
    t_sep = 0
    l_sep = 0
    img_bounds = int(input_bw.shape[1])
    r_sep = img_bounds
    for i in range(0,n_lines):
        b_sep = line_bounds[i,2] - 1
        bounds_b = (l_sep,r_sep,t_sep,b_sep)
        hor_sep_outline.append(bounds_b)
        t_sep = line_bounds[i,3] + 1
    
    b_sep = int(input_bw.shape[0])
    bounds_b = (l_sep,r_sep,t_sep,b_sep)
    hor_sep_outline.append(bounds_b)
    hor_sep_outline = np.array(hor_sep_outline,dtype = int)
    return hor_sep_outline

def table_detect(n_lines, block_count, textb_line_label, block_bounds):
    line_table_label = np.zeros(n_lines)
    block_freq = list(Counter(block_count))
    for i in range(0,len(block_freq)):
        count = block_freq[i]
        if count > 1:
            freq_lines = np.argwhere(block_count == count)[:,0]
#            print(freq_lines)
            len_f = freq_lines.shape[0]
            align_mat = np.zeros((len_f,len_f))
            for i in range(0,len_f):
                line_i = freq_lines[i] + 1
                for j in range(0,i):
                    line_j = freq_lines[j] + 1
                    align_mat[i,j] = line_align(textb_line_label,block_bounds,line_i,line_j)
                    align_mat[j,i] = align_mat[i,j]
#            print(align_mat)
            sum_t = np.sum(align_mat,axis = 0)
#            print(max(sum_t))
#            print(sum_t)
            while max(sum_t) >= 2 and align_mat.shape[0]:
#                print('once only')
                base = np.argwhere(sum_t == max(sum_t))[0][0]
                align_lines = np.argwhere(align_mat[:,base] == 1)[:,0]
#                print(align_lines)
                align_index = [freq_lines[base]]
                align_mat[:,base] = -1
                #align_index.append(freq_lines[base])
                for j in range(0,align_lines.shape[0]):
                    index = align_lines[j]
                    align_index.append(freq_lines[index])
                    line_table_label[freq_lines[index]] = 1
                    align_mat[:,index] = -1
                
                sum_t = np.sum(align_mat,axis = 0)
#                print(align_mat)
#                print(align_index)
        
    ground = np.argwhere(line_table_label == 1)[:,0]   
    expand = np.ones(ground.shape[0])
    expansion = np.sum(expand) 
    leap = 1
    while expansion and leap <= 2:
#        print('leap')
        for i in range(0,ground.shape[0]):
#            print(i)
            check_t = 0
            check_b = 0
            index = ground[i]
            if index - leap + 1 :
                check_t = line_table_label[index - leap]
                if check_t == 0:
                    blocks_g = block_count[index]
                    blocks_t = block_count[index - leap]
                    blocks_g_s = sort_lineb(textb_line_label, block_bounds, index + 1)
                    blocks_t_s = sort_lineb(textb_line_label, block_bounds, index + 1 - leap)
                    check_s_t = 0
                    counter = 0
                    for k in range(0, len(blocks_t_s)):
                        check_s_t = 0
                        for j in range(0, len(blocks_g_s)):
                            check_s_t += vertical_align(block_bounds, blocks_g_s[j] ,blocks_t_s[k])
                        if check_s_t > 1:
                            check_t = 1
                            break
                        counter += check_s_t
                    if counter == blocks_t:
                        line_table_label[index - leap] = 1
                    check_t = counter == blocks_t
                
            if index < n_lines - leap:
                check_b = line_table_label[index + leap]
                if check_b == 0:
                    blocks_g = block_count[index]
                    blocks_b = block_count[index - leap]
                    blocks_g_s = sort_lineb(textb_line_label, block_bounds, index + 1)
                    blocks_b_s = sort_lineb(textb_line_label, block_bounds, index + leap + 1)
                    check_s_b = 0
                    counter = 0
                    for k in range(0, len(blocks_b_s)):
                        check_s_b = 0
                        for j in range(0, len(blocks_g_s)):
                            check_s_b += vertical_align(block_bounds, blocks_g_s[j] ,blocks_b_s[k])
                        if check_s_b > 1:
                            check_b = 1
                            break
                        counter += check_s_b
                    check_b = counter == blocks_b
                    if counter == blocks_b :
                        line_table_label[index + leap] = 1
                    
            if check_t + check_b == 2:
                expand[i] = 0
        
        expansion = np.sum(expand)
        leap += 1
    return line_table_label



def main_table(input_im1, row_print , ver_line, hor_line ):
    #input_im1, ver_line, hor_line = line_removal(input_im)[1:4]
    #input_imx = copy.copy(np.uint8(input_im1))
    #cv2.imwrite('inputim1.jpg',np.uint8(input_im1))
    
    #input_im1 = text_inversion(np.uint8(input_im1))
    #cv2.imwrite("text_inversion.jpg" , np.uint8(input_im1))
    
    #input_im2, ver_line2, hor_line2 = line_removal(input_im1)[1:4]
    #cv2.imwrite('text_inversion_line_removal.jpg',np.uint8(input_im2))
    nlabels_hor,labels_hor,stats_hor,centroids_hor = cv2.connectedComponentsWithStats(hor_line)
    nlabels_ver,labels_ver,stats_ver,centroids_ver = cv2.connectedComponentsWithStats(ver_line)
    line_bounds_hor = line_bnds(stats_hor)
    line_bounds_ver = line_bnds(stats_ver)
    line_bounds_hor = line_bounds_hor[1:]
    line_bounds_ver = line_bounds_ver[1:]
    
    val,input_bw=cv2.threshold( np.uint8(input_im1) ,50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    vert_mode = vertical_mode(input_bw)

    print(vert_mode)
    cv2.imwrite('bw.jpg',input_bw)
    input_im2 = copy.copy(input_im1)

    input_cc=~input_bw

    cv2.imwrite('before.jpg',input_bw)
    input_bw = noise_reduction(input_cc, input_bw )
    cv2.imwrite('after_2.jpg',input_bw)

    input_bw = cv2.GaussianBlur(input_bw , (3,1) ,0)
    cv2.imwrite('after_3.jpg',input_bw)

    nlabels,labels_1,stats,centroids = cv2.connectedComponentsWithStats(~input_bw)
#    label_hue = np.uint8(179*labels_1/np.max(labels_1))
#    blank_ch = 255*np.ones_like(label_hue)
#    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    
    consc_det,consc_dist = consc_comp_det(stats)    ####fun call 2

    consc_dist_median = np.median(consc_dist)    
    consc_dist_mean = np.mean(consc_dist)
    consc_dist_hist = 1.0*vert_mode                                      #plt.hist(consc_dist)[1][1]+100
    print(plt.hist(consc_dist))
    print(consc_dist_hist)
    #consc_dist_hist = 10

    comp_textb_label, n_blocks = block_form(nlabels,consc_det,consc_dist_hist, stats, line_bounds_ver)
#    print(n_blocks)
#    print('out')
    comp_textb_label_1 = comp_textb_label

    block_bounds = bl_bounds(comp_textb_label,stats)
    comp_textb_label = blck_noise(block_bounds, comp_textb_label)
    comp_textb_label = re_arrange(comp_textb_label,0)
    block_bounds = bl_bounds(comp_textb_label,stats)
    bll_bnds = block_bounds
    comp_textb_label,chck_lbls = block_check(block_bounds,comp_textb_label,consc_dist_hist , line_bounds_ver)
    #check_lbls = block_check(block_bounds,comp_textb_label,consc_dist_hist)[1]
    comp_textb_label = re_arrange(comp_textb_label,0)
    block_bounds = bl_bounds(comp_textb_label,stats)
    conscB_det,conscB_dist = consc_blck_det(block_bounds)
 
    conscB_dist = np.array(conscB_dist)

    conscB_dist_median = np.median(conscB_dist)    
    conscB_dist_mean = np.mean(conscB_dist)
    
    textb_line_label, n_lines ,block_count= line_form(comp_textb_label, conscB_det,block_bounds)
        
    block_count = np.array(block_count)
    
    line_bounds = ln_bnds(n_lines, textb_line_label,block_bounds)

    hor_sep_inline = hor_sep_in(textb_line_label, block_bounds,n_lines,input_bw)

    hor_sep_outline = hor_sep_out(n_lines, line_bounds,input_bw)
    n_blocks = np.unique(comp_textb_label).shape[0]
    v_align = -1*np.ones((n_blocks-1,n_blocks-1))
    for i in range(0,v_align.shape[0]):
        for j in range(i + 1,v_align.shape[1]):
            v_align[i,j] = vertical_align(block_bounds,i,j)
            v_align[j,i] = v_align[i,j]
    
    block_v_align = []
    for i in range(0,v_align.shape[0]):
        aligned = np.argwhere(v_align[i] == 1)[:,0]
        block_v_align.append(aligned)


    img_test = 0*np.ones_like(input_bw)
    for i in range(0,n_blocks-1):
        l,r,t,b = block_bounds[i]
        for j in range(int(l),int(r)):
            for k in range(int(t),int(b)):
                img_test[k,j] = 255
    cv2.imwrite('img_check.jpg',img_test)
    
    img_test_1 = 0*np.ones_like(input_bw)
    for i in range(0,n_lines):
        l,r,t,b = line_bounds[i]
        for j in range(int(l),int(r)):
            for k in range(int(t),int(b)):
                img_test_1[k,j] = 255
    cv2.imwrite('img_check_1.jpg',img_test_1)
    
    line_table_label = table_detect(n_lines, block_count, textb_line_label, block_bounds)
   
    tables, colmns = n_tables(line_table_label, block_count)
    #colmns = n_tables(line_table_label, block_count)[1]
    num_tables = len(tables)
    for i in range(0, num_tables):
#        print('welcome')
        table = tables[i]
        n_colmns = colmns[i]
        if table.shape[0] > 2:
            block_column = table_column(table,n_colmns,textb_line_label,block_count, block_bounds)
            block_table_line = align_table(table,n_colmns, block_count,textb_line_label, block_bounds)
            table_detail = table_det(table, textb_line_label, n_colmns, block_count, block_bounds)
            column_bounds = col_bounds(block_column, block_bounds)
            images = col_image(column_bounds, input_im2)
            n_images = len(images)
            strings = []
            block_dict = {}
            for j in range(0,n_images):
                image_in = images[j]
                string = text_it(image_in)
#                print(j)
                b_2_s = block_to_text(block_column, string, j, block_bounds, image_in, input_im2)
#                print(b_2_s)
                
                block_dict.update(b_2_s)
                strings.append(string)
                f_name = 'input_im_' + str(j) + '.jpg'
                cv2.imwrite(f_name, image_in)
#            print(block_dict)
            table_print = []
            n_rows = len(table_detail)
            for k in range(0 ,n_rows):
                row_pr =[]
                row_i = table_detail[k]
                cells = len(row_i)
                for j in range(cells - 1, -1 ,-1):
                    cell = row_i[j]
                    if len(cell):
                        strr = ''
                        for l in range(0,len(cell)):
                            strr += block_dict[cell[l]]
                            strr += ' '
                    else:
                        strr = ' '
                    row_pr.append(strr)
                table_print.append(row_pr)
            
            if n_rows > 2:
                worksheet.write(row_print,0,i + 1)
                for j in range(0 , n_rows):
                    row_i = table_print[j]
                    cells = len(row_i)
                    for k in range(0, cells):
                        cell = row_i[k]
                        cell = cell              #.decode(encoding="utf-8", errors="ignore")
                        worksheet.write(row_print + 1,k,cell)
                    row_print += 1
                row_print += 1
        
    
    return row_print

def header_row(table_lines,line_bounds, line_bounds_hor, n_columns, block_count):
    line_bounds_t = line_bounds[table_lines]
    line_s = -1
    for i in range(0, table_lines.shape[0] - 1):
        line_i = table_lines[i]
        line_j = table_lines[i + 1]
        line_b_i = line_bounds_t[i]
        line_b_j = line_bounds_t[i + 1]
        check = blck_ln_chck(line_bounds, line_i,line_j, line_bounds_hor)
        print(check)
        if check == 1:
            if line_i:
                header_rows = table_lines[0 : line_i + 1]
                blck_cnt_h = block_count[header_rows]
                max_count = max(blck_cnt_h)
            else:
                header_rows = table_lines[0]
                blck_cnt_h = (block_count[header_rows])
                max_count = blck_cnt_h
            if 1 :
#                max_count = max(blck_cnt_h)
                if max_count < n_columns - 1 :
                    print (max_count)
                    print(n_columns)
                    line_s = line_i
                    continue
                else:
                    break
            else:
                continue
    print(line_s)
    print('line_s')
    return line_i, line_s

def header_cells(table_lines, header_row, n_columns, block_count,textb_line_label, block_bounds, image, line_ver,start_row):
    index = np.argwhere(table_lines == header_row)[0][0]
    header_rows = table_lines[start_row + 1 : index + 1]
    normal_rows = table_lines[index + 1:]
    blck_cnt_h = block_count[header_rows]
    blck_cnt_n = block_count[normal_rows]
    print(header_rows)
    print(normal_rows)
    index_n = -1
    index_h = -1
    index_h_1 = -1
    index_n1 = np.argwhere(blck_cnt_n == n_columns)[:,0]
    if index_n1.shape[0]:
        index_n = normal_rows[index_n1[0]]
    index_h1 = np.argwhere(blck_cnt_h == n_columns)[:,0]
    if index_h1.shape[0]:
        index_h = header_rows[index_h1[0]]
    index_h2 = np.argwhere(blck_cnt_h == n_columns - 1)[:,0]
    if index_h2.shape[0]:
        index_h_1 = header_rows[index_h2[0]]
    headers = []
    
    blck_column = table_column(normal_rows, n_columns, textb_line_label, block_count, block_bounds)
    column_bounds = col_bounds(blck_column, block_bounds)
    column_bounds_1 = col_bounds_2(column_bounds, image,line_ver)
    print(column_bounds_1)
    
    if index_n:
        blocks_n = np.argwhere(textb_line_label == index_n + 1)[:,0]
        for i in range(0, header_rows.shape[0]):
            row = []
            row_i = header_rows[i]
            blocks = np.argwhere(textb_line_label == row_i + 1)[:,0]
            print(blocks)
            print(n_columns)
            for j in range(0, n_columns):
                block_n = blocks_n[j]
                cell = []
                col_bnd_j = column_bounds_1[j]
                for k in range(0, blocks.shape[0]):
                    block_j = blocks[k]
                    check_p = vertical_align_2(col_bnd_j, block_bounds[block_j])
                    check_k = vertical_align(block_bounds, block_j, block_n)
                    if check_k or check_p:
                        cell.append(block_j)
                row.append(cell)
            headers.append(row)

    
    check_1 = np.argwhere(blck_cnt_h == n_columns)[:,0]
    index_1 = header_rows.shape[0]
    headers2 = copy.copy(headers)
    if check_1.shape[0]:
        print('check1')
        index_1 = check_1[0]
#        print(index_1)
        m_cells = headers[index_1]
        for i in range(index_1 + 1,header_rows.shape[0]):
#            print(m_cells)
            h_cells = headers2[i]
            for j in range(0, len(m_cells)):
                cell_m = m_cells[j]
                cell_h = h_cells[j]
                for k in range(0, len(cell_h)):
                    cell = cell_h[k]
                    cell_m.append(cell)
            headers.remove(h_cells)
    print(headers)
    print (len(headers))
    headers2 = copy.copy(headers)
    check_2 = np.argwhere(blck_cnt_h == n_columns - 1)[:,0]
    print(blck_cnt_h)
    if check_2.shape[0]:
        print(headers)
        index_2 = check_2[0]
        if index_2 < index_1 :
            m_cells = headers[index_2]
            for j in range(index_2 + 1,index_1):
                ind = j
                if ind < index_1:
                    h_cells = headers2[j]
                    for k in range(0, len(m_cells)):
                        cell_m = m_cells[k]
                        cell_h = h_cells[k]
                        for l in range(0, len(cell_h)):
                            cell = cell_h[l]
                            cell_m.append(cell)
                    headers.remove(h_cells)
    print(headers)
    return headers

def table_det_1(table_lines, textb_line_label, columns, block_count, block_bounds,headers):
    block_column = table_column(table_lines, columns, textb_line_label,block_count, block_bounds)
    block_table_line = align_table(table_lines, columns, block_count, textb_line_label, block_bounds)
    n_columns = np.unique(block_column[:,1]).shape[0]
    n_rows = np.unique(block_table_line[:,1]).shape[0]
    rows = np.unique(block_table_line[:,1])
    rows = rows.astype(int)
#    shape_1 = int(n_columns * n_rows)
#    shape_2 = 5    #depends on the presence of sub headers
#    table_details = -1 *  np.ones((shape_1,shape_2))
#    header_row = block_table_line[block_table_line[:,1] == 0][:,0]
#    for i in range(0,int(n_rows)):
#        blocks_i = block_table_line[block_table_line[:,1] == i][:,0]
#        col_i = -1 * np.ones(blocks_i.shape[0])
#        for j in range(0,blocks_i.shape[0]):
#            block = blocks_i[j]
#            col_i[j] = block_column[block_column[:,0] == block][:,1][0]
#        n_cells = np.unique(col_i).shape[0]
#        cell_i = np.unique(col_i)
#        for j in range(0,int(n_cells)):
#            cell = cell_i[j]
#            cells = np.argwhere(col_i == cell)[:,0]
##            table_det[]
##            for k in range(0,cells.shape[0]):
    n_headers = len(headers)
    table_detail =[]
#    for i in range(0, n_headers):
#        row = rows[i]
#        h_cells = row_cells(block_column, block_table_line, row)
#        headers.append(h_cells)
        
    for i in range(0, int(n_rows)):
        row = rows[i]
        cells = row_cells(block_column, block_table_line, row)
        n_cells = len(cells)
        for j in range(1, n_cells):
            row_det = []
            cell = cells[j]
            row_det.append(cell)
            row_det.append([])
            for k in range(n_headers, 0 , -1):      #column headers
                h_index = k - 1
                cell_index = j
                h_cell = headers[h_index][cell_index]
                row_det.append(h_cell)
            row_det.append([])
            row_det.append(cells[0])       #row header
########## append row sub headers here##############
            for k in range(n_headers, 0 , -1):      #column headers of hrow header
                h_index = k - 1
                cell_index = 0
                h_cell = headers[h_index][cell_index]
                row_det.append(h_cell)
            table_detail.append(row_det)
    return table_detail

def main_table_1(input_im1, row_print , ver_line, hor_line):
    #input_im1, ver_line, hor_line = line_removal(input_im)[1:4]
    #input_imx = copy.copy(np.uint8(input_im1))
    #cv2.imwrite('inputim1.jpg',np.uint8(input_im1))
    
    #input_im1 = text_inversion(np.uint8(input_im1))
    #cv2.imwrite("text_inversion.jpg" , np.uint8(input_im1))
    cv2.imwrite("dhuai.jpg" , hor_line)

    nlabels_hor,labels_hor,stats_hor,centroids_hor = cv2.connectedComponentsWithStats(hor_line)
    nlabels_ver,labels_ver,stats_ver,centroids_ver = cv2.connectedComponentsWithStats(ver_line)
    line_bounds_hor = line_bnds(stats_hor)
    line_bounds_ver = line_bnds(stats_ver)
    line_bounds_hor = line_bounds_hor[1:]
    line_bounds_ver = line_bounds_ver[1:]
    print(line_bounds_hor)
    
    val,input_bw=cv2.threshold(np.uint8(input_im1) ,100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    vert_mode = vertical_mode(input_bw)
    
    cv2.imwrite('bw.jpg',input_bw)
    input_im2 = copy.copy(input_im1)

    input_cc=~input_bw

    cv2.imwrite('before.jpg',input_bw)
    input_bw = noise_reduction(input_cc, input_bw )
    cv2.imwrite('after_2.jpg',input_bw)

    input_bw = cv2.GaussianBlur(input_bw , (3,1) ,0)
    cv2.imwrite('after_3.jpg',input_bw)

    nlabels,labels_1,stats,centroids = cv2.connectedComponentsWithStats(~input_bw)
    print(nlabels)
#    label_hue = np.uint8(179*labels_1/np.max(labels_1))
#    blank_ch = 255*np.ones_like(label_hue)
#    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    consc_det,consc_dist = consc_comp_det(stats)    ####fun call 2

    consc_dist_median = np.median(consc_dist)    
    consc_dist_mean = np.mean(consc_dist)
    consc_dist_hist = vert_mode*1.0                       #plt.hist(consc_dist)[1][1]
    print(plt.hist(consc_dist))
    #consc_dist_hist = 10

    comp_textb_label, n_blocks = block_form(nlabels,consc_det,consc_dist_hist , stats, line_bounds_ver)
#    print(n_blocks)
#    print('out')
    comp_textb_label_1 = comp_textb_label

    block_bounds = bl_bounds(comp_textb_label,stats)
    comp_textb_label = blck_noise(block_bounds, comp_textb_label)
    comp_textb_label = re_arrange(comp_textb_label,0)
    block_bounds = bl_bounds(comp_textb_label,stats)
    bll_bnds = block_bounds
    comp_textb_label,chck_lbls = block_check(block_bounds,comp_textb_label,consc_dist_hist , line_bounds_ver)
    #check_lbls = block_check(block_bounds,comp_textb_label,consc_dist_hist)[1]
    comp_textb_label = re_arrange(comp_textb_label,0)
    block_bounds = bl_bounds(comp_textb_label,stats)
    conscB_det,conscB_dist = consc_blck_det(block_bounds)
 
    conscB_dist = np.array(conscB_dist)

    conscB_dist_median = np.median(conscB_dist)    
    conscB_dist_mean = np.mean(conscB_dist)
    
    textb_line_label, n_lines ,block_count= line_form(comp_textb_label, conscB_det,block_bounds)
    print(textb_line_label)
    textb_line_label_2, n_lines_2, block_count = line_form_2(textb_line_label, block_bounds, conscB_det, n_lines, block_count)
    print(textb_line_label_2)
    block_count = np.array(block_count)
    line_bounds = ln_bnds(n_lines, textb_line_label,block_bounds)
    line_bounds_2 = ln_bnds(n_lines_2, textb_line_label_2, block_bounds)
    textb_line_label_2, block_count = line_sort(textb_line_label_2, line_bounds_2) 
    print(block_count)
    textb_line_label_2 = re_arrange(textb_line_label_2, 1)
    print(textb_line_label_2)
    
    textb_line_label = textb_line_label_2
    n_lines = n_lines_2
    n_lines = np.unique(textb_line_label_2).shape[0]
    block_count_2 = []
    for i in range(0, n_lines):
        blcks = np.argwhere(textb_line_label_2 == i + 1)[:,0]
        block_count_2.append(blcks.shape[0])
        
    block_count = np.array(block_count)
    block_count_2 = np.array(block_count_2)
    print(block_count.shape[0])
    print(block_count_2.shape[0])
    block_count = block_count_2
    line_bounds = ln_bnds(n_lines, textb_line_label,block_bounds)
    hor_sep_inline = hor_sep_in(textb_line_label, block_bounds,n_lines,input_bw)

    hor_sep_outline = hor_sep_out(n_lines, line_bounds,input_bw)
    n_blocks = np.unique(comp_textb_label).shape[0]
    v_align = -1*np.ones((n_blocks-1,n_blocks-1))
    for i in range(0,v_align.shape[0]):
        for j in range(i + 1,v_align.shape[1]):
            v_align[i,j] = vertical_align(block_bounds,i,j)
            v_align[j,i] = v_align[i,j]
    
    block_v_align = []
    for i in range(0,v_align.shape[0]):
        aligned = np.argwhere(v_align[i] == 1)[:,0]
        block_v_align.append(aligned)

    img_test = 0*np.ones_like(input_bw)
    for i in range(0,n_blocks-1):
        l,r,t,b = block_bounds[i]
        for j in range(int(l),int(r)):
            for k in range(int(t),int(b)):
                img_test[k,j] = 255
    cv2.imwrite('img_check.jpg',img_test)
    
    img_test_1 = 0*np.ones_like(input_bw)
    for i in range(0,n_lines):
        l,r,t,b = line_bounds[i]
        for j in range(int(l),int(r)):
            for k in range(int(t),int(b)):
                img_test_1[k,j] = 255
    cv2.imwrite('img_check_1.jpg',img_test_1)
    
    
    line_table_label = table_detect(n_lines, block_count, textb_line_label, block_bounds)
    table_lines_count = (np.argwhere(line_table_label == 1)[:,0]).shape[0]
    
    if table_lines_count > 2:
        line_table_label[0:] = 1
#        print(line_table_label)
#        print('table lines')
    
    tables, colmns = n_tables(line_table_label, block_count)
    #colmns = n_tables(line_table_label, block_count)[1]
    num_tables = len(tables)
    for i in range(0, num_tables):
#        print('WElcome')
        table = tables[i]
        n_colmns = colmns[i]
        if table.shape[0] > 2:
            block_column = table_column(table,n_colmns,textb_line_label,block_count, block_bounds)
            #print('blockkk columns')
#            block_table_line = align_table(table,n_colmns, block_count,textb_line_label, block_bounds)
            header_index, start_index = header_row(table, line_bounds, line_bounds_hor, n_colmns, block_count)
            h_cells = header_cells(table, header_index, n_colmns, block_count, textb_line_label, block_bounds, input_im1, line_bounds_ver, start_index)
            normal_rows = table[header_index + 1:]
            #print(normal_rows)
            #print(h_cells)
           # print(header_index)
            table_detail = table_det_1(normal_rows, textb_line_label, n_colmns, block_count, block_bounds, h_cells)
            column_bounds = col_bounds(block_column, block_bounds)
            table_2 = table[start_index + 1:]
            block_column_2 = table_column(table_2,n_colmns,textb_line_label,block_count, block_bounds)
            print(block_column_2)
            column_bounds_2 = col_bounds(block_column_2, block_bounds)
            images_2 = col_image_2(column_bounds_2, input_im2,table_2, line_bounds)
            print(table_2)
            print(block_column)
            print(column_bounds_2)
#            images = col_image(column_bounds, input_im2)
            images = images_2
            n_images = len(images)
            strings = []
            block_dict = {}
            for j in range(0,n_images):
                image_in = images[j]
                string = text_it(image_in)
                #print(j)
                b_2_s = block_to_text(block_column_2, string, j, block_bounds, image_in, input_im2)
                #print(b_2_s)
                
                block_dict.update(b_2_s)
                strings.append(string)
                f_name = 'input_im_' + str(j) + '.jpg'
                cv2.imwrite(f_name, image_in)
            #print(block_dict)
            table_print = []
            n_rows = len(table_detail)
            for k in range(0 ,n_rows):
                row_pr =[]
                row_i = table_detail[k]
                cells = len(row_i)
                for j in range(cells - 1, -1 ,-1):
                    cell = row_i[j]
                    if len(cell):
                        strr = ''
                        for l in range(0,len(cell)):
                            strr += block_dict[cell[l]]
                            strr += ' '
                    else:
                        strr = ' '
                    row_pr.append(strr)
                table_print.append(row_pr)
             
            worksheet.write(row_print,0,i + 1)
            for j in range(0 , n_rows):
                row_i = table_print[j]
                cells = len(row_i)
                for k in range(0, cells):
                    cell = row_i[k]
                    cell = cell     #.decode(encoding="utf-8", errors="ignore")
                    worksheet.write(row_print + 1,k,cell)
                row_print += 1
            row_print += 1
        
    
    return row_print 
 
# =============================================================================
# 
# =============================================================================


from PDF_2_JPG import file_name

path_f = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2"
path_out = "C:\Axis AI Challenge @ Akash_Abhishek\OUTPUT FILES"
#path_f_1 = r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2"
folders = os.listdir(path_f)
for i in range(0, len(folders)):
#    path_i = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES" 
    folder_i = folders[i]
    row_print = 0
    #path_out_i = path_out + '\output_' + str(i + 1) + '.xlsx'
    path_out_i = "C:\Axis AI Challenge @ Akash_Abhishek\OUTPUT FILES\{}.xlsx".format(file_name[i])
    workbook = xlsxwriter.Workbook(path_out_i)
    worksheet = workbook.add_worksheet() 
    path_i = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}".format(i + 1)
    folders_i = os.listdir(path_i)
    for p in range(0, len(folders_i)):
        folder_p = folders_i[p]
        if folder_p == 'Detected Tables':
            path_u = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables".format(i + 1) 
            folders_i = os.listdir(path_u)
            for j in range(0, len(folders_i)):
                index=int(folders_i[j])
                path_j = "C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables\{}".format(i + 1,index)
                folders_j = os.listdir(path_j)
                for k in range(0, len(folders_j)):
                    tables_1 = cv2.imread(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables\{}\{}\image.jpg".format(i + 1 , index , k + 1) ,0)
                    cv2.imwrite("adstable.jpg" , tables_1)
                    tables_vertical = cv2.imread(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables\{}\{}\vertical.jpg".format(i + 1 , index , k + 1) ,0)
                    cv2.imwrite("adsver.jpg" , tables_vertical)
                    tables_horizontal = cv2.imread(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\Detected Tables\{}\{}\horizontal.jpg".format(i + 1 ,index , k + 1) ,0)
                    cv2.imwrite("adshor.jpg" , tables_horizontal)
                    row_print = main_table_1(tables_1, row_print, tables_vertical, tables_horizontal)
                    
        
        else:
            tables_1 = cv2.imread(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\{}\image.jpg".format(i + 1 ,p + 1)  ,0)
            cv2.imwrite("ewewew.jpg", tables_1)
            tables_vertical = cv2.imread(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\{}\vertical.jpg".format(i + 1 ,p + 1)  ,0)
            cv2.imwrite("ewewver.jpg", tables_vertical)
            tables_horizontal = cv2.imread(r"C:\Axis AI Challenge @ Akash_Abhishek\PROCESSED FILES 2\{}\{}\horizontal.jpg".format(i + 1,p + 1)  ,0)
            cv2.imwrite("ewewhor.jpg", tables_horizontal)
            row_print = main_table(tables_1, row_print, tables_vertical, tables_horizontal)
    workbook.close() 

    
    
