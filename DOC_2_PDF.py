# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 03:38:25 2018

@author: Akash Singh
"""


import win32com.client
import glob

wdFormatPDF = 17

word = win32com.client.Dispatch('Word.Application')

docx = [word.Documents.Open(in_file) for in_file in glob.glob("C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.docx")]

docx_no = len(docx)
for i in range (0 , len(docx)):
    docx_no = i+1
    docx_orig = docx[i]    
    out_file = r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\PDF of DOCX_%d.pdf"%docx_no
    docx_orig.SaveAs(out_file, FileFormat=wdFormatPDF)
    docx_orig.Close()
    
doc = [word.Documents.Open(in_file) for in_file in glob.glob("C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\*.doc")]

doc_no = len(doc)
for i in range (0 , len(doc)):
    doc_no = i+1
    doc_orig = doc[i]    
    out_file = r"C:\Axis AI Challenge @ Akash_Abhishek\INPUT FILES\PDF of DOC_%d.pdf"%doc_no
    doc_orig.SaveAs(out_file, FileFormat=wdFormatPDF)
    doc_orig.Close()

total_word_file = docx_no + doc_no    
#word.Quit()
