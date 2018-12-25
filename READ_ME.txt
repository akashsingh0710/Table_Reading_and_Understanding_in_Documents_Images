1. Unzip the Files in C Drive. (So your path directory for all the files would look like this: "C:\Axis AI Challenge @ Akash_Abhishek\'Phython codes+ FOLDERS + Tesseract OCR exe file'"
2. Python Files in the 'Axis AI Challenge @ Akash_Abhishek' folder includes: (MAIN.py , Combined.py, DOC_2_PDF.py, PDF_2_JPG.py)
3. Sub-folders in the 'Axis AI Challenge @ Akash_Abhishek' folder includes: (INPUT FILES, PROCESSED FILES 1, PROCESSED FILES 2, OUTPUT FILES, poppler-0.68.0)
4. We have attached Tesseract OCR (tesseract-ocr-setup-4.00.00dev.exe) with our files, install it if you haven't installed Tesseract OCR already.
5. Install all the supported libraries using pip install, libraries needed to be installed(if already not installed) are:
   5.1 win32com.client
   5.2 glob
   5.3 shutil
   5.4 os
   5.5 pdf2image
   5.6 cv2
   5.7 numpy
   5.8 PILLOW
   5.9 scipy
   5.10 collections
   5.11 matplotlib
   5.12 copy
   5.13 pytesseract
   5.14 xlsxwriter
6.Hope everything sets up poperly if not read the section below it will help.
7. Now add the Input Files(PDF, Images, Word Document) in 'INPUT FILES' folder, and run:
  7.1 MAIN for python 2.py file(if you have python 2 installed in your system) 
		or 
  7.2 MAIN for python 3.py file(if you have python 3 installed in your system). 
   Once code runs itself, you can see the Excel output in 'OUT FILES' folder. 



IF YOU HAVEN'T INSTALLED TESSERACT AND POPPLER BEFORE YOU NEED TO MAKE THESE CHANGES TO MAKE THE INSTALLATION FILE WORK.

FOR TESSERACT: 
If facing problem in using pytesseract then add/replace the following config (Line 28 in the pytesseract.py), if you have tessdata error like: “Error opening data file…” 
        
        #This should be the address where you have installed Tesseract OCR(default adress will be same as I have written in the next two lines)
	
        tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
  
	tessdata_dir_config = '--tessdata-dir r"C:\Program Files (x86)\Tesseract-OCR\tessdata"'

	You need to search the destination of your pip module installation and add these two lines to pytesseract.py
	My destination for pip installation is "C:\Users\Akash Singh\Anaconda3\Lib\site-packages\pytesseract\pytesseract.py", hope this could help you in finding the  file.


FOR POPPLER: 
If facing problem in using pdf2img regarding poppler, then add this path "C:\Axis AI Challenge @ Akash_Abhishek\poppler-0.68.0\bin\"  in your system variables. Use this link for more help on how to add system varaible path 'https://docs.telerik.com/teststudio/features/test-runners/add-path-environment-variables'. After path addition you need to restart your PC.


