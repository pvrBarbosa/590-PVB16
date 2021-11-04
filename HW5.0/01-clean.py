# -*- coding: utf-8 -*-
"""
@author: pvict
"""

import sys
import os
import pandas as pd
import numpy as np
import re

in_path = "raw_input"
out_path = "clean_input"


x = []
y = []

for filename in os.listdir(in_path):
    with open(os.path.join(in_path, filename), 'r', encoding="utf-8") as f: 
        full_text = f.readlines()
       
        # Extract the book name from the first row
        book_name = re.sub("﻿The Project Gutenberg eBook of ","", full_text[0])
        book_name = re.sub(r', by.*', '', book_name)
        book_name = re.sub(r"[^a-zA-Z ]+", "", book_name)
        print("Book name:", book_name)
        
        
        # Initialize some variables
        start = -1
        end = -1
        book_body = []
        paragraph=""
        n_lines = 0
        
        # Loop the entire book and keep only the paragraphs
        for l in full_text:

            #Create a new list where avery paragraph is an item
            if start != -1 and end == -1:
                
                 
                # Add the paragraph if it has more than 3 rows and the loop finds
                # an empty line
                if l == "\n" and n_lines > 3:
                    book_body.append(re.sub(r"[^a-zA-Z ]+", "", paragraph).lower())
                    paragraph=""
                
                # Reset the paragraph every time the loop ecounters an empty line
                if l == "\n":
                    n_lines = 0
                    paragraph = ""
                   
                # If the line is not empty add it to the current paragraph
                if l != "\n" and re.sub(r"[^a-zA-Z]+","",l.lower())[:7] != "chapter":
                    n_lines = n_lines + 1
                    paragraph = paragraph + re.sub(r"\n", " ", l).lstrip()
               
            # Make sure we collect only the book body
            if start == -1: start = l.find("START OF THE PROJECT GUTENBERG")
            if end == -1: end = l.find("END OF THE PROJECT GUTENBERG")
               
           
        #append current book the to fnal dataframe
        x = x + book_body
        y = y + [book_name]*len(book_body)
               
        #Save the final dataframe
        df = pd.DataFrame(list(zip(x, y)))
        df.to_csv(out_path + "/clean_gutenberg.csv", header = False, index = False)
















