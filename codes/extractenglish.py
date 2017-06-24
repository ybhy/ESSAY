#-*- coding: utf-8 -*- 
import pymssql
import random
import re
import sys


file = open(r"F:\PAPER\Essay\experiment\testdoc\read6000.txt",'rU')
f = open(r"F:\PAPER\Essay\experiment\testdoc\ex6000.txt",'w')
for line in file:
        temp = line.split('\t')
        temp_title = temp[0]
        temp_abstract = temp[1]
        title = re.sub('[^a-zA-Z&&^\\t]',' ',temp_title)
        titles = re.sub(r'\s+', ' ',title)
        abstract = re.sub('[^a-zA-Z&&^\\t]',' ',temp_abstract)
        abstracts = re.sub(r'\s+', ' ',abstract)
##        print titles + '\t' + abstracts
        f.write(titles + '\t' + abstracts + '\n')
f.close( )
print 'ok'
count = len(open(r"F:\PAPER\Essay\experiment\testdoc\ex6000.txt",'rU').readlines())
print count

