#-*- coding: utf-8 -*- 
#读取SQL，简单处理成语料库

import pymssql
import random
import re
import sys
      
conn = pymssql.connect(host = "localhost",user = "sa",password = "123456", database = "LW")
cur = conn.cursor()
sql = "SELECT top 1200 [title] ,[abstracts] FROM [LW].[dbo].[TXsignalprocessing]"
cur.execute(sql)
conn.commit
rows = cur.fetchall()
numrows = int(cur.rowcount)
print numrows

f = open('read6000.txt','a') 
for row in rows:
        temptitle = row[0].encode('utf-8').replace('\r\n', '').replace('\n', '').replace('\t', '')
        tempabstracts =  row[1].encode('utf-8').replace('\r\n', '').replace('\n', '').replace('\t', '')
        f.write(temptitle + '\t' + tempabstracts+ '\n' )
f.close( )
print "OK"
cur.close()
conn.close()
count = len(open(r"F:\PAPER\Essay\experiment\testdoc\read6000.txt",'rU').readlines())
print count
