"pip install pymysql" 
import pymysql.cursors
import numpy as np
import math
import re
import datetime
from pathlib import Path
import cfg.cfg as cfg
connection = pymysql.connect(host=cfg.mysql_host,
                             user=cfg.mysql_user,
                             password=cfg.mysql_password,
                             port=3308,
                             db=cfg.mysql_db,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
f=["insert studtran (STUD_REF,GRADE,CLASS,C_NO) values "]
try:
    with connection.cursor() as cursor:
        sql="select STUD_REF from studmain;"
        cursor.execute(sql,())
        result=cursor.fetchall()
        for idx,row in enumerate(result):
            f.append(f"('{row["STUD_REF"]}','**','*',0)")
finally:
    connection.close()
p=Path('out.txt')
p.write_text(',\r\n'.join(f),encoding='utf8')