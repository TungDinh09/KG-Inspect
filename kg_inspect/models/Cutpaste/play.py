import os
x=[]
path = "/home/guest/Documents/TUNG/KG-Inspect/data/VisA"
for f in os.listdir(path):
    x.append(f)
print(x)
[ 'pcb4', 'cashew', 'pcb1', 'pcb2', 'pcb3', 'capsules', 'candle', 'fryum', 'macaroni1', 'pipe_fryum', 'macaroni2', 'chewinggum']