import csv
import matplotlib.pyplot as plt
from tools import HeightMap, ShapeObject, GridSort
from matplotlib.path import Path
import numpy as np
import scipy.interpolate as sci

fig = plt.figure(figsize = (10,10))

y = HeightMap(['nn1000060000','nn3000080000'])
h = y.height_map
x = y.gradient_map
plt.imshow(h, extent = [210000,240000,760000,790000])

catch = ShapeObject('C:\\Users\\Murray\\Documents\\python\\River Code\\nrfa_data\\catchment_data\\SHP_files\\90003', region = [210000,230000,760000,780000])
catch.plot(fig, color='k')
info,x,y = catch.shapes[0]
plt.show()

vertices = np.zeros((len(x),2))
codes = [Path.MOVETO] + [Path.LINETO]*(len(x)-2) + [Path.CLOSEPOLY]
vertices[:,0], vertices[:,1] = x, y
path = Path(vertices, codes)

indices = np.zeros((600,600,2))
indice_list = []

for i in range(600):
    for j in range(600):
        indices[j,i,:] = [i,j]
        indice_list.append((i,j))
        
grid = GridSort(list(zip(x,y)), x_range=(210000,230000), y_range=(760000, 780000), grid_size=(600,600))
ci = list(zip(grid.contained_indexs.astype(int)[:,0],grid.contained_indexs.astype(int)[:,1]))

heights = []

for i in range(600):
    for j in range(600):
        if (i,j) in ci:
            heights.append(h[j,i])        

sh1000 = []
for i in range(1000):
    sh1000.append(heights[i*62])
    
yn = sci.spline(sh1000,np.linspace(0,1,1000),np.arange(1288))
height_ranges = [round(i, 2) for i in yn]
                 
with open('nevis_catchment_height_profile.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([height_ranges])