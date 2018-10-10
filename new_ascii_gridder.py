import numpy as np
import bs4
import urllib.request as url
import imageio
import os
import datetime
import traceback
import shapefile as shp
import matplotlib.path as pth


def update():
    
    path = os.getcwd() + '\\rainfall_images\\'
    key = "598ca851-5617-4142-a187-7593fb22ea38"
    capabilities_url = "http://datapoint.metoffice.gov.uk/public/data/inspire/view/wmts?REQUEST=getcapabilities&key=" + key 
    request_url = "http://datapoint.metoffice.gov.uk/public/data/inspire/view/wmts?REQUEST=gettile&LAYER=RADAR_UK_Composite_Highres&FORMAT=image/png&TILEMATRIXSET=EPSG:27700&TILEMATRIX=EPSG:27700:{0}&TILEROW={1}&TILECOL={2}&TIME={3}&STYLE=Bitmap%20Interpolated%201km%20Blue-Pale%20blue%20gradient%200.25%20to%2032mm%2Fhr&key=" + key
    header = 'ncols 768\nnrows 1024\nxllcorner 1393.0196\nyllcorner 597350.76252\ncellsize 618.09012\nNODATA_value -9999\n'

    XML = url.urlopen(capabilities_url)
    soup = bs4.BeautifulSoup(XML, "html.parser")
    times = [time.text for time in soup.find_all('dimension')[0].find_all('value')]
    existing_files = [item for item in os.listdir(path) if item.split('.')[-1] == '.gz']

    colour_dict =  {193 : 0.,    0 : 0., 28 : 0.06, 103 :  0.13, 112 :  0.25, 
                    195 : 0.5, 165 : 1., 75 : 2.,   104 :  4.,   246 :  8.}
                    
    def im_to_rain(item):
        return colour_dict[item]  
    
    for time in times[0:1]:
        filename = 'radar_image_zoom_2_{0}'.format(str(time).replace(':','-'))
        extention = '.grd.gz'
        if filename + extention not in existing_files:
            try:
                ims = [[0] * 3 for x in range(4)]
                for row in range(4):
                    for column in range(3):
                        ims[row][column] = imageio.imread(url.urlopen(request_url.format(2, row, column, time)).read(), pilmode='L').astype(int)
                
                image = np.vstack([np.hstack(ims[i]) for i in range(4)])
                
                grid = np.array(list(map(im_to_rain, image.ravel()))).reshape(1024,768)
                np.savetxt(path + filename + extention, grid, fmt='%0.2g', comments='', header=header)
            
            except Exception as E:
                with open(path + 'error_report_{}.txt'.format(filename), 'w') as f:
                    f.write('Error creating file {0}'.format(filename + extention))
                    f.write('Error occured at {}.\n\n'.format(str(datetime.datetime.now())))
                    f.write(traceback.format_exc())
        

def get(filename):
    return np.loadtxt(filename, skiprows=7)


def create_coord_grid(mode='ll'):
    
    mode_dict = {'ll':(0,0),
                 'lr':(1,0), 
                 'ul':(0,1),
                 'ur':(1,1),
                 'c':(0.5,0.5)}
    
    lon = 1393.0196 + 618.09012 * np.arange(768) + mode_dict[mode][0]
    lat = 597350.76252 + 618.09012 * np.arange(1024) + mode_dict[mode][1]
    return np.meshgrid(lon, lat[::-1])


def contained_indexes(filename, lon, lat):
    
    data = shp.Reader(filename)
    shapes = data.shapes()
    points = np.array(shapes[0].points, dtype=int)
    path = pth.Path(points)
    
    contained_indexs = []
    
    for x in range(768):
        for y in range(1024):
            if path.contains_point((lon[y,x],lat[y,x])):
                contained_indexs.append((x,y))
                
    return np.array(contained_indexs)

    
    
    
    
    
    
    
    
    
    
    
    
    
