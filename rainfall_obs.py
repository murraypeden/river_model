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
    """
       Calling update() will automatically download the Metoffice INSPIRE 
       rainfall radar images for scotland at a zoom level of 2 and store them 
       in the "rainfall_images" directory as .grd.gz files.  An example of the
       file name is: radar_image_zoom_2_2018-10-08T00-00-00Z.grd.gz 
        
       If the "rainfall_images" directory does not exist one will be created.
       
       If a file already exists for a given time no images will be downloaded
       for that time.
        
       If a error occurs while downloading a file an error report will be
       written with the nature of the error and the program will move on to 
       the next file.
       
       Ascii grid has the format:
           
           ncols 768
           nrows 1024
           xllcorner 1393.0196
           yllcorner 597350.76252
           cellsize 618.09012
           NODATA_value -9999

           0 0 0 . . .
           0 0 0 
           0 0 0 
           .     .
           .       .
           .         .
           
        The reason the code is so compact and conveluted is because it has been
        time optimised!!  Please forgive me for the spagettii code...
    """
    
    #Useful strings.
    path = os.getcwd() + '\\rainfall_images\\'
    key = "598ca851-5617-4142-a187-7593fb22ea38"
    capabilities_url = "http://datapoint.metoffice.gov.uk/public/data/inspire/view/wmts?REQUEST=getcapabilities&key=" + key 
    request_url = "http://datapoint.metoffice.gov.uk/public/data/inspire/view/wmts?REQUEST=gettile&LAYER=RADAR_UK_Composite_Highres&FORMAT=image/png&TILEMATRIXSET=EPSG:27700&TILEMATRIX=EPSG:27700:{0}&TILEROW={1}&TILECOL={2}&TIME={3}&STYLE=Bitmap%20Interpolated%201km%20Blue-Pale%20blue%20gradient%200.25%20to%2032mm%2Fhr&key=" + key
    header = 'ncols 768\nnrows 1024\nxllcorner 1393.0196\nyllcorner 597350.76252\ncellsize 618.09012\nNODATA_value -9999\n'
    
    #If no folder exists create one.
    if 'rainfall_images' not in list(os.walk(os.getcwd()))[0][1]:
        os.makedirs('rainfall_images')
    
    #Create a list of times and existing images.    
    XML = url.urlopen(capabilities_url)
    soup = bs4.BeautifulSoup(XML, "html.parser")
    times = [time.text for time in soup.find_all('dimension')[0].find_all('value')]
    existing_files = [item for item in os.listdir(path) if item.split('.')[-1] == '.gz']
    
    #Dict for converting 8bit greyscale radar image to 15 rainfall total.
    colour_dict =  {193 : 0.,    0 : 0., 28 : 0.06, 103 :  0.13, 112 :  0.25, 
                    195 : 0.5, 165 : 1., 75 : 2.,   104 :  4.,   246 :  8.}
                    
    def im_to_rain(item):
        return colour_dict[item]  
    
    for time in times:
        filename = 'radar_image_zoom_2_{0}'.format(str(time).replace(':','-'))
        extention = '.grd.gz'
        if filename + extention not in existing_files:
            try:
                ims = [[0] * 3 for x in range(4)]
                for row in range(4):
                    for column in range(3):
                        #Create a matrix of greyscale radar tiles
                        ims[row][column] = imageio.imread(url.urlopen(request_url.format(2, row, column, time)).read(), pilmode='L').astype(int)
                
                image = np.vstack([np.hstack(ims[i]) for i in range(4)])
                
                #Convert greyscale radar image to 15 min rainfall totals and save as an ascii grid.
                grid = np.array(list(map(im_to_rain, image.ravel()))).reshape(1024,768)
                np.savetxt(path + filename + extention, grid, fmt='%0.2g', comments='', header=header)
            
            except Exception as E:
                #Write error report.
                with open(path + 'error_report_{}.txt'.format(filename), 'w') as f:
                    f.write('Error creating file {0}'.format(filename + extention))
                    f.write('Error occured at {}.\n\n'.format(str(datetime.datetime.now())))
                    f.write(traceback.format_exc())
        

def get(filename):
    """
       get(filename) returns the rainfall radar image of a given filename.
    """
    return np.loadtxt(filename, skiprows=7)


def create_coord_grid(mode='ll'):
    """
       create_coord_grid() returns a np.meshgrid of the coordinates of an image
       with the option to define which part of a cell the coordinate should 
       describe.
    """
    mode_dict = {'ll':(0,0),    #Lower left
                 'lr':(1,0),    #Lower right
                 'ul':(0,1),    #Upper left
                 'ur':(1,1),    #Upper right
                 'c':(0.5,0.5)} #Center
    
    lon = 1393.0196 + 618.09012 * np.arange(768) + mode_dict[mode][0]
    lat = 597350.76252 + 618.09012 * np.arange(1024) + mode_dict[mode][1]
    return np.meshgrid(lon, lat[::-1])


def contained_indexes(filename, lon, lat):
    """
       contained_indexes returns a numpy array containing the indexes of a 
       radar image contained within a shapefile.
       
       Catchment shapefiles for uk rivers may be found on:
           https://nrfa.ceh.ac.uk/
    """
    #create a matplotlib.path of the catchment
    data = shp.Reader(filename)
    shapes = data.shapes()
    points = np.array(shapes[0].points, dtype=int)
    path = pth.Path(points)
    
    contained_indexs = []
    
    #For all lat/lon combinations check if the point is in the catchment.
    for x in range(768):
        for y in range(1024):
            if path.contains_point((lon[y,x],lat[y,x])):
                contained_indexs.append((x,y))
                
    return np.array(contained_indexs)

    
    
    
    
    
    
    
    
    
    
    
    
    
