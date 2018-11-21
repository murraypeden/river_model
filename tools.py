import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import zipfile
import shapefile
import scipy as sci

class GridSort():

    def __init__(self, data, x_range=(-12.0, 5),
                 y_range=(48.0, 61.0), grid_size=(500, 500)):
        self.xcords = np.linspace(x_range[0], x_range[1], grid_size[0])
        self.ycords = np.linspace(y_range[0], y_range[1], grid_size[1])
        temp = np.mgrid[0:grid_size[0]:complex(grid_size[0]),
                        0:grid_size[1]:complex(grid_size[1])]
        self.grid_x, self.grid_y = temp
        self._raw_points = data
        self._format_input()
        self._sort()
        self._contained_indexs()
        
    def _sort(self):
        self.indexs = np.int_(np.zeros((len(self.data_points), 2)))
        for j in range(len(self.data_points)):
            point = self.data_points[j]

            self.x_index = 100000000
            for i in range(len(self.xcords)):
                if abs(point[0]-self.xcords[i]) < self.x_index:
                    self.x_index = abs(point[0]-self.xcords[i])
                    x_match = i

            self.y_index = 100000000
            for i in range(len(self.ycords)):
                if abs(point[1]-self.ycords[(len(self.xcords)-1)-i]) < self.y_index:
                    self.y_index = abs(point[1]-self.ycords[(len(self.xcords)-1)-i])
                    y_match = i

            self.indexs[j, 0] = int(x_match)
            self.indexs[j, 1] = int(y_match)
    
    def _contained_indexs(self):
        codes = [Path.MOVETO] + [Path.LINETO]*(np.shape(self.indexs)[0]-2) + [Path.CLOSEPOLY]
        vertices = self.indexs
        path = Path(vertices, codes)
        self._contained_xindexs = []
        self._contained_yindexs = []

        for x in range(len(self.xcords)):
            for y in range(len(self.ycords)):
                if path.contains_point((x,y)):
                    self._contained_xindexs.append(int(x))
                    self._contained_yindexs.append(int(y))
        
        self.contained_indexs = np.zeros((len(self._contained_yindexs),2), dtype=np.int32)
        self.contained_indexs[:,0] = self._contained_xindexs
        self.contained_indexs[:,1] = self._contained_yindexs
            
    def _format_input(self):
        if type(self._raw_points) == np.ndarray:
            if np.shape(self._raw_points)[1] == 2:
                self.data_points = self._raw_points
            else:
                raise ValueError('Invalid shape.')
        if type(self._raw_points) == list:
            for item in self._raw_points:
                if len(item) != 2:
                    message = 'List elements must be paired co-ordinates.'
                    raise ValueError(message)
            self.data_points = np.array(self._raw_points)
        else:
            self.data_points = self._raw_points


class RadarImage():

    def __init__(self, image):
        self.colour_dict = {'#0x00x00x0':    0,   '#0x00x00xfe':  0.01,
                            '#0x320x650xfe': 0.5, '#0x7f0x7f0x0': 1,
                            '#0xfe0xcb0x0':  2,   '#0xfe0x980x0': 4,
                            '#0xfe0x00x0':   8,   '#0xfe0x00xfe': 16,
                            '#0xe50xfe0xfe': 32}
        self.__input = image
        self.__format_input()

    def __format_input(self):
        if type(self.__input) == np.ndarray:
            shape = np.shape(self.__input)[2]
            if shape == 3 or shape == 4:
                self.RGBA_array = self.__input
                self.__convert_to_hex()
        else:
            raise ValueError('Invalid input.')

    def __convert_to_hex(self):
        self.rain_array = np.zeros(np.shape(self.__input)[:2])
        for x in range(np.shape(self.RGBA_array)[1]):
            for y in range(np.shape(self.RGBA_array)[0]):
                if self.RGBA_array[y, x, 3] == 255:
                    rgb = self.RGBA_array[y, x, :3]
                    HEX = '#' + hex(rgb[0]) + hex(rgb[1]) + hex(rgb[2])
                    self.rain_array[y, x] = self.colour_dict[HEX]


class HeightMap():
    
    def __init__(self, map_range = None, gradients = False):
        self.map_range = map_range
        self.square_name, self.square_numbers = self.__select_squares()
        self.height_map = self.__format_data()
        self.size = np.shape(self.height_map)[0]
        self.gradient_map = HeightMap.gradient_map(self.height_map, generate = gradients)
        
    def load_data(name):
        try:
            with zipfile.ZipFile('nrfa_data/grid_data/' + name[0:2] 
                                 + '/' + name + '_OST50GRID_20160726.zip'
                                 ) as myzip:
                with myzip.open(name[0:2].upper() + name[2:4] + '.asc', 'r') as f:
                    raw_data = []
                    data = []
                    for line in f:
                        raw_data.append(str(line).strip("'b").strip('\\r\\n').split(' '))
                    for item in raw_data[5:]:
                        data.append([float(i) for i in item])
                    return np.array(data)
        except FileNotFoundError:
            return np.zeros((200,200))
            
    def __select_squares(self):
        if type(self.map_range) == None:
            raise NameError('No selected squares.')
        else:
            ranges = Coordinate(self.map_range, 'alpha_os')
            limit_squares = ranges.grid10km_num
            
            E_range = int(limit_squares[1][0:2]) - int(limit_squares[0][0:2])
            N_range = int(limit_squares[1][2:4]) - int(limit_squares[0][2:4])
            squares = []
        
            for E in range(E_range+1):
                for N in range(N_range+1):
                    squares.append((int(str(int(limit_squares[0][0:2]) + E) + '0000'), 
                                    int(str(int(limit_squares[0][2:4]) + N) + '0000')))
        
            square_num = Coordinate(np.array(squares), 'num_os')
            square_names = square_num.grid10km_alpha
            square_numbers = square_num.grid10km_num           
            
            return square_names, square_numbers
            
    def __format_data(self):
        if type(self.square_name) == list:
            arrays = {}            
            for name, number in zip(self.square_name, self.square_numbers):
                if number[0:2] not in arrays: 
                    arrays[number[0:2]] = {number[2:4]:HeightMap.load_data(name)}
                else:
                    arrays[number[0:2]][number[2:4]] = HeightMap.load_data(name)
            xnames = list(arrays.keys())
            temp_row = []
            xnames.sort()
        
            for column in xnames:
                ynames = list(arrays[column].keys())
                temp_col= []
                ynames.sort()
                for row in ynames:
                    temp_col.append(arrays[column][row])
                temp_col.reverse()
                temp_row.append(np.vstack(temp_col))
            data = np.hstack(temp_row)
            return data
            
        else:
            data = HeightMap.load_data(self.square_name)
            return data
        
    def gradient_map(data, generate = False):
        if generate:
            grid = data
            l1 = np.shape(data)[0]
            l2 = np.shape(data)[0]
            gradients = np.zeros((l1,l2,3))
            
            for x in range(1,l2-1):
                for y in range(1,l1-1):
                    suround = [grid[y-1,x-1], grid[y-1,x], grid[y-1,x+1], grid[y,x-1],
                               grid[y,x+1], grid[y+1,x-1], grid[y+1,x], grid[y+1,x+1]]
                    
                    up = suround.index(min(suround))
                    
                    if grid[y,x] < min(suround):
                        gradients[y,x,:] = np.array([0,0,0])
                    elif up == 0:
                        gradients[y,x,:] = np.array([-1,-1,20*(grid[y,x]- suround[up])])/np.sqrt(2)
                    elif up == 1:
                        gradients[y,x,:] = np.array([0,-1,20*(grid[y,x] - suround[up])])
                    elif up == 2:
                        gradients[y,x,:] = np.array([1,-1,20*(grid[y,x] - suround[up])])/np.sqrt(2)
                    elif up == 3:
                        gradients[y,x,:] = np.array([-1,0,20*(grid[y,x] - suround[up])])
                    elif up == 4:
                        gradients[y,x,:] = np.array([1,0,20*(grid[y,x] - suround[up])])
                    elif up == 5:            
                        gradients[y,x,:] = np.array([-1,1,20*(grid[y,x] - suround[up])])/np.sqrt(2)
                    elif up == 6:
                        gradients[y,x,:] = np.array([0,1,20*(grid[y,x] - suround[up])])
                    elif up == 7:
                        gradients[y,x,:] = np.array([1,1,20*(grid[y,x] - suround[up])])/np.sqrt(2)
            
            return gradients
    
    def colorplot(self, fig, **kwargs):
        coords = Coordinate(self.map_range, 'alpha_os')
        num_coords = coords.num_os
        
        plt.xlim(num_coords[0,0], num_coords[1,0])
        plt.ylim(num_coords[0,1], num_coords[1,1])
        
        plt.imshow(self.height_map, 
                   extent = [num_coords[0,0] - num_coords[0,0] % 10000, 
                             num_coords[1,0] - num_coords[1,0] % 10000 + 10000, 
                             num_coords[0,1] - num_coords[0,1] % 10000, 
                             num_coords[1,1] - num_coords[1,1] % 10000 + 10000],
                   **kwargs)
        
    def contourplot(self, fig, contours = 10, **kwargs):
        coords = Coordinate(self.map_range, 'alpha_os')
        num_coords = coords.num_os
        
        plt.xlim(num_coords[0,0], num_coords[1,0])
        plt.ylim(num_coords[0,1], num_coords[1,1])
        
        plt.contour(np.linspace(num_coords[0,0] - num_coords[0,0] % 10000, 
                                num_coords[1,0] - num_coords[1,0] % 10000 + 10000, 
                                np.shape(self.height_map)[1]), 
                    np.linspace(num_coords[0,1] - num_coords[0,1] % 10000, 
                                num_coords[1,1] - num_coords[1,1] % 10000 + 10000, 
                                np.shape(self.height_map)[0]), 
                    np.flipud(self.height_map),
                    contours, 
                    **kwargs)

class ShapeObject():
    
    def __init__(self, filename, fill=False, region=[0,1000000,0,1000000]):
        self.filename = filename
        self._fill = fill
        self._region = region
        self.shapes = ShapeObject.load_data(filename, region=self._region)
        
    def load_data(filename, region=[0,1000000,0,1000000]):
        rd = shapefile.Reader(filename)
        shapes = rd.shapes()
        records = rd.records()
        
        points = []

        for record, shape in zip(records,shapes):
            x,y = zip(*shape.points)
            lon = np.array(x)
            lat = np.array(y)
            
            cond1 = (np.array(lon) >= region[0]).any()
            cond2 = (np.array(lon) <= region[1]).any()
            cond3 = (np.array(lat) >= region[2]).any()
            cond4 = (np.array(lat) <= region[3]).any()
        
            if cond1 and cond2 and cond3 and cond4:
                points.append([record,lon,lat])
                
        return points
        
    def plot(self, fig, **kwargs):
        for item in self.shapes:
            x, y = item[1], item[2]
            
            plt.xlim(self._region[0],self._region[1])
            plt.ylim(self._region[2],self._region[3])
            
            if self._fill:
                vertices = np.zeros((len(x),2))
                codes = [Path.MOVETO] + [Path.LINETO]*(len(x)-2) + [Path.CLOSEPOLY]
                vertices[:,0], vertices[:,1] = x, y
                path = Path(vertices, codes)
                
                ax = fig.add_subplot(111)
                patch = patches.PathPatch(path, **kwargs)
                ax.add_patch(patch)
                
            else:
                plt.plot(x, y, **kwargs)


class Coordinate():
    
    style_list = ['num_os', 'alpha_os', 'latlon']
    
    def __init__(self, points, style):
        self._raw_points = points
        self.style = style
        self.__format_coords()
        self.__convert()

    def __format_coords(self):
        if self.style == 'num_os':
            if type(self._raw_points) == np.ndarray and np.shape(self._raw_points)[1] == 2:
                self.coords = self._raw_points
            elif type(self._raw_points) == list:
                self.coords = np.array(self._raw_points)
            else:
                raise ValueError('Invalid input')
        elif self.style == 'alpha_os':
            if type(self._raw_points) == list:
                self.coords = self._raw_points
            else:
                raise ValueError('Invalid input')
        elif self.style == 'latlon':
            if type(self._raw_points) == np.ndarray and np.shape(self._raw_points)[1] == 2:
                self.coords = self._raw_points
            elif type(self._raw_points) == list:
                self.coords = np.array(self._raw_points)
            else:
                raise ValueError('Invalid input')
        else:
            raise ValueError('Invalid coordinate system type.')
    
    def __convert(self):
        if self.style == 'num_os':
            self.num_os = self.coords
            self.alpha_os = Coordinate.NumtoAlphaOS(self.coords)
            self.latlon = Coordinate.OStoWGS84(self.num_os)
        if self.style == 'alpha_os':
            self.alpha_os = self.coords
            self.num_os = Coordinate.AlphatoNumOS(self.coords)
            self.latlon = Coordinate.OStoWGS84(self.num_os)
        if self.style == 'latlon':
            self.latlon = self.coords
            self.num_os = Coordinate.WGS84toOS(self.latlon)
            self.alpha_os = Coordinate.NumtoAlphaOS(self.num_os)
            
        self.grid10km_alpha = [item[0:3] + item[7] for item in self.alpha_os]
        self.grid10km_num = []
        for item in self.num_os:
            self.grid10km_num.append('{:0>2n}'.format(int(item[0]/10000)) + '{:0>2n}'.format(int(item[1]/10000)))
        
    def NumtoAlphaOS(points):
        grid100 = {(1, 2): 'm', (3, 2): 'o', (0, 0): 'v', (3, 0): 'y', 
                   (2, 2): 'n', (1, 4): 'b', (1, 3): 'g', (2, 3): 'h', 
                   (2, 1): 's', (4, 2): 'p', (1, 0): 'w', (0, 3): 'f', 
                   (4, 0): 'z', (0, 1): 'q', (0, 2): 'l', (3, 3): 'j', 
                   (3, 4): 'd', (3, 1): 't', (4, 4): 'e', (2, 4): 'c', 
                   (2, 0): 'x', (4, 3): 'k', (0, 4): 'a', (4, 1): 'u', 
                   (1, 1): 'r'}
        grid500 = {(0, 0): 's', (5, 0): 't', (0, 5): 'n', (5, 5): 'o'}
        
        converted_points = []

        for i in range(len(points[:,0])):  
            numE = int((points[i,0] - points[i,0] %100000)/100000)
            numN = int((points[i,1] - points[i,1] %100000)/100000)            
            key100 = (numE % 5, numN % 5)            
            key500 = (5*int(numE/5), 5*int(numN/5))
            string = '{}{}{:0>5}{:0>5}'.format(grid500[key500], 
                                               grid100[key100],
                                               str(int(points[i,0] %100000)),
                                               str(int(points[i,1] %100000)))
            converted_points.append(string)
        
        return converted_points
             
    def AlphatoNumOS(points):
        coords = np.zeros((len(points),2))
        for k in range(len(points)):
            al = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
            
            N = {}
            for i in range(5):
                for j in range(5):
                    N[al[i*5+j]] = (j,9-i)
            S = {}
            for i in range(5):
                for j in range(5):
                    S[al[i*5+j]] = (j,4-i)
            O = {}
            for i in range(5):
                for j in range(5):
                    O[al[i*5+j]] = (5+j,9-i)
            T = {}
            for i in range(5):
                for j in range(5):
                    T[al[i*5+j]] = (5+j,4-i)
            
            dic = {'N':N,'S':S,'O':O,'T':T}
    
            r = dic[points[k][0].upper()][points[k][1].upper()]
            coords[k,:] = [int(str(r[0])+points[k][2:7]), int(str(r[1])+points[k][7:12])]
        
        return coords
        
    def WGS84toOS(points):
        coords = np.zeros(np.shape(points))
        for i in range(np.shape(points)[0]):
            if points[i,0] < 0:
                lon = points[i,0] + 360
            else:
                lon = points[i,0]
            if points[i,1] < 0:
                lat = points[i,1] + 360
            else:
                lat = points[i,1]
            lat_1, lon_1 = lat*sci.pi/180, lon*sci.pi/180
    
            a_1, b_1 =6378137.000, 6356752.3141
            e2_1 = 1- (b_1*b_1)/(a_1*a_1)
            nu_1 = a_1/sci.sqrt(1-e2_1*sci.sin(lat_1)**2)
            
            H = 0 
            x_1 = (nu_1 + H)*sci.cos(lat_1)*sci.cos(lon_1)
            y_1 = (nu_1+ H)*sci.cos(lat_1)*sci.sin(lon_1)
            z_1 = ((1-e2_1)*nu_1 +H)*sci.sin(lat_1)
    
            s = 20.4894*10**-6
            tx, ty, tz = -446.448, 125.157, -542.060
            rxs,rys,rzs = -0.1502, -0.2470, -0.8421
            rx, ry, rz = rxs*sci.pi/(180*3600.), rys*sci.pi/(180*3600.), rzs*sci.pi/(180*3600.)
            x_2 = tx + (1+s)*x_1 + (-rz)*y_1 + (ry)*z_1
            y_2 = ty + (rz)*x_1+ (1+s)*y_1 + (-rx)*z_1
            z_2 = tz + (-ry)*x_1 + (rx)*y_1 +(1+s)*z_1
    
            a, b = 6377563.396, 6356256.909
            e2 = 1- (b*b)/(a*a)
            p = sci.sqrt(x_2**2 + y_2**2)
    
            lat = sci.arctan2(z_2,(p*(1-e2)))
            latold = 2*sci.pi
            while abs(lat - latold)>10**-16:
                lat, latold = latold, lat
                nu = a/sci.sqrt(1-e2*sci.sin(latold)**2)
                lat = sci.arctan2(z_2+e2*nu*sci.sin(latold), p)
    
            lon = sci.arctan2(y_2,x_2)
            H = p/sci.cos(lat) - nu
    
            F0 = 0.9996012717
            lat0 = 49*sci.pi/180
            lon0 = -2*sci.pi/180
            N0, E0 = -100000, 400000
            n = (a-b)/(a+b)
    
            rho = a*F0*(1-e2)*(1-e2*sci.sin(lat)**2)**(-1.5)
            eta2 = nu*F0/rho-1
            
            M1 = (1 + n + (5/4)*n**2 + (5/4)*n**3) * (lat-lat0)
            M2 = (3*n + 3*n**2 + (21/8)*n**3) * sci.sin(lat-lat0) * sci.cos(lat+lat0)
            M3 = ((15/8)*n**2 + (15/8)*n**3) * sci.sin(2*(lat-lat0)) * sci.cos(2*(lat+lat0))
            M4 = (35/24)*n**3 * sci.sin(3*(lat-lat0)) * sci.cos(3*(lat+lat0))
    
            M = b * F0 * (M1 - M2 + M3 - M4)
            
            I = M + N0
            II = nu*F0*sci.sin(lat)*sci.cos(lat)/2
            III = nu*F0*sci.sin(lat)*sci.cos(lat)**3*(5- sci.tan(lat)**2 + 9*eta2)/24
            IIIA = nu*F0*sci.sin(lat)*sci.cos(lat)**5*(61- 58*sci.tan(lat)**2 + sci.tan(lat)**4)/720
            IV = nu*F0*sci.cos(lat)
            V = nu*F0*sci.cos(lat)**3*(nu/rho - sci.tan(lat)**2)/6
            VI = nu*F0*sci.cos(lat)**5*(5 - 18* sci.tan(lat)**2 + sci.tan(lat)**4 + 14*eta2 - 58*eta2*sci.tan(lat)**2)/120
            
            N = I + II*(lon-lon0)**2 + III*(lon- lon0)**4 + IIIA*(lon-lon0)**6
            E = E0 + IV*(lon-lon0) + V*(lon- lon0)**3 + VI*(lon- lon0)**5 
            
            coords[i,:] = [int(E), int(N)]
        
        return coords
        
    def OStoWGS84(points):
        coords = np.zeros(np.shape(points))
        for i in range(np.shape(points)[0]):
            E, N = int(points[i,0]), int(points[i,1])

            a, b = 6377563.396, 6356256.909
            F0 = 0.9996012717
            lat0 = 49*sci.pi/180
            lon0 = -2*sci.pi/180
            N0, E0 = -100000, 400000
            e2 = 1 - (b*b)/(a*a)
            n = (a-b)/(a+b)
            lat,M = lat0, 0
            
            while N-N0-M >= 0.00001:
                lat = (N-N0-M)/(a*F0) + lat;
                M1 = (1 + n + (5/4)*n**2 + (5/4)*n**3) * (lat-lat0)
                M2 = (3*n + 3*n**2 + (21/8)*n**3) * sci.sin(lat-lat0) * sci.cos(lat+lat0)
                M3 = ((15/8)*n**2 + (15/8)*n**3) * sci.sin(2*(lat-lat0)) * sci.cos(2*(lat+lat0))
                M4 = (35/24)*n**3 * sci.sin(3*(lat-lat0)) * sci.cos(3*(lat+lat0))
                M = b * F0 * (M1 - M2 + M3 - M4)

            nu = a*F0/sci.sqrt(1-e2*sci.sin(lat)**2)
            rho = a*F0*(1-e2)*(1-e2*sci.sin(lat)**2)**(-1.5)
            eta2 = nu/rho-1
            
            secLat = 1./sci.cos(lat)
            VII = sci.tan(lat)/(2*rho*nu)
            VIII = sci.tan(lat)/(24*rho*nu**3)*(5+3*sci.tan(lat)**2+eta2-9*sci.tan(lat)**2*eta2)
            IX = sci.tan(lat)/(720*rho*nu**5)*(61+90*sci.tan(lat)**2+45*sci.tan(lat)**4)
            X = secLat/nu
            XI = secLat/(6*nu**3)*(nu/rho+2*sci.tan(lat)**2)
            XII = secLat/(120*nu**5)*(5+28*sci.tan(lat)**2+24*sci.tan(lat)**4)
            XIIA = secLat/(5040*nu**7)*(61+662*sci.tan(lat)**2+1320*sci.tan(lat)**4+720*sci.tan(lat)**6)
            dE = E-E0
            
            lat = lat - VII*dE**2 + VIII*dE**4 - IX*dE**6
            lon = lon0 + X*dE - XI*dE**3 + XII*dE**5 - XIIA*dE**7

            lat = lat*180/sci.pi
            lon = lon*180/sci.pi
            
            coords[i,:] = [lon, lat]
        
        return coords
    

        
        
        
        
        
        
