import bs4
import urllib.request as url
import datetime as dt
import json
from secret import WEATHBIT_KEY


def get_YR(lon=-5.105218, lat=56.819817):
    """
       This function gets the latest rainfall forecast from the Norwegian 
       Metoffice. Default location is Fort William.
    """
    YR_URL = "https://api.met.no/weatherapi/locationforecast/1.9/?lat={0:0.6f}&lon={1:0.6f}&msl=0"
    
    XML = url.urlopen(YR_URL.format(lat,lon))
    soup = bs4.BeautifulSoup(XML, "html.parser")
    
    children = list(soup.weatherdata.product.children)
    
    times = []
    rainfall = []
    
    for child in children:
        try:
            start = dt.datetime.strptime(child['from'],'%Y-%m-%dT%H:%M:%SZ')
            stop = dt.datetime.strptime(child['to'],'%Y-%m-%dT%H:%M:%SZ')
            if str(stop-start) == '1:00:00':
                times.append(start)
                rainfall.append(float(child.precipitation['value']))
        except KeyError:
            continue
        except TypeError:
            continue
        
    return times, rainfall


def get_metcheck(lon=-5.105218, lat=56.819817):
    """
       This function gets the latest rainfall forecast from metcheck.
       Default location is Fort William.
    """
    MetCheck_URL = "http://ws1.metcheck.com/ENGINE/v9_0/json.asp?lat={0:0.1f}&lon={1:0.1f}&Fc=No"
    
    MC = json.load(url.urlopen(MetCheck_URL.format(lat,lon)))
    forecasts = MC['metcheckData']['forecastLocation']['forecast']
    
    times = []
    rainfall = []
    
    for forecast in forecasts:
        times.append(dt.datetime.strptime(forecast['utcTime'], '%Y-%m-%dT%H:%M:%S.%f'))
        rainfall.append(float(forecast['rain']))
        
    return times, rainfall


def get_weatherbit(lon=-5.105218, lat=56.819817):
    """
       This function gets the latest rainfall forecast from the weatherbit.
       Default location is Fort William.
    """
    WeathBit_URL = "https://api.weatherbit.io/v2.0/forecast/hourly?lat={0:0.6f}&lon={1:0.6f}&key=" + WEATHBIT_KEY
    
    WB = json.load(url.urlopen(WeathBit_URL.format(lat,lon)))
    
    times = []
    rainfall = []
    
    for forecast in WB['data']:
        times.append(dt.datetime.strptime(forecast['datetime'], '%Y-%m-%d:%H'))
        rainfall.append(forecast['precip'])
        
    return times, rainfall




