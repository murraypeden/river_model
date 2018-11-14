import numpy as np
import matplotlib.pyplot as plt
import urllib.request as url
import datetime as dt


levels = <list of levels>
times = <list of datetime objects>

n = len(levels)

#Define list of days of the week.
week = ['Monday',
        'Tuesday',
        'Wednesday', 
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday']

xlabels = []
x_minor_loc = []
x_major_loc = []

#Create list of minor tick locations on the hour and a list of major tick 
#locations on midnight. If the time is 6:00 or 18:00 add a label with the time 
#and if the time is 12:00 add the time and day of the week under it.  For all
#other minor ticks add a empty string.
for time in times:
    if time.minute == 0:
        x_minor_loc.append(times.index(time))
        if time.hour in [6, 18]:
            xlabels.append(dt.datetime.strftime(time, '%H:%M'))
        elif time.hour == 12:
            xlabels.append(dt.datetime.strftime(time, '%H:%M') 
                           + '\n' + week[time.weekday()])
        elif time.hour == 0:
            x_major_loc.append(times.index(time))
            xlabels.append('')
        else:
            xlabels.append('')

#Initiate figure and axis.
fig = plt.figure(figsize=(15,10))    
ax = fig.add_axes([0.1,0.1,0.8,0.8])

#Plot levels
ax.plot(levels, 'k')
ax.fill_between(np.arange(n), levels, color='lightblue')

#Set limits
ax.set_xlim([0,n])
ax.set_ylim([0,1.8])

#Add y label
ax.set_ylabel('River level. (m)', fontsize=14)  

#Add grid lines coming from the major ticks on the y axis.
ax.yaxis.grid(True, linestyle='-', alpha = 0.2)

#Create ticks and label the minor ones with the labels defined earlier.
#Parameters may be controlled using the ax.tick_params() method.
ax.set_xticks(x_major_loc)
ax.set_xticks(x_minor_loc, minor=True)
ax.set_xticklabels(xlabels, minor=True)
ax.tick_params(axis = 'x', length=5, width=1, which='minor', labelsize=12)
ax.tick_params(axis = 'x', length=30, width=1, which='major', labelsize=0)

#Repeat for y axis but just use a simple linspace for labels. (rounded cos I 
#getting floating point errors)
ax.set_yticks(np.linspace(0,1.8,10))
ax.set_yticklabels(round(i, 1) for i in np.linspace(0,1.8,10))
ax.set_yticks(np.linspace(0.1,1.7,9), minor=True)
ax.tick_params(axis = 'y', length=5, width=1, which='minor', labelsize=0)
ax.tick_params(axis = 'y', length=5, width=1, which='major', labelsize=12)

#Add plot title
ax.text(0, 1.85, 'River Nevis.', fontsize=15)

plt.show()