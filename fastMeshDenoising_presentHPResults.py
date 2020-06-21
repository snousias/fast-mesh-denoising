import tensorflow as tf
import numpy as np
import json
import seaborn as sns
import pandas as pd
import fastMeshDenoising_Config_Train as conf
import matplotlib.pyplot as plt
import matplotlib.colors as cl
jsonFile1='F:/_Groundwork/FastMeshDenoisingProduction/results/hopresults.json'  #_1
jsonFile2='F:/_Groundwork/FastMeshDenoisingProduction/results/hopresults_1.json'
jsonFile3='F:/_Groundwork/FastMeshDenoisingProduction/results/hopresults_2.json'
with open(jsonFile1, 'r') as f:
    jsonData1 = json.load(f)
with open(jsonFile2, 'r') as f:
    jsonData2 = json.load(f)
with open(jsonFile3, 'r') as f:
    jsonData3 = json.load(f)
jsonData=jsonData1
jsonData.extend(jsonData2)
jsonData.extend(jsonData3)
losses=[]
for row in jsonData:
    losses.append((row["loss"]))
    jsonData[jsonData.index(row)]['minLoss']=np.amin(np.asarray(row["loss"]))
_losses=np.asarray(losses)
minVal=1000000
minValRow=0
for i in losses:
    for j in i:
        if j < minVal:
            minVal=j
            minValRow=losses.index(i)

# Get the indices of minimum element in numpy array
print(jsonData[minValRow])



from matplotlib.colors import hsv_to_rgb
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

M=pd.DataFrame.from_dict(jsonData)


M2 = M[['BS', 'CL', 'D1','D2', 'E1', 'E2','KP','LR','minLoss']].copy()

M2.columns = ['Batch size','Cluster size','Decoding Layer 1',
                     'Decoding Layer 2','Encoding Layer 1','Encoding Layer 2','Keep Ratio','Learning Rate','minLoss']


M2=M2.sort_values(by=['minLoss'],ascending=False)
minVal=np.min(M2['minLoss'])
maxVal=np.max(M2['minLoss'])




# Set data
# number of variable
categories = M2.columns.drop('minLoss')
maxcat=np.zeros(np.shape(categories))
N = len(categories)

for index,cat in enumerate(categories):
    maxcat[index]=M2[cat].max()
    if (M2[cat].max() - M2[cat] .min())==0:
        M2[cat]=np.repeat(0.5,len(M2[cat]))
    else:
        M2[cat]=(M2[cat]) / (M2[cat].max())
    print('ok')



fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, polar=True)
ax.grid(True)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.rc('axes', axisbelow=True)
[line.set_zorder(30) for line in ax.lines]
tVals=[]

for index, row in M2.iterrows():
    stats=M2.loc[index,categories].values
    angles=np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    # close the plot
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    val=((256/360)*((M2.at[index, 'minLoss']-minVal)/(maxVal-minVal)))


    vc=128/360
    newval=(-(val-vc))+vc
    lognewval=np.log(100*val+0.5)

    logval = np.log(100*val+0.5)
    tVals.append(logval)
    logval=np.clip(logval,0,1)
    HSV=np.asarray([logval,1,1])
    RGB=hsv_to_rgb(HSV)
    print(HSV)
    print(RGB)
    ax.plot(angles, stats, '-', linewidth=2,color=RGB,zorder=(2.5+newval))
    if index==(M2.shape[0]-1):
        ax.fill(angles, stats, alpha=0.55,color=RGB,zorder=(2.5+newval))


#ax.set_thetagrids(angles * 180/np.pi, categories)
# legend = ax.legend(categories, loc=(0.9, .95),
#                        labelspacing=0.1, fontsize='small')
# fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
#              horizontalalignment='center', color='black', weight='normal',
#              size='large')


# ticks = np.linspace(0, 2*np.pi, 20, endpoint=False)
# plt.xticks(angles, categories, size=16)
# for label,rot in zip(ax.get_xticklabels(),ticks):
#     label.set_rotation(rot*180./np.pi)
#     label.set_horizontalalignment("right")
#     label.set_rotation_mode("anchor")

catlabels=[]

for index,cat in enumerate(categories):
    #str(categories[index])+"\n"+
    catlabels.append(str(maxcat[index]))


plt.xticks(angles, categories, size=16)
for label,i in zip(ax.get_xticklabels(),range(0,len(angles))):
    angle_rad=angles[i]
    if angle_rad <= np.pi/2:
        ha= 'left'
        va= "bottom"
        angle_text=angle_rad*(-180/np.pi)+90
    elif np.pi/2 < angle_rad <= np.pi:
        ha= 'right'
        va= "top"
        angle_text=angle_rad*(-180/np.pi)+90
    elif np.pi < angle_rad <= (3*np.pi/2):
        ha= 'right'
        va= "top"
        angle_text=angle_rad*(-180/np.pi)-90
    else:
        ha= 'left'
        va= "bottom"
        angle_text=angle_rad*(-180/np.pi)-90
    label.set_rotation(angle_text)
    label.set_verticalalignment(va)
    label.set_horizontalalignment(ha)



plt.show()



# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# import numpy as np
# from matplotlib.mlab import bivariate_normal
# N = 1000
# X, Y = np.mgrid[0.6:0.9:complex(0, N), 0.6:0.9:complex(0, N)]
# Z1 = minVal+0.0001 *bivariate_normal(X, Y, 0.05,0.05, ((maxVal+minVal)/2),((maxVal+minVal)/2))
# plt.subplot(2, 1, 1)
# plt.pcolor(X, Y, Z1, norm=LogNorm(vmin=Z1.min(), vmax=Z1.max()), cmap='rainbow_r')
# from matplotlib.ticker import LogFormatter
# formatter = LogFormatter(10, labelOnlyBase=False)
# cb = plt.colorbar(ticks=[0.783,0.784,0.785,0.786,0.787,0.789], format=formatter)
# plt.subplot(2, 1, 2)
# plt.pcolor(X, Y, Z1, cmap='rainbow_r')
# plt.colorbar()
# plt.show()


