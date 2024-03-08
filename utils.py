# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 14:45:07 2022

@author: sletizia
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def make_video(image_folder,fps,output):   
    #makes mp4 video form png in a folder
    import os
    import moviepy.video.io.ImageSequenceClip
    image_files = [os.path.join(image_folder,img) for img in os.listdir(image_folder) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(output+'.mp4')

def make_video2(folder,output,fps=1):
    import cv2
    import os
    images = [img for img in os.listdir(folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
def make_video_v3(folder,output,fps=1):
    import cv2
    import os
    
    # Get the list of PNG images in the directory
    image_files = [file for file in os.listdir(folder) if file.endswith('.png')]
    
    print(str(len(image_files))+' images found')
    # Sort the image files to ensure proper ordering
    image_files.sort()
    
    # Get the first image to determine the video dimensions
    first_image = cv2.imread(os.path.join(folder, image_files[0]))
    height, width, _ = first_image.shape
    
    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your desired video format
    video_writer = cv2.VideoWriter(output, fourcc, fps, (width, height))  # Adjust the frame rate (here set to 25)
    
    # Iterate over the image files and write each frame to the video
    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        frame = cv2.imread(image_path)
        
        # Resize frame if its dimensions are different from the first image
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        
        video_writer.write(frame)
    
    # Release the VideoWriter and close the video file
    video_writer.release()
    
    print('Video saved as '+output)
    
def mkdir(path):
    #makes recursively folder from path, no existing error
    import os
    path.replace('\\','/')
    folders=path.split('/')
    upper=''
    for f in folders:
        try:
            os.mkdir(upper+f)           
        except:
            pass
                
        upper+=f+'/'
        
def percentile_filt(x,percentiles):
    #filter based on percentiles
    x_filt=x.copy()
    if np.sum(~np.isnan(x))>0:        
        excl=(x<np.nanpercentile(x,percentiles[0])) | (x>np.nanpercentile(x,percentiles[1]))
        x_filt[excl]=np.nan
    else:
        excl=np.zeros(np.shape(x))==1
    return x_filt,excl

def datenum(string,format="%Y-%m-%d %H:%M:%S.%f"):
    from datetime import datetime
    num=(datetime.strptime(string, format)-datetime(1970, 1, 1)).total_seconds()
    return num

# def dtinum(dt):
#     import pandas as pd
#     return (dt -pd.Timestamp("1970-01-01"))/pd.Timedelta("1s")

def dti2num(dt):
    import pandas as pd
    
    try:
        tz=dt.tz
    except:
        tz=None    
    
    if tz==None:
        n=(dt -pd.Timestamp("1970-01-01"))/pd.Timedelta("1s")
    else:
        n=(dt -pd.Timestamp("1970-01-01",tz=str(tz)))/pd.Timedelta("1s")
    return n
    
    
def datestr(num,format="%Y-%m-%d %H:%M:%S.%f"):
    from datetime import datetime
    string=datetime.utcfromtimestamp(num).strftime(format)
    return string
    
def file_list(path,recursive=False):
    #returns file names in a directory
    import glob       
    filenames0=glob.glob(path,recursive=recursive)
    filenames1 = [f.replace('\\','/') for f in filenames0]
    
    filenames=[f.split('/')[-1] for f in filenames1]
    return filenames

def AR1(phi,x0,N,mu,sigma):
    import numpy as np
    x=np.zeros((N))
    x[0]=x0
    for i in range(1,N):
        x[i]=phi*x[i-1]+np.random.normal(0,sigma*(1-phi**2)**0.5)
    return x+mu

def autocorr(x):
    mu=np.nanmean(x)
    ones=np.zeros(np.shape(x))+1
    N=len(x)
    r=np.convolve(x-mu,np.flip(x)-mu,'full')
    n=np.convolve(ones,ones)
    
    return r[N-1:]/n[N-1:]/(r[N-1]/N)

def xcorr(x,y,N_ratio):
    N=int(len(x)*N_ratio)
    lags=np.arange(-N+1,N)
    xx=x-np.nanmean(x)
    yy=y-np.nanmean(y)
    
    rho_pos=[np.nansum(xx[:-i]*yy[i:])/(np.nansum(xx[:-i]**2)**0.5*np.nansum(yy[i:]**2)**0.5) for i in range(1,N)]
    rho_neg=[np.nansum(xx[i:]*yy[:-i])/(np.nansum(xx[i:]**2)**0.5*np.nansum(yy[:-i]**2)**0.5) for i in range(N-1,0,-1)]
    rho=np.array(rho_neg+[np.nansum(xx*yy)/(np.nansum(xx**2)**0.5*np.nansum(yy**2)**0.5) ]+rho_pos)
    return rho,lags

def resample(DataIn,period,func):
    x=DataIn.index
    x_round=np.floor(x/period)*period+period/2    
    IndexNew=np.unique(x_round)
    DataOut=DataIn.groupby(x_round).apply(func)    
    if np.shape(DataOut)[0]==len(IndexNew):
        DataOut.index=IndexNew
    return  DataOut

def poly_detrend(DataIn,degree):
    DataOut=DataIn+np.nan
    x=DataIn.index.to_numpy()
    x0=np.nanmean(x)
    sx=np.nanstd(x) +10**(-10)            
    try:
        for col in DataIn.columns:    
            y=DataIn[col].to_numpy()
            reals=~np.isnan(x+y)
            if np.sum(reals)>degree:
                y0=np.nanmean(y)
                sy=np.nanstd(y)+10**(-10)   
                y_det=y+np.nan        
                coeff=np.polyfit((x[reals]-x0)/sx,(y[reals]-y0)/sy,degree)
                y_tr=np.polyval(coeff,(x[reals]-x0)/sx)*sy+y0
                y_det[reals]=y[reals]-y_tr     
                DataOut[col]=y_det
    except: 
        pass
    return DataOut

def poly_detrend_v2(DataIn,degree):
    DataOut=DataIn+np.nan
    x=DataIn.index.to_numpy()
    if type(x[0])=='numpy.datetime64':
        x=numpy_datetime2num(x)
    x0=np.nanmean(x)
    sx=np.nanstd(x) +10**(-10)            
    try:
        for col in DataIn.columns:    
            y=DataIn[col].to_numpy()
            reals=~np.isnan(x+y)
            if np.sum(reals)>degree:
                y0=np.nanmean(y)
                sy=np.nanstd(y)+10**(-10)   
                y_det=y+np.nan        
                coeff=np.polyfit((x[reals]-x0)/sx,(y[reals]-y0)/sy,degree)
                y_tr=np.polyval(coeff,(x[reals]-x0)/sx)*sy+y0
                y_det[reals]=y[reals]-y_tr     
                DataOut[col]=y_det
    except: 
        pass
    return DataOut

def bootstrap(x,launches):#(Politis and White, Econometric Reviews 2004)
#02/15/2022: created, finalized
    N=len(x)
    mu=np.nanmean(x)
    r=np.convolve(x-mu,np.flip(x)-mu,'full')/N
    
    #(Politis, J. Nonparametric Statist 2003)
    min_r=2*(np.log10(N)/N)**0.5
    Kn=int(np.max([5.00,np.log10(N)**0.5]))
    
    for Q in range(N-Kn-1):
        if np.max(np.abs(r[N-1+Q+1:N-1+Q+Kn+1]/r[N-1]))<min_r:
            break
    Q=min(Q+1,N/2-1)
    q=np.arange(-Q*2,Q*2+1)
   
    l=np.zeros(len(q))+1
    taper=np.abs(q/(2*Q))>0.5
    l[taper]=2*(1-np.abs(q[taper]/(2*Q)))
    
    G=np.sum(l*np.abs(q)*r[N-1+q])
    D=4/3*np.sum(l*r[N-1+q])**2
    B=max(min(int(np.round((2*G**2/D)**(1/3)*N**(1/3))),N),1)
    
    K=int(np.ceil(N/B))
    x_BS_all=np.zeros((N,launches))
    for m in range(launches):
        x_BS=[]
        for i in range(K):
            j=np.random.randint(0,N)+np.arange(B)
            x_BS=np.append(x_BS,x[j%N])
        x_BS_all[:,m]=x_BS[:N]
    mu_BS=np.mean(x_BS_all,0)
    return mu_BS,B

def ang_diff(angle1,angle2=None,unit='deg',mode='scalar'):
    try:
        if 'list' in str(type(angle1)) or 'int' in str(type(angle1)):
            angle1=np.array(angle1).astype(float)
        if 'list' in str(type(angle1)) or 'int' in str(type(angle2)):
            angle2=np.array(angle2).astype(float)
    except: pass
    if unit=='rad':
        angle1=angle1*180/np.pi
        angle2=angle2*180/np.pi
    angle1=angle1 % 360
    
    if mode=='scalar':
        angle2=angle2 % 360
    
   
    if mode=='scalar':
        dx=angle1-angle2
    elif mode=='vector':
        dx=np.diff(angle1)

    if len2(dx)>1:
        dx[dx>180]= dx[dx>180]-360
        dx[dx<-180]= dx[dx<-180]+360
    else:
        if dx>180:
            dx-=360
        if dx<-180:
            dx+=360
        
    if unit=='rad':
        dx=dx/180*np.pi
        
    return dx

def scan_identifier(azi,ele,ang_tol):
    
    dazi=np.max(np.abs(ang_diff(azi,mode='vector')))
    dele=np.max(np.abs(ang_diff(ele,mode='vector')))
    vert=np.abs(ele-90)>ang_tol
    if np.sum(vert)>0:
        azi_vert_wrap=np.append(np.sort(azi[vert]),np.min(azi[vert]))
    
    if dazi<ang_tol and dele<ang_tol:
        scan_type='stare'
    elif (np.max(np.diff(np.arctan(np.tan(azi/180*np.pi))*180/np.pi))<ang_tol) and (dele>ang_tol):
        scan_type='RHI'
    elif dazi>ang_tol and dele<ang_tol: 
        if np.abs(np.abs(ang_diff(np.min(azi),np.max(azi)))-np.nanmean(np.abs(np.diff(np.sort(azi)))))>ang_tol:
            scan_type='PPI'
        else:
            scan_type='VAD'     
    elif np.max(np.abs(ang_diff(ele[vert],mode='vector')))<ang_tol and np.sum(~vert)==1:
        if np.max(np.abs(np.abs(ang_diff(azi_vert_wrap,mode='vector'))-90))<ang_tol:
            scan_type='DBS'
        else:
            scan_type='custom profile ('+str(len(vert))+' beams)'
    else:
        scan_type='custom'
    return scan_type
        
def interp_df(df, new_index,max_dx):
    import pandas as pd
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():    
        df_out[colname] =interp_dist(df.index,col,new_index,max_dx)

    return df_out

def interp_df_v2(df, new_index,max_dx):
    import pandas as pd
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.items():    
        df_out[colname] =interp_dist(df.index,col,new_index,max_dx)

    return df_out

def interp_dist(x,y,xs,max_dx,method='linear'):
    from scipy import interpolate
    f=interpolate.interp1d(x,x,kind='nearest',bounds_error=False)
    d=np.abs(xs-f(xs))
    f=interpolate.interp1d(x,y,kind=method,bounds_error=False)
    ys=f(xs)
    ys[d>max_dx]=np.nan
    return ys

def findall(string,substring):
    i=np.arange(len2(string))
    f=[s==substring for s in string]
    return i[f]



def cosd(x):
    return np.cos(x/180*np.pi)

def sind(x):
    return np.sin(x/180*np.pi)
    
def tand(x):
    return np.tan(x/180*np.pi)

def arctand(x):
    return np.arctan(x)*180/np.pi

def arccosd(x):
    return np.arccos(x)*180/np.pi

def arcsind(x):
    return np.arcsin(x)*180/np.pi

def factorial(x):
    """This is a recursive function
    to find the factorial of an integer"""

    if x == 1:
        return 1
    else:
        # recursive call to the function
        return (x * factorial(x-1))
    
def vec_diff(x1,x2):
    a=np.tile(x1,(len(x2),1))
    b=np.tile(x2,(len(x1),1))
    return np.transpose(a)-b

def vec_diff2(x1,x2):
    d=np.zeros((len(x1),len(x2)))
    i=0
    for a in x1:
        j=0
        for b in x2:
            d[i,j]=a-b
            j+=1
        i+=1
    return d

def vstack(a,b):
    if len(a)>0:
        ab=np.vstack((a,b))
    else:
        ab=b
    return ab

def hstack(a,b):
    return vstack(a,b).T

def hstack_v2(a,b):
    if len(np.shape(b))==1:
        b=np.reshape(b,(len(b),1))
    if len(a)>0:
        ab=np.hstack((a,b))
    else:
        ab=b
    return ab

def vec2str(vec,separator=' ',format='%f'):
    s=''
    for v in vec:
        s=s+format % v+separator
    return s[:-len(separator)]

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def normalize(x,method='maxmin'):
    if method=='maxmin':
        x_norm= (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
    elif method=='meanstd':
        x_norm= (x-np.nanmean(x))/np.nanstd(x)
        
    return(x_norm)
    

def round2(x,resolution):
    return np.round(x/resolution)*resolution

def floor(x,resolution):
    return np.floor(x/resolution)*resolution

def ceil(x,resolution):
    return np.ceil(x/resolution)*resolution

def find_nearest(x, x0,max_dx=np.inf):
    i_n = np.abs(x - x0).argmin()
    if np.abs(x[i_n]-x0)<=max_dx:
        x_n=x[i_n]
    else:
        i_n=np.nan
        x_n=np.nan
    return x_n,i_n

def str2float(x):
    f=np.zeros(len(x))
    for i in range(len(x)):
        try:
            f[i]=float(x[i])
        except:
            f[i]=np.nan
    return f

def len2(x):
    if 'int' in str(type(x)) or 'float' in str(type(x)):
        return 1
    elif 'list' in str(type(x)) or 'array' in str(type(x))  or 'str' in str(type(x)) or 'series' in str(type(x)):
        return len(x)
    else:
        raise ValueError
        
def subplots(vert_size,hor_size,margin=0.05,spacing=0.02):

    w=hor_size/np.sum(hor_size)*(1-2*margin-len(hor_size)*spacing)
    x=margin+np.append(0,np.cumsum(w[:-1]+spacing))
    
    h=vert_size/np.sum(vert_size)*(1-2*margin-len(vert_size)*spacing)
    y=np.flip(margin+np.append(0,np.cumsum(np.flip(h[1:])+spacing)))
    
    ax=[]
    #fig=plt.figure()
    for j in range(len(hor_size)):
        for i in range(len(vert_size)):
            ax.append(plt.gcf().add_axes((x[j],y[i],w[j],h[i])))
    return ax

def cart2pol(x,y):
    r=(x**2+y**2)**0.5
    th=arctand(y/x)
    if len2(th)>1:
        th[x<0]-=180
    else:
        if x<0:
            th-=180
    return r,th%360

def plot_linfit(x,y):
    
    from scipy.stats import linregress
    
    reals=~np.isnan(x+y)
    if np.sum(reals)>1:
        lf=np.round(linregress(x[reals],y[reals]),3)
  
        rho=np.round(np.corrcoef(x[reals],y[reals])[0][1],3)
        MAE=np.round(np.nanmean(np.abs(x-y)),3)
        
        lb=np.nanmin(np.append(x,y))
        ub=np.nanmax(np.append(x,y))
        
        scatterplot=plt.plot(x,y,'.k',alpha=0.5)[0]
        plt.plot([lb,ub],np.array([lb,ub])*lf[0]+lf[1],'r')
        plt.plot([lb,ub],[lb,ub],'--b')
        txt=plt.text(lb+(ub-lb)*0.05,lb+(ub-lb)*0.8,r'$y='+str(lf[1])+r'+'+str(lf[0])+r'x$'+'\n'+
                 r'$\rho='+str(rho)+r'$'+'\n'+r'MAE $='+str(MAE)+r'$',color='r')
        plt.xlim([lb,ub])
        plt.ylim([lb,ub])
        plt.grid()
        plt.gca().set_box_aspect(1)
        
        gof={'Intercept':lf[1],'Slope':lf[0],r'$\rho$':rho,'MAE':MAE}
        
        return gof,scatterplot,txt
    
def plot_linfit_v2(x,y):
#05/01/2023 (v 2): ticks equalized
    from scipy.stats import linregress
    
    reals=~np.isnan(x+y)
    if np.sum(reals)>1:
        lf=np.round(linregress(x[reals],y[reals]),3)
  
        rho=np.round(np.corrcoef(x[reals],y[reals])[0][1],3)
        MAE=np.round(np.nanmean(np.abs(x-y)),3)
        
        lb=np.nanmin(np.append(x,y))
        ub=np.nanmax(np.append(x,y))
        
        scatterplot=plt.plot(x,y,'.k',alpha=0.5)[0]
        plt.plot([lb,ub],np.array([lb,ub])*lf[0]+lf[1],'r')
        plt.plot([lb,ub],[lb,ub],'--b')
        txt=plt.text(lb+(ub-lb)*0.05,lb+(ub-lb)*0.8,r'$y='+str(lf[1])+r'+'+str(lf[0])+r'x$'+'\n'+
                 r'$\rho='+str(rho)+r'$'+'\n'+r'MAE $='+str(MAE)+r'$',color='r')
        plt.xlim([lb,ub])
        plt.ylim([lb,ub])
        plt.grid()
        plt.gca().set_box_aspect(1)
        plt.yticks(plt.gca().get_xticks())
        
        gof={'Intercept':lf[1],'Slope':lf[0],r'$\rho$':rho,'MAE':MAE}
        
        return gof,scatterplot,txt
        
def plot_linfit_v3(x,y,lb=[],ub=[]):
#05/01/2023 (v 2): ticks equalized
#11/09/2023 (v 3): boundaries as inputs
    from scipy.stats import linregress
    
    reals=~np.isnan(x+y)
    if np.sum(reals)>1:
        lf=np.round(linregress(x[reals],y[reals]),3)
  
        rho=np.round(np.corrcoef(x[reals],y[reals])[0][1],3)
        MAE=np.round(np.nanmean(np.abs(x-y)),3)
        
        if lb==[]:
            lb=np.nanmin(np.append(x,y))
        if ub==[]:
            ub=np.nanmax(np.append(x,y))
        
        scatterplot=plt.plot(x,y,'.k',alpha=0.5)[0]
        plt.plot([lb,ub],np.array([lb,ub])*lf[0]+lf[1],'r')
        plt.plot([lb,ub],[lb,ub],'--b')
        txt=plt.text(lb+(ub-lb)*0.05,lb+(ub-lb)*0.8,r'$y='+str(lf[1])+r'+'+str(lf[0])+r'x$'+'\n'+
                 r'$\rho='+str(rho)+r'$'+'\n'+r'MAE $='+str(MAE)+r'$',color='r')
        
        plt.grid()
        plt.gca().set_box_aspect(1)
        plt.xlim([lb,ub])
        plt.ylim([lb,ub])
        plt.yticks(plt.gca().get_xticks())
        
        gof={'Intercept':lf[1],'Slope':lf[0],r'$\rho$':rho,'MAE':MAE}
        
        return gof,scatterplot,txt
    
def plot_linfit_v4(x,y,lb=[],ub=[]):
#05/01/2023 (v 2): ticks equalized
#11/09/2023 (v 3): boundaries as inputs
#12/01/2023 (v 4): RMSD instead of MAE, finalized
    from scipy.stats import linregress
    
    reals=~np.isnan(x+y)
    if np.sum(reals)>1:
        lf=np.round(linregress(x[reals],y[reals]),3)
  
        rho=np.round(np.corrcoef(x[reals],y[reals])[0][1],3)
        RMSD=np.round(np.nanmean((x-y)**2)**0.5,3)
        
        if lb==[]:
            lb=np.nanmin(np.append(x,y))
        if ub==[]:
            ub=np.nanmax(np.append(x,y))
        
        scatterplot=plt.plot(x,y,'.k',alpha=0.5)[0]
        plt.plot([lb,ub],np.array([lb,ub])*lf[0]+lf[1],'r')
        plt.plot([lb,ub],[lb,ub],'--b')
        txt=plt.text(lb+(ub-lb)*0.05,lb+(ub-lb)*0.8,r'$y='+str(lf[1])+r'+'+str(lf[0])+r'x$'+'\n'+
                 r'$\rho='+str(rho)+r'$'+'\n'+r'RMSD $='+str(RMSD)+r'$',color='r')
        
        plt.grid()
        plt.gca().set_box_aspect(1)
        plt.xlim([lb,ub])
        plt.ylim([lb,ub])
        plt.yticks(plt.gca().get_xticks())
        
        gof={'Intercept':lf[1],'Slope':lf[0],r'$\rho$':rho,'RMSD':RMSD}
        
        return gof,scatterplot,txt        
        
def circmean(x):
    return cart2pol(np.nanmean(cosd(x)), np.nanmean(sind(x)))[1]

def utcdate(x):
    from datetime import datetime
    if 'array' in str(type(x)) or 'list' in str(type(x)) or 'float' in str(type(x)) or 'int' in str(type(x)) or 'Float64Index' in str(type(x)):
        output= [datetime.utcfromtimestamp(t) for t in x]
    elif 'Series' in str(type(x)) or 'DataFrame' in str(type(x)):
        x2= [datetime.utcfromtimestamp(t) for t in x.index]
        output=x.copy()
        output.index=x2
    else:
        raise ValueError('Incorrect type')
    
    return output

def ang_comp(ang_ref,ang):
    da=ang_diff(ang,ang_ref,mode='scalar')
    
    return (ang_ref+da)

def save_all_fig(name,newfolder=False,resolution=300):
    mkdir('figures')
    if newfolder:
        mkdir('figures/'+name)
    figs = [plt.figure(n) for n in plt.get_fignums()]
    inc=0
    for fig in figs:
        if newfolder:
            fig.savefig('figures/'+name+'/'+'{i:02d}'.format(i=inc)+'.png',dpi=resolution, bbox_inches='tight')
        else:
            fig.savefig('figures/'+name+'_'+'{i:02d}'.format(i=inc)+'.png',dpi=resolution, bbox_inches='tight')
        inc+=1

def round_time(timestamp):
    from datetime import datetime

    t=datetime.utcfromtimestamp(timestamp)
    
    t0=datetime(t.year,t.month,t.day)
    return (t0-datetime(1970, 1, 1)).total_seconds()

def mid(x):
    return (x[:-1]+x[1:])/2

def rev_mid(x):
    return np.concatenate([[x[0]-(x[1]-x[0])/2],(x[:-1]+x[1:])/2,[x[-1]+(x[-1]-x[-2])/2]])

def mean_str(DataIn):
    tp=[str(type(DataIn[c].iloc[0])) for c in DataIn.columns]
    
    D_avg=DataIn.mean()
    
    sel=['str' in t for t in tp]
    for s in DataIn.columns[sel]:
        if len(np.unique(DataIn[s]))==1:
            D_avg[s]=DataIn[s].iloc[0]
        else:
            D_avg[s]=''
    return D_avg

def resample_flex(DataIn,x1,x2,func):
    import pandas as pd
  
    DataOut=pd.DataFrame([])
    for c in DataIn.columns:
        DataOut[c]=np.nan*np.zeros(len(x1))
    
    x_bin=np.zeros(len(x1)*2)
    x_bin[np.arange(0,len(x_bin),2)]=x1
    x_bin[np.arange(1,len(x_bin),2)]=x2

    index=np.digitize(DataIn.index,x_bin)
    index[index/2==np.round(index/2)]=-1
    index=[int((i-1)/2) for i in index]
    
    D=DataIn.groupby(index).apply(func) 
    for c in DataIn.columns:
        DataOut[c].iloc[D.index]=D[c]

    DataOut.index=(x1+x2)/2

    return  DataOut


def resample_flex_v2(DataIn,x1,x2,func):
#07/19/2022 (v 2): added detrend option
    import pandas as pd
  
    DataOut=pd.DataFrame([])
    for c in DataIn.columns:
        DataOut[c]=np.nan*np.zeros(len(x1))
    
    x_bin=np.zeros(len(x1)*2)
    x_bin[np.arange(0,len(x_bin),2)]=x1
    x_bin[np.arange(1,len(x_bin),2)]=x2
    index=(np.digitize(DataIn.index,x_bin)-1)/2
    index[index!=np.round(index)]=-1
    DataIn2=DataIn.copy()
    DataIn2.iloc[index==-1,:]=np.nan
    
    D=DataIn2.groupby(index).apply(func)

    if len(D)==len(index):
            DataOut=D.copy()
    else:
        DataOut=D.iloc[D.index>=0,:]
        DataOut.index=(x1+x2)[DataOut.index.astype(int)]/2

    return  DataOut

def resample_flex_v2_1(DataIn,x1,x2,func):
#07/19/2022 (v 2): added detrend option
#04/28/2023 (v 2.1): can handle numpy.datetime64
    import pandas as pd
  
    DataOut=pd.DataFrame([])
    for c in DataIn.columns:
        DataOut[c]=np.nan*np.zeros(len(x1))
    
    x_bin=np.zeros(len(x1)*2)
    x_bin[np.arange(0,len(x_bin),2)]=x1
    x_bin[np.arange(1,len(x_bin),2)]=x2
    index=(np.digitize(DataIn.index,x_bin)-1)/2
    index[index!=np.round(index)]=-1
    DataIn2=DataIn.copy()
    DataIn2.iloc[index==-1,:]=np.nan
    
    D=DataIn2.groupby(index).apply(func)

    if len(D)==len(index):
            DataOut=D.copy()
    else:
        DataOut=D.iloc[D.index>=0,:]
        DataOut.index=(x1+(x2-x1)/2)[DataOut.index.astype(int)]

    return  DataOut

def resample_flex_v2_2(DataIn,x1,x2,func):
#07/19/2022 (v 2): added detrend option
#04/28/2023 (v 2.1): can handle numpy.datetime64
#05/03/2023 (v 2.2): fixed bug on missing peridods
    import pandas as pd
  
    DataOut=pd.DataFrame([])
    for c in DataIn.columns:
        DataOut[c]=np.nan*np.zeros(len(x1))
    
    x_bin=np.zeros(len(x1)*2)
    x_bin[np.arange(0,len(x_bin),2)]=x1
    x_bin[np.arange(1,len(x_bin),2)]=x2
    index=(np.digitize(DataIn.index,x_bin)-1)/2
    index[index!=np.round(index)]=-1
    DataIn2=DataIn.copy()
    DataIn2.iloc[index==-1,:]=np.nan
    
    D=DataIn2.groupby(index).apply(func)

    if len(D)==len(index):
            DataOut=D.copy()
    else:
        DataOut.iloc[D.index[D.index>=0].astype(int),:]=D.iloc[D.index>=0,:]
        DataOut.index=(x1+(x2-x1)/2)[DataOut.index.astype(int)]

    return  DataOut


def cart(a,b):
    AB=np.meshgrid(a,b)
    return AB[0]*AB[1]

def threshold(DataIn,lim):
    tp=[str(type(DataIn[c].iloc[0])) for c in DataIn.columns]
    
    sel=np.where(['float' in t for t in tp])[0]
    Data_sel=DataIn.iloc[:,sel]
    Data_sel[Data_sel<lim[0]]=np.nan
    Data_sel[Data_sel>lim[1]]=np.nan
    
    DataOut=DataIn.copy()
    DataOut.iloc[:,sel]=Data_sel
    
    return DataOut

def surf(Z,X=None,Y=None):
    from matplotlib import cm
    if X is None:
        x=np.arange(np.shape(Z)[1])
        y=np.arange(np.shape(Z)[0])
        X,Y=np.meshgrid(x,y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    return surf,ax,fig

def legend_spec(ax,indices):
    handles,labels  = ax.get_legend_handles_labels()
    plt.legend([handles[i] for i in indices],[labels[i] for i in indices])

def general_inv(A):
    try:
        A_inv=np.linalg.inv(A)
    except:
        A_inv=np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A))
        
    return A_inv


# def draw_turbine(x0,y0,z0,D,yaw,H,alpha):
    
#     b1=np.zeros((3,100))
#     b1[0,:]=0
#     b1[1,:]=0
#     b1[2,:]=np.linspace(0,0.5,100)*D
    
#     Mb=np.array([[1,0,0],[0,cosd(120),sind(120)],[0,-sind(120),cosd(120)]])
#     b2=np.matmul(Mb,b1)
#     b3=np.matmul(Mb,b2)
    
#     Mt=[[cosd(90-yaw),sind(90-yaw),0],[-sind(90-yaw),cosd(90-yaw),0],[0,0,1]]
    
#     rotor0=np.hstack((b1,b2,b3))
#     rotor=np.matmul(Mt,rotor0)
    
#     plt.plot(rotor[0,:]+x0,rotor[1,:]+y0,rotor[2,:]+z0,'.k',markersize=0.75,alpha=alpha)
#     plt.plot(np.zeros(100)+x0,np.zeros(100)+y0,np.linspace(-H,0,100)+z0,'.k',markersize=2,alpha=alpha)

def lidar_xyz(r,ele,azi):
    Nr=len2(r)
    Nb=len2(ele)
    
    R=np.transpose(np.tile(r,(Nb,1)))
    A=np.tile(azi,(Nr,1))
    E=np.tile(ele,(Nr,1))
    
    X=R*cosd(E)*cosd(90-A)
    Y=R*cosd(E)*sind(90-A)
    Z=R*sind(E)
    
    return X,Y,Z

def dynamic_filter(t,r,ele,azi,rws,snr,Dx,Dy,Dz,Dt,max_rws,min_p,min_N,min_ratio):
    from scipy import stats
    import time as tm
    
    #performance
    t0=tm.time()
    t_calc=0
    steps=['Stats RWS','Index RWS','Stats SNR','Index SNR','Stats RWS/SNR','Index RWS/SNR','QC1','Stats N','Index N','QC2']

    X,Y,Z=lidar_xyz(r, ele, azi)
    T,R=np.meshgrid(t,r)
    
    bin_x=np.arange(np.min(X),np.max(X)+Dx,Dx)
    bin_y=np.arange(np.min(Y),np.max(Y)+Dy,Dy)
    bin_z=np.arange(np.min(Z),np.max(Z)+Dz,Dz)
    bin_t=np.arange(np.min(T),np.max(T)+Dt,Dt)

    S=np.transpose(np.vstack((X.ravel(),Y.ravel(),Z.ravel(),T.ravel())))
    B=stats.binned_statistic_dd(S,rws.ravel(),'mean',[bin_x,bin_y,bin_z,bin_t],expand_binnumbers=True)
    rws_avg=B[0]
    index=B[2]
    t_calc=np.append(t_calc,tm.time()-t0)

    rws_avg_all=np.reshape(rws_avg[(index[0]-1,index[1]-1,index[2]-1,index[3]-1)],np.shape(rws))
    rws_norm=rws-rws_avg_all
    t_calc=np.append(t_calc,tm.time()-t0)

    snr_avg=stats.binned_statistic_dd(S,snr.ravel(),'mean',[bin_x,bin_y,bin_z,bin_t])[0]
    t_calc=np.append(t_calc,tm.time()-t0)

    snr_avg_all=np.reshape(snr_avg[(index[0]-1,index[1]-1,index[2]-1,index[3]-1)],np.shape(snr))
    snr_norm=snr-snr_avg_all
    t_calc=np.append(t_calc,tm.time()-t0)
                     
    #%probability
    sel=~np.isnan(rws_norm.ravel()+snr_norm.ravel())
    drws=3.49*np.nanstd(rws_norm.ravel()[sel])/(np.sum(sel))**(1/3)
    bin_rws=np.arange(np.min(rws_norm.ravel()[sel]),np.max(rws_norm.ravel()[sel])+drws,drws)
    dsnr=3.49*np.nanstd(snr_norm.ravel()[sel])/(np.sum(sel))**(1/3)
    bin_snr=np.arange(np.min(snr_norm.ravel()[sel]),np.max(snr_norm.ravel()[sel])+dsnr,dsnr)

    S=np.transpose(np.vstack((rws_norm.ravel()[sel],snr_norm.ravel()[sel])))
    B=stats.binned_statistic_dd(S,[],'count',[bin_rws,bin_snr],expand_binnumbers=True)
    t_calc=np.append(t_calc,tm.time()-t0)
                     
    p=B[0]/np.max(B[0])                         
    index2=B[2]
    p_sel=np.zeros(len(rws_norm.ravel()))
    p_sel[sel]=p[(index2[0]-1,index2[1]-1)]
    p_all=np.reshape(p_sel,np.shape(rws))   
    t_calc=np.append(t_calc,tm.time()-t0) 

    #qc1
    good=p_all>min_p
    rws_qc=rws.copy()
    rws_qc[np.abs(rws_qc)>max_rws]=np.nan 
    rws_qc[good==0]=np.nan   
    t_calc=np.append(t_calc,tm.time()-t0)

    #qc2
    S=np.transpose(np.vstack((X.ravel(),Y.ravel(),Z.ravel(),T.ravel())))
    N=stats.binned_statistic_dd(S,[],'count',[bin_x,bin_y,bin_z,bin_t],expand_binnumbers=True)[0]
    N_good=stats.binned_statistic_dd(S,good.ravel(),'sum',[bin_x,bin_y,bin_z,bin_t],expand_binnumbers=True)[0]
    t_calc=np.append(t_calc,tm.time()-t0)
                     
    N_all=np.reshape(N[(index[0]-1,index[1]-1,index[2]-1,index[3]-1)],np.shape(rws))
    N_good_all=np.reshape(N_good[(index[0]-1,index[1]-1,index[2]-1,index[3]-1)],np.shape(rws))
    ratio_all=N_good_all/(N_all+10**(-16))
    t_calc=np.append(t_calc,tm.time()-t0)

    rws_qc2=rws_qc.copy()
    rws_qc2[N_all<min_N]=np.nan
    rws_qc2[ratio_all<min_ratio]=np.nan
    t_calc=np.append(t_calc,tm.time()-t0)
    
    return rws_qc,rws_qc2,rws_norm,snr_norm,t_calc,steps


def dynamic_filter_v2(t,r,ele,azi,rws,snr,Dx,Dy,Dz,Dt=600,max_rws=30,min_p=0.0025,min_N=10,min_ratio=0.2):
    #09/05/2022: created
    #09/06/2022: finalized
    from scipy import stats
    import time as tm
    
    #performance
    t0=tm.time()
    t_calc=0
    steps=['Stats N','Stats RWS','Index RWS','Stats SNR','Index SNR','Stats RWS/SNR','Index RWS/SNR','QC1','Stats N good','Index N','QC2']

    X,Y,Z=lidar_xyz(r, ele, azi)
    T,R=np.meshgrid(t,r)
    
    bin_x=np.arange(np.min(X),np.max(X)+Dx,Dx)
    bin_y=np.arange(np.min(Y),np.max(Y)+Dy,Dy)
    bin_z=np.arange(np.min(Z),np.max(Z)+Dz,Dz)
    bin_t=np.arange(np.min(T),np.max(T)+Dt,Dt)

    S=np.transpose(np.vstack((X.ravel(),Y.ravel(),Z.ravel(),T.ravel())))
    B=stats.binned_statistic_dd(S,[],'count',[bin_x,bin_y,bin_z,bin_t],expand_binnumbers=True)
    N=B[0]
    index=B[2]
    t_calc=np.append(t_calc,tm.time()-t0)
    
    B=stats.binned_statistic_dd(S,rws.ravel(),'mean',[bin_x,bin_y,bin_z,bin_t])
    rws_avg=B[0]
    t_calc=np.append(t_calc,tm.time()-t0)

    rws_avg_all=np.reshape(rws_avg[(index[0]-1,index[1]-1,index[2]-1,index[3]-1)],np.shape(rws))
    rws_norm=rws-rws_avg_all
    t_calc=np.append(t_calc,tm.time()-t0)

    snr_avg=stats.binned_statistic_dd(S,snr.ravel(),'mean',binned_statistic_result=B)[0]
    t_calc=np.append(t_calc,tm.time()-t0)

    snr_avg_all=np.reshape(snr_avg[(index[0]-1,index[1]-1,index[2]-1,index[3]-1)],np.shape(snr))
    snr_norm=snr-snr_avg_all
    t_calc=np.append(t_calc,tm.time()-t0)
                     
    #%probability
    sel=~np.isnan(rws_norm.ravel()+snr_norm.ravel())
    drws=3.49*np.nanstd(rws_norm.ravel()[sel])/(np.sum(sel))**(1/3)
    bin_rws=np.arange(np.min(rws_norm.ravel()[sel]),np.max(rws_norm.ravel()[sel])+drws,drws)
    dsnr=3.49*np.nanstd(snr_norm.ravel()[sel])/(np.sum(sel))**(1/3)
    bin_snr=np.arange(np.min(snr_norm.ravel()[sel]),np.max(snr_norm.ravel()[sel])+dsnr,dsnr)

    S2=np.transpose(np.vstack((rws_norm.ravel()[sel],snr_norm.ravel()[sel])))
    B2=stats.binned_statistic_dd(S2,[],'count',[bin_rws,bin_snr],expand_binnumbers=True)
    t_calc=np.append(t_calc,tm.time()-t0)
                     
    p=B2[0]/np.max(B2[0])                         
    index2=B2[2]
    p_sel=np.zeros(len(rws_norm.ravel()))
    p_sel[sel]=p[(index2[0]-1,index2[1]-1)]
    p_all=np.reshape(p_sel,np.shape(rws))   
    t_calc=np.append(t_calc,tm.time()-t0) 

    #qc1
    good=p_all>min_p
    rws_qc1=rws.copy()
    rws_qc1[np.abs(rws_qc1)>max_rws]=np.nan 
    rws_qc1[good==0]=np.nan   
    t_calc=np.append(t_calc,tm.time()-t0)

    #qc2
    N_good=stats.binned_statistic_dd(S,good.ravel(),'sum',binned_statistic_result=B)[0]
    t_calc=np.append(t_calc,tm.time()-t0)
                     
    N_all=np.reshape(N[(index[0]-1,index[1]-1,index[2]-1,index[3]-1)],np.shape(rws))
    N_good_all=np.reshape(N_good[(index[0]-1,index[1]-1,index[2]-1,index[3]-1)],np.shape(rws))
    ratio_all=N_good_all/(N_all+10**(-16))
    t_calc=np.append(t_calc,tm.time()-t0)

    rws_qc2=rws_qc1.copy()
    rws_qc2[N_all<min_N]=np.nan
    rws_qc2[ratio_all<min_ratio]=np.nan
    t_calc=np.append(t_calc,tm.time()-t0)
    
    return rws_qc1,rws_qc2,rws_norm,snr_norm,p_all,N_all,ratio_all,t_calc,steps


def axis_color(color):
    from matplotlib import pyplot as plt
    ax=plt.gca()
    ax.w_xaxis.set_pane_color(color)
    ax.w_yaxis.set_pane_color(color)
    ax.w_zaxis.set_pane_color(color)
    plt.draw()
    
    return None

def logabs(x):
    p=(np.log10(np.abs(x))*np.sign(x))
    return p


def plot_hist(x,bins):
    from scipy import stats
    m1=np.round(np.nanmean(x),3)
    m2=np.round(np.nanstd(x),3)
    m3=np.round(stats.skew(x,nan_policy='omit'),3)
    m4=np.round(stats.kurtosis(x,nan_policy='omit'),3)
    
    y_text=(plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])+plt.gca().get_ylim()[0]
    p=plt.hist(x,bins)
    t=plt.text(bins[0],y_text,r'$\mu={m}$'.format(m=m1)+'\n'+ r'$\sigma={m}$'.format(m=m2)+'\n'+'$\mu_3={m}$'.format(m=m3)+'\n'+'$\mu_4={m}$'.format(m=m4))
    plt.grid()
    
    return p,t

def median_filter(Data,window_mean,window_median,min_points=10,max_MAD=10):
#median despiking [Brock 1986]
#Data: data frame
#window_mean: number of point used for rolling normalization
#window_median: number of point used for rolling median filter
#max_MAD: maximum deviation from median expected

#11/01/2022: created
#11/04/2022: finalized
#11/18/2022: embedded normalization
#12/07/2022: added minimum number of points, rolling window centered
    
    import pandas as pd

    #inputs
    N_bin=[10,20,50,100,200,500]
    
    #data normalization
    Data_avg=Data.rolling(window_mean,center=True,min_periods=min_points).mean()
    Data_std=Data.rolling(window_mean,center=True,min_periods=min_points).std()
    Data_norm=(Data-Data_avg)/(Data_std+10**-16)
    
    MAD=Data_norm-Data_norm.rolling(window_median).median()

    #histogram-based threshold
    H_min=pd.DataFrame()
    H_min.index=[1]
    for c in MAD.columns:
        H_min[c]=max_MAD
    for n in N_bin:
        H_x=pd.DataFrame()
        H_y=pd.DataFrame()
        for c in MAD.columns:
            y,x=np.histogram(np.abs(MAD[c]),bins=n,range=[0,max_MAD],density=True)
            H_x[c],H_y[c]=mid(x),y 
       
        H_diff=H_y.diff()    
        for c in H_diff.columns:
            if H_min[c].values[0]==max_MAD:
                i_min=np.where(H_diff[c]>0)[0]
                if len(i_min)>0:
                    H_min[c]=H_x[c].iloc[i_min[0]-1]
               
        if H_min.max(axis='columns').values[0]<max_MAD:
            break
            
    excl=pd.DataFrame()
    for c in MAD.columns:
        excl[c]=np.abs(MAD[c])>H_min[c].values[0]

        
    return excl

def detrend(DataIn):
    DataOut=DataIn+np.nan
    for col in DataIn.columns:    
        y=DataIn[col].to_numpy()
        reals=~np.isnan(y)
        if np.sum(reals)>0:
            y0=np.nanmean(y)
            y_det=y+np.nan        
            y_det[reals]=y[reals]-y0     
            DataOut[col]=y_det

    return DataOut


def multiline(x, y, c, ax=None, **kwargs):
    from matplotlib.collections import LineCollection
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax
    
    if len(np.shape(x))==1:
        xs = np.array([x for i in range(len(y))]) # could also use np.tile
    else:
        xs=x.copy()
    
    if len(np.shape(y))==1:
        ys = np.array([y for i in range(len(x))]) # could also use np.tile
    else:
        ys=y.copy()

    # create LineCollection
    segments = [np.column_stack([xx, yy]) for xx, yy in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def poly_detrend_v2(DataIn,degree):
    if degree==0:
        DataOut=DataIn+np.nan
        for col in DataIn.columns:    
            y=DataIn[col].to_numpy()
            reals=~np.isnan(y)
            if np.sum(reals)>0:
                y0=np.nanmean(y)
                y_det=y+np.nan        
                y_det[reals]=y[reals]-y0     
                DataOut[col]=y_det
    else:
        
        DataOut=DataIn+np.nan
        x=DataIn.index.to_numpy()
        x0=np.nanmean(x)
        sx=np.nanstd(x) +10**(-10)            
        try:
            for col in DataIn.columns:    
                y=DataIn[col].to_numpy()
                reals=~np.isnan(x+y)
                if np.sum(reals)>degree:
                    y0=np.nanmean(y)
                    sy=np.nanstd(y)+10**(-10)   
                    y_det=y+np.nan        
                    coeff=np.polyfit((x[reals]-x0)/sx,(y[reals]-y0)/sy,degree)
                    y_tr=np.polyval(coeff,(x[reals]-x0)/sx)*sy+y0
                    y_det[reals]=y[reals]-y_tr     
                    DataOut[col]=y_det
        except: 
            pass
    return DataOut


def resample_v3(DataIn,period,x1,x2,func):
#07/19/2022 (v 2): added detrend option
#11/14/2022 (v 3): different binning options

    DataOut=pd.DataFrame([])
    for c in DataIn.columns:
        DataOut[c]=np.nan*np.zeros(len(x1))
    
    if period==[]:
        #flexible binning
        x_bin=np.zeros(len(x1)*2)
        x_bin[np.arange(0,len(x_bin),2)]=x1
        x_bin[np.arange(1,len(x_bin),2)]=x2
        index=(np.digitize(DataIn.index,x_bin)-1)/2
        index[index!=np.round(index)]=-1
        DataIn2=DataIn.copy()
        DataIn2.iloc[index==-1,:]=np.nan
    else:
        #fixed-window binning
        index=np.floor(DataIn.index/period)*period+period/2    
        DataIn2=DataIn.copy()
    
    D=DataIn2.groupby(index).apply(func)

    if len(D)==len(index):#if the function applied did not change the length of the dataset (e.g. detrending)
            DataOut=D.copy()
    else:#if the function applied produced a single point per bin (e.g. average)
        DataOut=D.iloc[D.index>=0,:]
        
        if period==[]:
            DataOut.index=(x1+x2)[DataOut.index.astype(int)]/2
        else:
            DataOut.index=(np.unique(index))
            
    return  DataOut

def neighbors(x,xs,max_dx=10):
    if xs<x[0]:
        i1=i2=0
    elif xs>x[-1]:
        i1=i2=-1
    elif np.min(np.abs(x-xs))==0:
        i1=i2=np.argmin(np.abs(x-xs))
    else:
        i1=np.where(xs>=x)[0][-1]
        i2=i1+1
        
    d1=np.abs(xs-x[i1])
    d2=np.abs(xs-x[i2])
    
    if d1>max_dx:
        i1=[]
    if d2>max_dx:
        i2=[]
    return i1,i2

def neighbors_v2(x,xs,max_dx=10):
    #improved cases with max distance violated
    if xs<x[0]:
        i1=i2=0
    elif xs>x[-1]:
        i1=i2=-1
    elif np.min(np.abs(x-xs))==0:
        i1=i2=np.argmin(np.abs(x-xs))
    else:
        i1=np.where(xs>=x)[0][-1]
        i2=i1+1
        
    d1=np.abs(xs-x[i1])
    d2=np.abs(xs-x[i2])
    
    # print(d1)
    
    if d1>max_dx:
        i1=[]
    if d2>max_dx:
        i2=[]
    
    if i1==[] and i2!=[]:
        i1=i2.copy()
    
    if i2==[] and i1!=[]:
        i2=i1.copy()
        
    return i1,i2

def neighbors_v3(x,xs,max_dx=10,extrap=True):
    #improved cases with max distance violated
    #11/06/2023: extrapolation flag
    
    if xs<x[0]:
        if extrap:
            i1=i2=0
        else:
            i1=i2=-9999
    elif xs>x[-1]:
        if extrap:
            i1=i2=-1
        else:
            i1=i2=[]
    elif np.min(np.abs(x-xs))==0:
        i1=i2=np.argmin(np.abs(x-xs))
    else:
        i1=np.where(xs>=x)[0][-1]
        i2=i1+1
        
    if i1>=0 and i2>=0:
        d1=np.abs(xs-x[i1])
        d2=np.abs(xs-x[i2])
    
        if d1>max_dx:
            i1=-9999
        if d2>max_dx:
            i2=-9999
    
    if i1==-9999 and i2!=-9999:
        i1=i2.copy()
    
    if i2==-9999 and i1!=-9999:
        i2=i1.copy()
        
    return i1,i2
def draw_error_band(ax, x, y, err, **kwargs):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))
    
def multi_bar(y,width,colors,**kwargs):
    N=len(y)
    x=np.arange(len(y[0]))*N*width*1.25
    ctr=0
    for yy in y:
        plt.bar(x+(ctr-N/2)*width,yy,width=width,color=colors[ctr],**kwargs)
        ctr+=1
    return x
    
def str_dec(num):
    if len2(num)>1:
        s=''
        for n in num:
            if n!=np.round(n):
                s=s+ str(n)+', '
            else:
                s=s+ str(int(n))+', '
        return s[:-2]
    else:
        if num!=np.round(num):
            return str(num)
        else:
            return str(int(num))
        
        
def axis_equal():
    
    from mpl_toolkits.mplot3d import Axes3D
    ax=plt.gca()
    is_3d = isinstance(ax, Axes3D)
    if is_3d:
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        zlim=ax.get_zlim()
        ax.set_box_aspect((np.diff(xlim)[0],np.diff(ylim)[0],np.diff(zlim)[0]))
    else:
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))

def tms2utc(time):
    from datetime import datetime
    if len2(time)==1:
        time=[time]
    
    return [datetime.utcfromtimestamp(t) for t in time]

def match_arrays(x1,x2,max_diff):
    d=np.abs(vec_diff(x1,x2))
    sel=d<max_diff
    matches=np.array([np.where(sel)[0],np.where(sel)[1]]).T
    
    single1=np.setdiff1d(np.arange(len(x1)),matches[:,0])
    single2=np.setdiff1d(np.arange(len(x2)),matches[:,1])
    matches=vstack(matches,np.array([single1,-999*(single1**0)]).T)
    matches=vstack(matches,np.array([-999*(single2**0),single2,]).T).astype(int)
    
    return matches


def numpy_datetime2datetime(dt64):
    from datetime import datetime
    t=(dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    
    dt=datetime.utcfromtimestamp(t)
    return dt

def nancorr(x1,x2):
    
    reals=~np.isnan(x1+x2)
    
    return np.corrcoef(x1[reals],x2[reals])[0,1]

def rotate_grid(X,Y,theta):
    X_rot=cosd(theta)*X-sind(theta)*Y
    Y_rot=sind(theta)*X+cosd(theta)*Y
    
    return X_rot, Y_rot


def numpy_datetime2num(dt64):
    from datetime import datetime
    t=(dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    
    return t

def draw_turbine(x,y,D,wd):
    import matplotlib.image as mpimg
    from matplotlib import transforms
    from matplotlib import pyplot as plt
    img = mpimg.imread('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions/Turbine5.png')
    ax=plt.gca()
    tr = transforms.Affine2D().scale(D/700,D/700).translate(-100*D/700+x,-370*D/700+y).rotate_deg(90-wd)
    ax.imshow(img, transform=tr + ax.transData)


def gradient2D(x,y,f):
    
    X,Y=np.meshgrid(x,y)
    df_dx=f*0
    df_dx[:,0]=(f[:,1]-f[:,0])/(x[1]-x[0])
    df_dx[:,-1]=(f[:,-1]-f[:,-2])/(x[-1]-x[-2])
    df_dx[:,1:-1]=(f[:,2:]-f[:,0:-2])/(X[:,2:]-X[:,0:-2])
    
    df_dy=f*0
    df_dy[0,:]=(f[1,:]-f[0,:])/(y[1]-y[0])
    df_dy[-1,:]=(f[-1,:]-f[-2,:])/(y[-1]-y[-2])
    df_dy[1:-1,:]=(f[2:,:]-f[0:-2,:])/(Y[2:,:]-Y[0:-2,:])
    
    return df_dx,df_dy


def gradient1D(x,f):

    df_dx=f*0
    df_dx[0]=(f[1]-f[0])/(x[1]-x[0])
    df_dx[-1]=(f[-1]-f[-2])/(x[-1]-x[-2])
    df_dx[1:-1]=(f[2:]-f[0:-2])/(x[2:]-x[0:-2])
    
    return df_dx

def cart2sphere(x,y,z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    beta = np.arcsin(z / rho)*180/np.pi
    theta = np.arctan2(y, x)*180/np.pi
    return rho,theta,beta


def sum_datetime(dt1,dt2):
    from datetime import timedelta
    sum_dt = dt1 + timedelta(years=dt2.year, months=dt2.month, days=dt2.day, hours=dt2.hour, minutes=dt2.minute, seconds=dt2.second)
    
    return sum_dt


def add_gap(x,y,F,gap_x,gap_y,ext_x,ext_y):
    
    diff_x=np.append(np.diff(x)[0],np.diff(x))
    i_gap_x=np.concatenate([[0],np.where(diff_x>gap_x)[0]])
    
    if isinstance(x[0], np.datetime64):
        x2=np.array([],dtype= 'datetime64')
    else:    
        x2=[]
    F2=[]
    
    if ext_x!=0:
        x2=np.append(x2,x[0]-ext_x)
        F2=hstack_v2(F2,np.reshape(F[:,0],(-1,1)))
    
    for i1,i2 in zip(i_gap_x[:-1],i_gap_x[1:]):
        x2=np.append(x2,x[i1:i2])
        F2=hstack_v2(F2,F[:,i1:i2])
        
        if ext_x!=0:
            x2=np.append(x2,x[i2-1]+ext_x)
            F2=hstack_v2(F2,np.reshape(F[:,i2-1],(-1,1)))
        
        x2=np.append(x2,x[i2-1]+diff_x[i2]/2)
        F2=hstack_v2(F2,np.zeros((len(y),1))+np.nan)
        
        if ext_x!=0:
            x2=np.append(x2,x[i2]-ext_x)
            F2=hstack_v2(F2,np.reshape(F[:,i2],(-1,1)))
            
    x2=np.append(x2,x[i_gap_x[-1]:])
    F2=hstack_v2(F2,F[:,i_gap_x[-1]:])
    
    diff_y=np.append(np.diff(y)[0],np.diff(y))
    i_gap_y=np.concatenate([[0],np.where(diff_y>gap_y)[0]])
    
    if isinstance(y[0], np.datetime64):
        y2=np.array([],dtype= 'datetime64')
    else:    
        y2=[]
    F3=[]
    
    if ext_y!=0:
        y2=np.append(y2,y[0]-ext_y)
        F3=vstack(F3,np.reshape(F2[0,:],(1,-1)))
    for i1,i2 in zip(i_gap_y[:-1],i_gap_y[1:]):
        y2=np.append(y2,y[i1:i2])
        F3=vstack(F3,F2[i1:i2,:])
        
        if ext_y!=0:
            y2=np.append(y2,y[i2-1]+ext_y)
            F3=vstack(F3,np.reshape(F2[i2-1,:],(1,-1)))
        
        y2=np.append(y2,y[i2-1]+diff_y[i2]/2)
        F3=vstack(F3,np.zeros((1,len(x2)))+np.nan)
        
        if ext_y!=0:
            y2=np.append(y2,y[i2]-ext_y)
            F3=vstack(F3,np.reshape(F2[i2,:],(1,-1)))
            
    y2=np.append(y2,y[i_gap_y[-1]:])
    F3=vstack(F3,F2[i_gap_y[-1]:,:])
    
    return x2,y2,F3

def dt64_to_num(dt64):
    tnum=(dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return tnum

def num_to_dt64(tnum):
    dt64= np.datetime64('1970-01-01T00:00:00Z')+np.timedelta64(int(tnum*10**9), 'ns')
    return dt64

def fill_nan(x):
    #fills isolated nan with the mean of the neighbors
    #2023/06/21: created, finalized
    x_int=x.copy()
    for i in range(1,len(x)-1):
        if ~np.isnan(x[i-1]) and ~np.isnan(x[i+1]) and np.isnan(x[i]):
            x_int[i]=(x[i-1]+x[i+1])/2
    return x_int

def plot_matrix(data,scale='linear',cmap='coolwarm',vmin=0,vmax=1,textcolor='k',fontsize=16):
    if scale=='log':
        plt.imshow(np.log(data), cmap=cmap,vmin=vmin,vmax=vmax)  # You can choose any colormap you prefer
    else:
        plt.imshow(data, cmap=cmap,vmin=vmin,vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, '{:.2f}'.format(data[i, j]),
                     ha='center', va='center', color=textcolor,fontsize=fontsize)
            
            
def remove_labels(fig):
    axs=fig.axes

    for ax in axs:
        loc=ax.get_subplotspec()
        try:
            
            if loc.is_last_row()==False:
                ax.set_xticks(ax.get_xticks(),[])
                ax.set_xlabel('')
            if loc.is_first_col()==False:
                ax.set_yticks(ax.get_yticks(),[])
                ax.set_ylabel('')
        except:
            pass