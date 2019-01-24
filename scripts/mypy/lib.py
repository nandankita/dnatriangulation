
import string
import glob
import random
import sys
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import itertools as it
import types
import pdb

wrb_cmap=matplotlib.colors.LinearSegmentedColormap.from_list('wrb',colors=['white','red','black'])
wrb_cmap.set_bad([0.82,0.82,0.82])
plt.register_cmap(cmap=wrb_cmap)

worb_cmap=matplotlib.colors.LinearSegmentedColormap.from_list('worb',colors=['white','orange','red',[0.5,0,0],'black'])
worb_cmap.set_bad([0.82,0.82,0.82])
plt.register_cmap(cmap=worb_cmap)

worb2_cmap=matplotlib.colors.LinearSegmentedColormap.from_list('worb2',colors=['white','orange','red',[0.5,0,0],'black'])
worb2_cmap.set_bad([1,1,1])
plt.register_cmap(cmap=worb2_cmap)

dekker_cmap=matplotlib.colors.LinearSegmentedColormap.from_list('dekker',colors=[[1,1,1],[1,0.65,0],[1,0,0],[0.35,0,0]])
dekker_cmap.set_bad([0.82,0.82,0.82])
plt.register_cmap(cmap=dekker_cmap)

bwr_cmap=matplotlib.colors.LinearSegmentedColormap.from_list('bwr',colors=['blue','white','red'])
bwr_cmap.set_bad([0.82,0.82,0.82])
plt.register_cmap(cmap=bwr_cmap)

def heatmap(A,cmap="worb",rotate=False,clip=0,top_half=False,log=False,colorbar=True,chrs=None,vmin=None,vmax=None,imshow_interpolation="none",coords=False):
    '''
    clip_top clips top fraction of data
    top_half shows only top_half of matrix
    '''
    
    if clip>0:
        A=np.clip(A,-np.inf,np.percentile(A[~np.isnan(A)],(1.0-clip)*100))
    if log:
        A=np.log(1+A)
    if rotate:
        rotplot(A,cmap=cmap,mask=top_half,vmin=vmin,vmax=vmax,coords=coords)
    else:
        plt.imshow(A,interpolation=imshow_interpolation,cmap=cmap,vmin=vmin,vmax=vmax)
        if chrs != None:
            for i in range(1,len(chrs)):
                if chrs[i]!=chrs[i-1]:
                    plt.axvline(i-0.5,color="black")
                    plt.axhline(i-0.5,color="black")

    if colorbar:
        plt.colorbar()


def func_reduce_2d(A,keys1=None,keys2=None,func=np.mean):
    '''
    keys1 are row keys
    keys2 are col keys

    if keys1 or keys2 are None, the matching dimension will not be reduced
    '''

    flat=False
    if A.ndim==1:
        flat=True
        A=np.c_[A]
        sys.stderr.write("A is 1D and will be reduced as a column vector\n")
    
    if keys1 is None:
        keys1=np.arange(A.shape[0])
    if keys2 is None:
        keys2=np.arange(A.shape[1])
      
    unique_keys1,keys1_ind=np.unique(keys1,return_inverse=True)
    unique_keys2,keys2_ind=np.unique(keys2,return_inverse=True)
        
    nk1=len(unique_keys1)
    nk2=len(unique_keys2)

    if len(keys1)!=A.shape[0] or len(keys2)!=A.shape[1]:
        sys.exit("number of keys doesnt match array dimensions")
   
    newA=np.zeros((nk1,nk2))
    
    for i,j in it.product(range(nk1),range(nk2)):
        
        newA[i,j]=func(A[np.ix_(keys1_ind==i,keys2_ind==j)])

    if flat:
        newA=newA.flatten()

    return newA
    


def downsample(A,chunksize,axis,func,trim):

    """
    downsamples array A by applying func to chunks of size chunksize along axis.
    trim=True trims A (ignores last chunk) if size of A is not divisible by chunksize
    func should accept an axis argument
    """
    
    A=np.rollaxis(A,axis)
    numchunks=A.shape[0]//chunksize
    
    if trim:
        A=A[:chunksize*numchunks]
        
    shape=(-1,chunksize)+A.shape[1:]
    return np.rollaxis(func(A.reshape(shape),axis=1),0,start=axis+1)

    
# output all pairs of intersecting loci

def intersection_iter(loc1_iter,loc2_iter,posf1,posf2):

    loc2_buffer=[]

    for loc1 in loc1_iter:

        loc1_chr,loc1_start,loc1_end=posf1(loc1)

        if loc1_start>loc1_end:
            sys.exit('loc1 start>end: '+str((loc1_chr,loc1_start,loc1_end)))

        # remove from buffer locations that have been passed

        new_loc2_buffer=[]
        
        for i in loc2_buffer:
            if i!=None:
                i_chr,i_start,i_end=posf2(i)
            if i==None or i_chr>loc1_chr or (i_chr==loc1_chr and i_end>=loc1_start):
                new_loc2_buffer.append(i)
         
        loc2_buffer=new_loc2_buffer

        # add to buffer locations that intersect
        
        while True:

            if len(loc2_buffer)>0:

                if loc2_buffer[-1]==None:
                    break
                
                last_chr,last_start,last_end = posf2(loc2_buffer[-1])
                
                if last_chr>loc1_chr:
                    break

                if last_chr==loc1_chr and last_start>loc1_end:
                    break

            try:

                newloc2=loc2_iter.next()
                
                newloc2_chr,newloc2_start,newloc2_end=posf2(newloc2)

                if newloc2_start>newloc2_end:
                    sys.exit('loc2 start>end: '+str((newloc2_chr,newloc2_start,newloc2_end)))

                # add location to buffer if relevant
                if newloc2_chr==None or newloc2_chr>loc1_chr or (newloc2_chr==loc1_chr and newloc2_end>=loc1_start):
                    loc2_buffer.append(newloc2)
              
            except StopIteration: # if loc2_iter ended

                loc2_buffer.append(None)

        # yield loc1 x loc2_buffer
            
        for loc2 in loc2_buffer[:-1]:
            yield loc1,loc2
            


def random_string(n):
    options = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return ''.join(random.choice(options) for x in range(n))


def args_to_dict(**kwargs):
    return kwargs


def args_to_array(*args):
    return args

def iter_len(it):
    return sum(1 for _ in it)

# resize numpy array, trimming/zero-padding as needed
def hard_resize(A,shape):
    A=A.flatten()
    A.resize(shape)
    return A


def bool_mat_to_int64_array(M):
    bits=64
    length=((M.shape[0]*M.shape[1])//bits)+1
    return np.dot(hard_resize(M,(length,bits)),1<<np.arange(bits,dtype='int64'))


def int64_array_to_bool_mat(A,shape):
    bits=64
    M = (np.c_[A].T & np.c_[1<<np.arange(bits,dtype='int64')]).astype(bool).T
    return hard_resize(M,shape)


    

#blocksize uses row blocks for use with hdf5 matrices (use with return_copy=False)
def balance(A,threshold=1e-5,return_factors=False,blocksize=None,return_copy=True,keep_sum=False):

    sys.stderr.write('balancing, please ensure matrix is symmteric and entries are non-negative')
  
    if return_copy:
        A=A.copy()

    # remeber count

    #################################################blocknansum A and blocknanmean A
    # choo choo

    n=A.shape[0]


    
    if blocksize==None or blocksize>n or blocksize<1:
        blocksize=n
    
    total_sum=0
    for i in range(0,n,blocksize):
        total_sum+=np.nansum(A[i:i+blocksize,:])
    
    
    axis=0
    factors=np.mean(A,axis)
   
    factors_final=np.ones(n,dtype='float64')
    delta=np.max(np.abs(factors-1.))
    
    while(delta>threshold):
 
        if axis==0:
            for i in range(0,n,blocksize):
                A[i:i+blocksize,:]/=factors
            
        else:
            for i in range(0,n,blocksize):
                A[i:i+blocksize,:]/=np.c_[factors[i:i+blocksize]]
    
        factors_final*=factors.ravel()
        axis=1-axis
        
        factors=np.nansum(A,axis)/np.sum(~np.isnan(A))
        
        delta=np.max(np.abs(factors-1))

    if(keep_sum):
        A*=total_sum/n**2
        
    if(return_factors):
        return A,np.sqrt(factors_final)
    else:
        return A


def remove_nan_rowcols(M):
      nan_rowcols=np.c_[(np.sum(np.isnan(M+M.T),0)+np.sum(np.isnan(M+M.T),1))==(2*M.shape[0])]
      valid_rowcols=~nan_rowcols
      n=np.sum(valid_rowcols)

      newM=M[valid_rowcols&valid_rowcols.T].reshape((n,n))
      
      return (newM,nan_rowcols)

def add_nan_rowcols(M,nan_rowcols,fill_value=np.nan):
      n=M.shape[0]+np.sum(nan_rowcols)
      valid_rowcols=~nan_rowcols
      newM=fill_value*np.zeros((n,n),dtype=M.dtype)
      newM[valid_rowcols&valid_rowcols.T]=M.flatten()
      return newM



def scatter_hist(x,y,bins=10,xlabel=None,ylabel=None):

      nullfmt   = matplotlib.ticker.NullFormatter()         # no labels             

# definitions for the axes                                                          
      left, width = 0.1, 0.65
      bottom, height = 0.1, 0.65
      bottom_h = left_h = left+width+0.02

      rect_scatter = [left, bottom, width, height]
      rect_histx = [left, bottom_h, width, 0.2]
      rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure                                                   
      plt.figure()

      axScatter = plt.axes(rect_scatter)
      if(xlabel):
            plt.xlabel(xlabel)
      if(ylabel):
            plt.ylabel(ylabel)
      axHistx = plt.axes(rect_histx)
      axHisty = plt.axes(rect_histy)

# no labels                                                                         
      axHistx.xaxis.set_major_formatter(nullfmt)
      axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:                                                                 
      axScatter.scatter(x,y)

      axHistx.hist(x, bins=bins)
      axHisty.hist(y, bins=bins, orientation='horizontal')

      axHistx.set_xlim( axScatter.get_xlim() )
      axHisty.set_ylim( axScatter.get_ylim() )

      plt.axes(axScatter)


def hist2d_hist(x,y,bins=10,xlabel=None,ylabel=None):

      nullfmt   = matplotlib.ticker.NullFormatter()         # no labels             

# definitions for the axes                                                          
      left, width = 0.1, 0.65
      bottom, height = 0.1, 0.65
      bottom_h = left_h = left+width+0.02

      #rect_scatter = [left, bottom, width, height]
      rect_scatter = [0.025, bottom, 0.8, height]
      rect_histx = [left, bottom_h, width, 0.2]
      rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure                                                   
      plt.figure(figsize=(8,8))

      axScatter = plt.axes(rect_scatter)
      if(xlabel):
            plt.xlabel(xlabel)
      if(ylabel):
            plt.ylabel(ylabel)
      axHistx = plt.axes(rect_histx)
      axHisty = plt.axes(rect_histy)

# no labels                                                                         
      axHistx.xaxis.set_major_formatter(nullfmt)
      axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:    
      h,yedges,xedges=np.histogram2d(y,x,bins=bins)


      extent=(min(xedges),max(xedges),min(yedges),max(yedges))
      aspect=(extent[1]-extent[0])/(extent[3]-extent[2])

      axScatter.imshow(h,interpolation='none',aspect=aspect,extent=extent,origin='lower')
      
      #axScatter.axis(rect_scatter)

      axHistx.hist(x, bins=bins)
      axHisty.hist(y, bins=bins, orientation='horizontal')
      plt.setp(plt.xticks()[1], rotation=45)

      axHistx.set_xlim( axScatter.get_xlim() )
      axHisty.set_ylim( axScatter.get_ylim() )

      plt.axes(axScatter)
      plt.setp(plt.xticks()[1], rotation=45)

# show heatmap rotated by 45 deg
# mask shows top half
# ax = axis
# antialiased=True is currently broken in matplotlib
def rotplot(C,coords=False,mask=False,cmap='worb',vmin=None,vmax=None,marks=None,ax=None,antialiased=False,norm=None):

    if(C.shape[0]!=C.shape[1]):
        sys.exit('damn')

    n=C.shape[0]
  
    if coords is False:
        coords=np.arange(n+1)

    if len(coords) != n+1 :
        sys.exit('coords length should be n+1')

    t=np.array([[1,0.5],[-1,0.5]])
    A=np.dot(np.array([(i[1],i[0]) for i in it.product(coords[::-1],coords)]),t)
    
    if np.any(np.isnan(C)) or mask:
        C=np.ma.array(C,mask=np.isnan(C))
    
    if mask:

        C.mask|=np.tri(n,k=-1,dtype=bool)

    if ax:
        cax=ax
    
        meshobj=cax.pcolormesh(A[:,1].reshape(n+1,n+1),A[:,0].reshape(n+1,n+1),np.flipud(C),antialiased=antialiased,cmap=cmap,vmin=vmin,vmax=vmax,norm=norm)

        if mask:
            cax.set_ylim((0,coords[-1]-coords[0]))

    else:
        plt.pcolormesh(A[:,1].reshape(n+1,n+1),A[:,0].reshape(n+1,n+1),np.flipud(C),antialiased=antialiased,cmap=cmap,vmin=vmin,vmax=vmax,norm=norm)
        
        if mask:
            plt.ylim((0,coords[-1]-coords[0]))

    if(marks!=None): # might not work correctly with coords
        t=np.array([[0.5,1],
                    [-0.5,1]])
        x=np.array(np.nonzero(marks)[1])
        y=n-np.array(np.nonzero(marks)[0])
        xy=np.c_[x,y]
        xyt=np.dot(xy,t)
        xyt[:,0]+=n/2+0.5
        xyt[:,1]-=n

        if ax:
            cax.plot(xyt[:,0],xyt[:,1],'o',color='blue')
        else:
            plt.plot(xyt[:,0],xyt[:,1],'o',color='blue')

    if ax:
        cax.set_xlim((coords[0],coords[-1]))

        return meshobj
    else:
        plt.xlim((coords[0],coords[-1]))
        
    

        # iterate in batches

def batch_iter(iterable,batch_size):
    c=0
    batch=[]
    for i in iterable:
        batch.append(i)
        c+=1
        if c==batch_size:
            c=0
            yield batch
            batch=[]

    if c>0:
        yield batch


def load_matrix(fh,hrows=0,hcols=0,np_dtype='float32',row_block_size=1000,numpy_mode=True,max_rows=None,verbose=False,return_all=False,pad=None):
    """
    load a np.array or a list of lists from a text file handle (but works with any iterator) or filename, more memory efficient than numpy.genfromtxt(), headers are returned as lists of strings
    """

    fh_from_filename=False
    
    if type(fh)==str:
        if (fh=='-'):
            fh=sys.stdin
        else:
            fh=open(fh,'r')
            fh_from_filename=True

    original_fh=fh
        
    # init

    firstline=fh.next()

    fh=it.chain([firstline],fh)
    
    cols=len(firstline.rstrip("\n").split("\t"))
    rows=row_block_size
    if (max_rows!=None and max_rows<row_block_size):
        rows=max_rows

    if(hcols):
        cols-=hcols

   
    if numpy_mode:
        data=np.zeros((rows,cols),dtype=np_dtype)
  
    else:
        data=[]

    header_rows=[[] for i in range(hrows)]

    for i in range(hrows):
        header_rows[i]=fh.next().rstrip("\n").split("\t")[hcols:]
 
    header_cols=[[] for i in range(hcols)]
    
    # fill one line at a time

    prev_cols=-1

    r=0

    if (max_rows==None or r<max_rows):
  
        for i in fh:
            line=i.rstrip("\n").split("\t")

            cols=len(line)-hcols

           # if(cols==0):
           #     sys.exit('no valid columns in input line '+str(r))

            if(prev_cols>-1 and cols!=prev_cols):
                if(pad and cols<prev_cols):
                    line=line+['']*(prev_cols-cols)
                    cols=len(line)-hcols
                else:
                    sys.exit('inconsistent number of columns in input line '+str(r))

            prev_cols=cols

            if numpy_mode:
                try: # if np_dtype does not except '' as a value
                    np.dtype(np_dtype).type('')
                except ValueError:
                    try:
                        np.dtype(np_dtype).type('nan')
                        line=[('nan' if i=='' else i) for i in line] # '' are replaced with 'nan'
                    except ValueError:
                        pass
        
                
            for j in range(hcols):
                header_cols[j].append(line[j])

            if numpy_mode:
                data[r,:]=line[hcols:]

                # enlarge data if needed
                if r==(data.shape[0]-1) and (max_rows==None or r<max_rows-1):
                    data=np.resize(data,(data.shape[0]+row_block_size,cols))
                    rows=data.shape[0]

            else:
                data.append(line[hcols:]) 

            r+=1

            if (max_rows!=None and r>=max_rows):
                break

    rows=r
    
    if numpy_mode:
        data=np.resize(data,(rows,cols))

    if (fh_from_filename):
        original_fh.close()

    if (hcols==1):
        header_cols=header_cols[0]
        
    if (hrows==1):
        header_rows=header_rows[0]

    if(verbose):
        sys.stderr.write("loaded matrix with dimensions ("+str(len(data))+","+str(cols)+")\n")
    
    if (return_all or (hrows and hcols)):
        return data,header_rows,header_cols
    if(hrows):
        return data,header_rows
    if(hcols):
        return data,header_cols

    return data



def nprint(data,fh=sys.stdout,fmt=None,delim="\t",newline="\n",trimerr=True):

    # data are arrays, lists, or iterators (e.g. count(), repeat())
    # fmt is list/tuple of formats
    
    if fmt is None:
        fmt=('',)*len(data)

    fmt=['{'+i+'}' for i in fmt]
  
    dlengths=[]
    for i in data:
        try:
            dlengths += [len(i)]
        except TypeError:
            dlengths += [None]

    maxlength=max(dlengths)
    
    for c,i in enumerate(zip(*data)):
  
        strlist=[]
        for di,d in enumerate(i):
            try:
                if isinstance(d,str):
                    raise TypeError('d is a string')
                    
                strlist += map(fmt[di].format,d)
            except TypeError:
                strlist += [fmt[di].format(d)]

        fh.write(delim.join(strlist)+newline)

    if c+1<maxlength and trimerr==True:
        sys.stderr.write("unequal lists, data trimmed.\n")


def cumsum(A,axis=None,dtype=None,out=None,reverse=False):
    if(reverse==True):
        if(axis==1):
            return np.fliplr(np.cumsum(np.fliplr(A),axis=axis,dtype=dtype,out=out))
        elif(axis==0):
            return np.flipud(np.cumsum(np.flipud(A),axis=axis,dtype=dtype,out=out))
        else:
            sys.exit('reverse=True not supported when axis='+str(axis))
    else:
        return np.cumsum(A,axis=axis,dtype=dtype,out=out)


        
def cumprod(A,axis=None,dtype=None,out=None,reverse=False):
    if(reverse==True):
        if(axis==1):
            return np.fliplr(np.cumprod(np.fliplr(A),axis=axis,dtype=dtype,out=out))
        elif(axis==0):
            return np.flipud(np.cumprod(np.flipud(A),axis=axis,dtype=dtype,out=out))
        else:
            sys.exit('reverse=True not supported when axis='+str(axis))
    else:
        return np.cumprod(A,axis=axis,dtype=dtype,out=out)



# 0 1 2 3
# 1 0 1 2
# 2 1 0 1
# 3 2 1 0...

def distmat(n):
    v=np.c_[range(n)]
    return np.abs(v-v.T)

# returns iqm,iqr
# return_std=True will return iqr/1.349 as an estimate of the std

def interquartile_stats(x,return_std=False):
    x=x[~np.isnan(x)]
    if(len(x)<4):
        sys.stderr.write('interquartile_stats() requires at least 4 data points\n')
        return np.nan,np.nan
        
    percentile25=sp.stats.scoreatpercentile(x,25)
    percentile75=sp.stats.scoreatpercentile(x,75)
    
    interquartile_data=x[(x>=percentile25)&(x<=percentile75)]
    iqm=np.mean(interquartile_data)
    iqr=(percentile75-percentile25)

    if return_std:
        iqr/=1.349
    
    return iqm,iqr


# return an iterator over diagonal indices (or elements) for specified diagonals

# zero is central diagonal

# diags can be given as single diag number, list/np.array of diag numbers, or (mindiag,maxdiag) tuple

# mat can be given as a numpy array (from which shape is taken), a shape tuple, or a single number (for square matrix)

# return_single iterates one cell at a time instead of one diagonal at a time
# return_elements returns the actual elements of mat instead of indices

def diag_iter(diags,mat,return_single=False,return_elements=False):
    
    if type(diags)==int:
        diags=[diags]
    elif type(diags)==tuple:
        diags=xrange(diags[0],diags[1]+1,1)

    if type(mat)==int:
        shape=(mat,mat)
    elif type(mat)==tuple:
        shape=mat
    else:
        shape=mat.shape

        
    if return_elements:
             
        if return_single:
            for j in diags:
               
                startrow = -min(j,0)
                startcol = max(j,0)

                diaglen = min(min(shape[1]-startcol,min(shape)),shape[0]-startrow)
                
                
                for i in xrange(diaglen):
                    
                    yield mat[startrow+i,startcol+i]
                    
        else:
            
            for j in diags:
                
                startrow = -min(j,0)
                startcol = max(j,0)          
                diaglen = min(min(shape[1]-startcol,min(shape)),shape[0]-startrow)
             
                yield mat[startrow+np.arange(diaglen),startcol+np.arange(diaglen)]

    else:
          
        if return_single:
            for j in diags:
               
                startrow = -min(j,0)
                startcol = max(j,0)

                diaglen = min(min(shape[1]-startcol,min(shape)),shape[0]-startrow)
                
                
                for i in xrange(diaglen):
                    
                    yield startrow+i,startcol+i
                    
        else:
            
            for j in diags:
                
                startrow = -min(j,0)
                startcol = max(j,0)          
                diaglen = min(min(shape[1]-startcol,min(shape)),shape[0]-startrow)
             
                yield startrow+np.arange(diaglen),startcol+np.arange(diaglen)


# verbosity: 1 = details why convexity holds/fails; 2 = individual points tested (can be useful for finding a convex subset)

def convexity_check(func,point_generator,n=1000,verbosity=0,acc_threshold=0,func_returns_grad=True,return_failed=False,args=None,midpoint_only=False):
    f=func
    if func_returns_grad:
        f=lambda *x: func(*x)[0]
    
    for i in xrange(n):
        sys.stderr.write(str(i)+"\n")
        x1=point_generator.next()
        x2=point_generator.next()

        r=np.linspace(0.1,0.9,9)
        if midpoint_only:
            r=[0.5]
        for t in r:
            if args is None:
                p1 = f(t*x1+(1-t)*x2)
                p2 = t*f(x1)+(1-t)*f(x2)
            else:
                p1 = f(t*x1+(1-t)*x2,*args)
                p2 = t*f(x1,*args)+(1-t)*f(x2,*args)
            if p1 > p2+acc_threshold:
                if verbosity>=1:
                    details = str(p1)+' = f('+str(t)+'*'+str(x1)+'+'+str(1-t)+'*'+str(x2)+') > '+str(t)+'*f('+str(x1)+')+'+str(1-t)+'*f('+str(x2)+') = '+str(p2)
                    sys.stderr.write('Convexity fails for (x1,x2) = ('+str(x1)+','+str(x2)+')\n')
                    sys.stderr.write(details+'\n')


                if return_failed:
                    return x1,x2
                else:
                    return False

        if verbosity>=2:
            sys.stderr.write('Convexity holds for (x1,x2) = ('+str(x1)+','+str(x2)+')\n')

    if verbosity>=1:

        sys.stderr.write('Convexity holds after checking '+str(n)+' point pairs.\n')
        
    return True


# returns iterator over folds
# nans are ignored (not shuffled)
# if 0<folds<1: folds is the size of the test set
    
def StratifiedCV(labels,folds=3):
    
    valid_labels=labels[~np.isnan(labels)]
    classes=np.unique(valid_labels)

    if len(classes)>2:
        sys.exit('more than 2 classes found!')

    class_count=dict()
    for c in classes:
        class_count[c]=np.sum(valid_labels==c)
    
    if folds>=2:

        r=dict()
        for c in classes:
            r[c]=np.random.permutation(np.arange(class_count[c]))
                
        for f in range(folds):
            train_all=[]
            test_all=[]
            
            for c in classes:

                start=len(r[c])*f/float(folds)
                end=len(r[c])*(f+1)/float(folds)
                test=r[c][start:end]
                train=np.r_[r[c][:start],r[c][end:]]
            
                ctrain=np.nonzero(labels==c)[0][train].tolist()
                ctest=np.nonzero(labels==c)[0][test].tolist()

                train_all+=ctrain
                test_all+=ctest
                
            yield sorted(train_all),sorted(test_all)


            
    elif folds<1 and folds>0:
        train_all=[]
        test_all=[]
        for c in classes:
            r=np.random.permutation(np.arange(class_count[c]))
            test=r[:len(r)*folds]
            train=r[len(r)*folds:]
            
            ctrain=np.nonzero(labels==c)[0][train].tolist()
            ctest=np.nonzero(labels==c)[0][test].tolist()

            train_all+=ctrain
            test_all+=ctest
        yield sorted(train_all),sorted(test_all)
            
    else:
        sys.exit('incorrect number of folds!')
    

# logsumexp() from scipy 0.12

def logsumexp(a, axis=None, b=None):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
    Input array.
    axis : int, optional
    Axis over which the sum is taken. By default `axis` is None,
    and all elements are summed.
    
    .. versionadded:: 0.11.0
    b : array-like, optional
    Scaling factor for exp(`a`) must be of the same shape as `a` or
    broadcastable to `a`.
    
    .. versionadded:: 0.12.0
    
    Returns
    -------
    res : ndarray
    The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
    more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
    is returned.
    
    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2
    
    Notes
    -----
    Numpy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.
    
    Examples
    --------
    >>> from scipy.misc import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    
    With weights
    
    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647
    """
    
    a = np.asarray(a)
    if axis is None:
        a = a.ravel()
    else:
        a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    if b is not None:
        b = np.asarray(b)
        if axis is None:
            b = b.ravel()
        else:
            b = np.rollaxis(b, axis)
        out = np.log(np.sum(b * np.exp(a - a_max), axis=0))
    else:
        out = np.log(np.sum(np.exp(a - a_max), axis=0))
    out += a_max
    return out



# calculates log of dot(x,exp(y))
def logdotexp(x,y,dot_func=np.dot):
    y_max=np.c_[np.max(y,0)]
    return y_max+np.log(dot_func(x,np.exp(y-y_max)))


# calculates dot product of A and B in chunks of A rows
# (useful when A is very large)
# transpose=True will calculate dot product of transpose(A) and B
# blocksize specifies number of rows in each chunk

def blockdot(A,B,blocksize=None,transpose=False):
    if len(A.shape)!=2:
        sys.exit('A must have 2 dimensions')

    if len(B.shape)!=2:
        sys.exit('B must have 2 dimensions')
    
    if transpose:
        if A.shape[0]!=B.shape[0]:
            sys.exit('dimension mismatch')
        AB=np.zeros((A.shape[1],B.shape[1]))
        if blocksize==None:
            blocksize=A.shape[1]
        for i in range(0,A.shape[0],blocksize):
            AB+=np.dot(A[i:i+blocksize,:].T,B[i:i+blocksize,:])

    else:
        if A.shape[1]!=B.shape[0]:
            sys.exit('dimension mismatch')
        AB=np.zeros((A.shape[0],B.shape[1]))
        if blocksize==None or blocksize<1:
            blocksize=A.shape[0]
        for i in range(0,A.shape[0],blocksize):
            AB[i:i+blocksize,:]=np.dot(A[i:i+blocksize,:],B)

    return AB


def check_gradient(f,x0,eps=None,mode="forward",*args):
    '''
    assumes f returns funcval,gradient
    e.g. eps=1e-8
    '''

    if eps is None:
        eps = max((np.max(np.abs(x0))),1)*1e-8
    
    #fval=lambda x,*args: f(x,*args)[0]
    approx_g=approx_gradient(f,x0,eps,mode,*args)
    calc_g=f(x0,*args)[1]
    print ('approximate gradient:',approx_g)
    print ('calculated grad:',calc_g)
    print ('difference:',np.abs(calc_g-approx_g))
    print ('max difference:',np.max(np.abs(calc_g-approx_g)))
    print ('max(difference/abs(approx_g)):',np.nanmax(np.abs(calc_g-approx_g)/np.abs(approx_g)))
                                
def approx_gradient(f,x,eps,mode="forward",*args):
    '''
    uses forward/central finite difference method to approx gradient (central is more accurate, dangerous at bounds)
    note: be careful at bounds...
    assumes f returns funcval,gradient
    e.g. eps=1e-8
    '''
    grad = np.zeros(len(x))
    ei = np.zeros(len(x))
    fx = f(*((x,) + args))[0]
    for k in range(len(x)):
        ei[k] = 1.0
        d = eps * ei
        if mode == "forward":
            grad[k] = (f(*((x + d,) + args))[0] - fx) / d[k]
        elif mode == "central":
            grad[k] = (f(*((x + d,) + args))[0] - f(*((x - d,) + args))[0]) / (2*d[k])
        ei[k] = 0.0
    return grad
    
