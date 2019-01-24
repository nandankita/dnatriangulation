

import sys
import numpy as np
import scipy as sp
import multiprocessing as mp
import scipy.optimize
import time

import pdb
import matplotlib.pyplot as plt
import itertools as it
import sklearn.naive_bayes



def load_data_txt(file_txt,remove_nans=False,retain=1,chrs=None):

    '''
    load Hi-C interaction matrix from text file
    
    parameters:

    file_txt: file name. format "chr\tstart\tend\tdata1\tdata2\t..."
    remove_nans: removes nan rows/columns from all returned variables.
    retain: retain every x-th bin.
    chrs: load only these chrs. None mean load all chrs.
    
    returns:
    
    d: data matrix over the selected set of chromosomes.
    bin_chr: list of chr index assignment of each bin.
    bin_position: start and end position of each bin

    '''

    fh=open(file_txt,'r')
    n=len(fh.readline().split("\t"))-3
    fh.seek(0)
    d=np.zeros((n,n),dtype='float32')
    bin_chr=[]
    bin_position=np.zeros((n,2),dtype='float64')
    for i,line in enumerate(fh):
        t=line.rstrip("\n").split("\t")

        bin_chr.append(t[0])
        bin_position[i,:]=t[1:3]
        d[i,:]=t[3:]
    
    fh.close()
    bin_chr=np.array(bin_chr)

    
    if chrs!=None:
    
        relevant_indices=np.any(bin_chr[None].T==chrs,1)

        d=d[:,relevant_indices][relevant_indices,:]
        bin_chr=bin_chr[relevant_indices]
        bin_position=bin_position[relevant_indices,:]
        
    if retain>1:
        
        d=d[::retain,::retain]
        bin_chr=bin_chr[::retain]
        bin_position=bin_position[::retain,:]
        

    if remove_nans:
        valid_rowcols=~(np.sum(np.isnan(d),0)==d.shape[0])
        d=d[:,valid_rowcols][valid_rowcols,:]
        bin_chr=bin_chr[valid_rowcols]
        bin_position=bin_position[valid_rowcols,:]

    return d,bin_chr,bin_position

    
def chr_color_plot(bin_position,bin_chr,predicted_chr,chrs,chr_spacing=10000000):

    '''
    create plot of chromosome prediction vs actual chromosome.

    parameters:

    bin_position: true position of each bin
    bin_chr: true chromosome of each bin
    predicted_chr: predicted chromosome of each bin
    chrs: list of chromosomes
    chr_spacing: space to leave between chromosomes (in bp)

    '''
 
    pos_add=0
    for i,c in enumerate(chrs):
  
        c_bin_position = bin_position[bin_chr==c]
        maxpos = np.max(c_bin_position)
        
        plt.plot([pos_add,maxpos+pos_add],[i,i],color='black')
        
        plt.scatter(c_bin_position+pos_add,predicted_chr[bin_chr==c],color=plt.cm.Paired(i/float(len(chrs))),lw=1,marker='|',s=100)
        pos_add += maxpos+chr_spacing

    plt.xlabel('actual chr')
    plt.ylabel('cluster')
    plt.xlim((0,pos_add))
    plt.ylim((-1,len(chrs)))
   

    
def _translate(a,kv,copy=True):
    
    '''
    translate every value in a using kv for as a map: a[i]=kv[a[i]]
    '''

    a=a.copy()
    
    for i in range(a.shape[0]):
        try:
            a[i]=kv[a[i]]
        except IndexError:
            pass

    return a


def predict_karyotype(d,nchr=1000,pred_nchr=True,transform=None,shuffle=True,seed=None,rand_frac=0.8,rand_n=20):

    '''
    predict chromosome assignments of contigs from distance matrix by clustering.

    parameters:
    
    d: symmetric distance matrix
    nchr: known number of chromosomes or maximal number of chromosomes (for estimation)
    predn_chr: if True, predict the number of chromosomes
    transform: transformation to apply to d
    shuffle: if True, shuffles contigs before prediction and unshuffles after, to avoid order bias
    seed: seed for shuffle
    rand_frac: fraction of data to use for average step length calculation
    rand_n: number of iterations for average step length calculation

    returns:

    if predn_chr:
        clust_assign, Z, nchr, mean_step_len
    else:
        clust_assign, Z

    clust_assign: vector with cluster assignments
    Z: clustering tree as returned by scipy.cluster.hierarchy.linkage()
    nchr: estimated number of chromosomes
    mean_step_len: mean clustering step length
    
    '''

    
    prng=np.random.RandomState(seed)
    
    if shuffle:
        perm=prng.permutation(d.shape[0])
        inv_perm=np.argsort(perm)
        d=d[perm,:][:,perm]

    if transform:
        d=transform(d)
    
    n=d.shape[0]
    
    Z=sp.cluster.hierarchy.linkage(d[np.triu_indices(n,1)],method='average')

    if pred_nchr:
        dZ=[]
        prng=np.random.RandomState(seed)
        
        for i in range(rand_n):
            random_set=prng.permutation(n)[:n*rand_frac]
            iZ=sp.cluster.hierarchy.linkage(d[random_set,:][:,random_set][np.triu_indices(int(n*rand_frac),1)],method='average')
            dZ.append(np.diff(iZ[:,2]))

        mean_step_len=np.mean(np.array(dZ),0)

        maxnumchr=nchr

        nchr=np.argmax(mean_step_len[::-1][:maxnumchr])+2
            
    clust_assign=sp.cluster.hierarchy.fcluster(Z,t=nchr,criterion='maxclust')
    
    if shuffle:
        Z[:,0]=_translate(Z[:,0],perm)
        Z[:,1]=_translate(Z[:,1],perm)
        clust_assign=clust_assign[inv_perm]
    
    clust_assign-=1

    if pred_nchr:
        return clust_assign, Z, nchr, mean_step_len
    else:
        return clust_assign, Z
 



def _ALP_fit_scale_Q(scale,abs_dists,interactions,sia,return_grad=True):

    logsumexp_scaled_abs_dists=sp.misc.logsumexp(scale*abs_dists)

    fval= -(scale*sia-logsumexp_scaled_abs_dists)

    if return_grad:

        grad= -(sia - np.exp(sp.misc.logsumexp(scale*abs_dists,b=abs_dists))/np.exp(logsumexp_scaled_abs_dists))

        return fval,np.array([grad])

    else:

        return fval


     

def _ALP_fit_loc_Q(loc,scale,positions,interactions):

    scaled_abs_dists=scale*np.abs(positions-loc)

    f = -(np.sum(interactions*scaled_abs_dists)-sp.misc.logsumexp(scaled_abs_dists))

    return f


    
class AugmentationLocPredModel:

    def __init__(self,sfactor=1e6):

        '''
        initialize model

        parameters:

        sfactor: positions will be internally divided by this number in order to avoid numerical errors during optimization (transparent for user).

        '''
        
        self.sfactor=sfactor
        self.scale=None
 

    def estimate_scale(self,pos,interactions):

        '''
        estimate scale parameter from interaction matrix of contigs of known positions.

        parameters:

        pos: known positions of contigs
        interactions: interaction matrix
        
        '''
        
        if pos.ndim!=1:
            sys.exit('pos.ndim!=1')
            
        if interactions.ndim!=2:
            sys.exit('interactions.ndim!=2')
        
        pos=pos/self.sfactor
        
        abs_dists=np.abs(pos[None].T-pos)
        
        interactions=interactions/np.sum(interactions)

        sia=np.sum(interactions*abs_dists)

        self.scale=sp.optimize.fmin_l_bfgs_b(_ALP_fit_scale_Q,x0=-1e-3,args=(abs_dists,interactions,sia),bounds=[(None,0)],approx_grad=False)[0][0]
                                                                                               
    def estimate_position(self,positions,interactions,x0_iter):
        
        '''
        estimate contig position given estimated scale parameter and interaction profile with contigs of known positions.

        parameters:

        positions: known positions of contigs
        interactions: interaction profile
        x0_iter: iterator that provides initialization points for the optimizer
        
        returns:

        estimated_position
        
        '''
        
        if positions.ndim!=1:
            sys.exit('positions.ndim!=1')
            
        if interactions.ndim!=1:
            sys.exit('interactions.ndim!=1')
            
        interaction_sum=np.sum(interactions)
            
        interactions=interactions/interaction_sum

        positions=positions/self.sfactor
            
        best=[[np.nan],np.inf]
                
        if interaction_sum>10:
                
            res=[sp.optimize.fmin_l_bfgs_b(_ALP_fit_loc_Q,x0=x0/self.sfactor,args=(self.scale,positions,interactions),approx_grad=True) for x0 in x0_iter]

            for i in res:
                if i[1]<best[1]:
                    best=i

            best[0][0]=np.clip(best[0][0],np.min(positions),np.max(positions))*self.sfactor
                        
        return best[0][0]


class AugmentationChrPredModel:

    def __init__(self,model=sklearn.naive_bayes.MultinomialNB(alpha=0,fit_prior=True)):
        self.model=model

    def fit(self,data,labels):
        '''
        train model with average interaction frequencies for each chromosome

        parameters:

        data: (m,n) matrix of m contigs with n average interaction frequencies each
        labels: label (chromosome) each contig belongs to
        
        '''

        return self.model.fit(data,labels)

    def predict(self,data):
        '''
        predict chromosome given average interaction frequencies for each chromosome

        parameters:

        data: (m,n) matrix of m contigs with n average interaction frequencies each
        
        returns:

        predicted_chr,prob
        
        '''
        return self.model.predict(data),np.max(self.model.predict_proba(data),1)

        

def average_reduce(A,keys):
    return func_reduce(A,keys,func=np.mean)
    
              
def average_reduce_2d(A,keys):
    return average_reduce(average_reduce(A,keys).T,keys).T

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
    
        
def func_reduce(A,keys,func,allkeys=None):

    '''
    reduces along first dimension by aggregating rows with the same keys.
    new row order will be sorted by keys, i.e. given by: np.unique(keys)
    '''
    
    unique_keys=np.unique(keys)
    if allkeys==None:
        allkeys=unique_keys
    newshape=(len(allkeys),) + A.shape[1:]
    newA=np.zeros(newshape,dtype=A.dtype)
    for i,k in enumerate(allkeys):
        indices = (keys==k)       
        newA[i]=func(A[indices],axis=0)
            
    return newA

                          
  
def _assemble_chromosome_Q2(x,data,approx_grad):

    '''
    Q function for assemble_chromosome
    '''
    
    scale=x[0]
    pos=x[1:]
    
    dists=pos[None].T-pos
    
    abs_dists=np.abs(dists)

    scaled_abs_dists= scale*np.abs(dists)

    logsumexp_scaled_abs_dists=sp.misc.logsumexp(scaled_abs_dists)
    
    fval= -(np.sum(data*(scaled_abs_dists))-np.sum(data)*logsumexp_scaled_abs_dists)

    if approx_grad:

        return fval

    else:

        # gradient calculation assumes that data matrix is symmetric
    
        exp_scale_abs_dists = np.exp(scaled_abs_dists)
    
        grad_a = - ( np.sum(data*abs_dists) - np.sum(data)*np.sum(abs_dists*exp_scale_abs_dists) / np.sum(exp_scale_abs_dists) )

        grad_s = - ( 2*scale*np.sum(data*np.sign(dists),1) - np.sum(data)*2*scale*np.sum(np.sign(dists)*exp_scale_abs_dists,1) / np.sum(exp_scale_abs_dists) )
    
        return fval,np.r_[grad_a,grad_s]


def _assemble_chromosome_Q(x,data,approx_grad):

    '''
    Q function for assemble_chromosome
    '''

    # assumes data is normalized!!!
    
    scale=x[0]
    pos=x[1:]
    eps=1e-2
    n=pos.shape[0]

    dists=pos-pos[None].T
    
    absdists = np.abs(dists)+eps

    absdists[np.tril_indices(n)] = np.nan

    valid = np.triu(np.ones(n,dtype=bool),1)

    absdists_v = absdists[valid]

    d_v = data[valid]
      
    log_absdists_v = np.log(absdists_v)
    
    logZ = sp.misc.logsumexp(scale*log_absdists_v)
    
    fval = - ( scale*np.sum(d_v*log_absdists_v) - logZ )
    
    if approx_grad:

        return fval

    else:
        
        # gradient calculation assumes that data matrix is symmetric

        log_absdists = np.log(absdists)

        scaled_sign_dists = scale*np.sign(dists)

        grad_a = - ( np.sum(d_v*log_absdists_v) -  np.sum(log_absdists_v*np.exp(scale*log_absdists_v)) / np.exp(logZ) )

        W = np.nan_to_num(data*scaled_sign_dists / absdists) #data is not upper tri, W is upper tri due to abs dists

        H = np.nan_to_num( (np.exp(scale*log_absdists) * scaled_sign_dists) / absdists)
        
        grad_s = - ( (np.sum(W,0) - np.sum(W,1)) - ( np.sum(H,0) - np.sum(H,1) ) / np.exp(logZ) )
    
        return fval,np.r_[grad_a,grad_s]

    

def assemble_chromosome(d,pnum=1,iterations=1,shuffle=True,shuffle_seed=None,init_seed=None,log_data=True,scale_init=None,scale_init_min=-10,lbfgs_factr=1e4,lbfgs_pgtol=1e-9,return_all=False,approx_grad=False,lbfgs_show=False,swapper=False,known_scale=None,known_pos=None,binsize=40000):
    
    '''
    predicts chromosomal positions of contigs from distance matrix.

    parameters:

    d: interaction matrix
    pnum: number of processes to use
    iterations: number of times to rerun LBFGS
    shuffle: shuffle row/column order to avoid ordering bias
    shuffle_seed: seed for shuffle
    init_seed: seed for rand x0 initialization
    log_data: use d=log(d+1)
    scale_init: initialization value for scale parameter (otherwise random initialization)
    scale_init_min: minimum initialization value for scale parameter
    lbfgs_factr: see lbfgs documentation
    lbfgs_pgtol: see lbfgs documetnation
    return_all: return results (scale,pos,x0,fval) for all runs instead of only for best run
    approx_grad: approximate the gradient instead of calculating it
    lbfgs_show: show lbfgs output
    
    returns:
    
    scale: scale parameter
    pos: predicted positions
    x0: initialization
    fval: value of Q
    
    '''

    orig_d=d
    d=d.copy()
    
    if shuffle:
        prng=np.random.RandomState(shuffle_seed)
        perm=prng.permutation(d.shape[0])
        inv_perm=np.argsort(perm)
        d=d[perm,:][:,perm]
    else:
        sys.stderr.write("Warning: Running in unshuffled mode!\n")
    
    n=d.shape[0]
    if log_data:
        d=np.log(d+1)
 
    #d/=np.sum(d)
 
    
    res=[]
    jobs=[]
    
    if pnum>1:
        pool=mp.Pool(processes=pnum)

    max_x = 1.0 #float(binsize*n*2)

    prng=np.random.RandomState(init_seed)
    x0=prng.rand(iterations,n+1)

    x0[:,1:] *= max_x

    x0[:,0] = -1.0

    if not shuffle:
        sys.stderr.write('not shuffled, trying to init x0=input order')
        x0[0,1:]=max_x*np.arange(n)/float(n)

    bounds=[(-1.05,-0.95)]+[(0,max_x)]*n

    d = d / np.sum(np.triu(d,1))
    
    st_time=time.time()
    for i in range(iterations):
        #'L-BFGS-B'/'TNC'/'SLSQP'
        parameters_dict={'fun':_assemble_chromosome_Q,'method':'L-BFGS-B','x0':x0[i,:],'args':(d,approx_grad),'jac':not(approx_grad),'bounds':bounds,'options':{'gtol':lbfgs_pgtol,'factr':lbfgs_factr}}

        if known_scale is not None:
            parameters_dict['x0'][0]=known_scale
            ########### known_pos issue because of shuffle!!!!!!!!!!
        if known_pos is not None:
            known_pos=np.array(known_pos)
            parameters_dict['x0'][np.r_[False,~np.isnan(known_pos)]]=known_pos[~np.isnan(known_pos)]

        if pnum>1:
            
            jobs.append(pool.apply_async(sp.optimize.minimize,kwds=parameters_dict))
            
        else:
            if (lbfgs_show):
                parameters_dict['options']={'disp':True}

            res.append(sp.optimize.minimize(**parameters_dict))
    
        sys.stderr.write('optimization '+str(i+1)+'/'+str(iterations)+' started.\n')

            
    if pnum>1:
        pool.close()
        pool.join()

        for i in range(iterations):
            res.append(jobs[i].get())

    sys.stderr.write('optimizations complete.\n')
    sys.stderr.write('time: '+str(time.time()-st_time)+"\n")
    
    sorted_i=np.argsort([i['fun'] for i in res])

    fvals=np.array([res[i]['fun'] for i in sorted_i])
    scales=np.array([res[i]['x'][0] for i in sorted_i])
    pos=np.array([res[i]['x'][1:] for i in sorted_i])
    x0=x0[sorted_i,:]
    
    if shuffle:
        pos=pos[:,inv_perm]
        x0[:,1:]=x0[:,1:][:,inv_perm]

    fval=fvals[0]
    scale=scales[0]
    pos=pos[0]
    x0=x0[0]


    if swapper:
        #swapper
        sys.stderr.write("swapper (ignores known_pos)\n")
        #sys.stderr.write("estimated swapper time: ~30000 swaps per min, so for this matrix ~"+str(n**2/30000)+" mins per full swap round\n")
        fun=_assemble_chromosome_Q
        p2=np.array(pos)

        orig_d=orig_d/np.sum(np.triu(orig_d,1))
        swap_c=0
        sys.stderr.write("pre-swap score: "+str(fun(np.r_[scale,p2],orig_d,True))+"\n")

        st_time=time.time()
        improved=True
        best_score=fun(np.r_[scale,p2],orig_d,True)
        while improved:
            sys.stderr.write("swapper round "+str(swap_c)+"; time="+str(time.time()-st_time)+"\n")
            swap_c+=1
            improved=False
            for i in np.arange(n):
                for j in np.arange(n):
                    p2_swapped=np.array(p2)
                    p2_swapped[i],p2_swapped[j] = p2[j],p2[i]
                    swapped_score=fun(np.r_[scale,p2_swapped],orig_d,True)
                    if swapped_score<best_score:
                        #sys.stderr.write("swapped "+str(i)+","+str(j)+"\n")
                        improved=True
                        best_score=swapped_score
                        p2=p2_swapped

            sys.stderr.write("swapped score: "+str(fun(np.r_[scale,p2],orig_d,True))+"\n")

        pos = np.array(p2)
        fval = fun(np.r_[scale,p2],orig_d,True)
  
        sys.stderr.write("final opt\n")
        parameters_dict['x0']=np.r_[scale,p2]
        parameters_dict['args']=(orig_d,approx_grad)
        res=sp.optimize.minimize(**parameters_dict)
        fval=res['fun']
        pos=res['x'][1:]
        scale=res['x'][0]
        
        sys.stderr.write("final score: "+str(fval)+"\n")

        sys.stderr.write("returned x0 is not correct because of swapper\n")
        
    returns=(scale,pos,x0,fval)
        
    return returns


        
