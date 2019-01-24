
import argparse
import scipy as sp
import scipy.cluster
import sys
import collections
import triangulation
import numpy as np
import matplotlib.pyplot as plt
import pdb
from mypy.lib import nprint

def main():
    
    parser=argparse.ArgumentParser(description='De novo karyotyping of Hi-C data.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-in',help='Hi-C interaction matrix input file',dest='infile',type=str,required=True)
    parser.add_argument('-out',help='prefix for output files',dest='outfile',type=str,required=True)
    parser.add_argument('-nchr',help='number of chromosomes/clusters. 0 will automatically estimate this number.',dest='nchr',type=int,default=0)
    parser.add_argument('-drop',help='leaves every nth bin in the data, ignoring the rest. 1 will use whole dataset.',dest='drop', type=int,default=1)
    parser.add_argument('-ci',help='list of chromosomes/contigs to include. If empty, uses all chromosomes.',dest='included_chrs',nargs='*',type=str,default=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15','chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22','chrX'])
    parser.add_argument('-s',help='seed for randomizations',dest='seed',type=int,default=0)
    parser.add_argument('-f',help='fraction of data to use for average step length calculation',dest='rand_frac',type=float,default=0.8)
    parser.add_argument('-n',help='number of iterations for average step length calculation',dest='rand_n',type=int,default=20)
    parser.add_argument('-e',help='evaluation mode. chromosome names are assumed to be the true chromosomal assignment.',dest='evaluate',action='store_true')
    parser.add_argument('-minnumchr',help='minimum number of chromosomes',dest='minnumchr',type=int,default=2)
    parser.add_argument('-maxnumchr',help='maximum number of chromosomes',dest='maxnumchr',type=int,default=1000)
    parser.add_argument('-p',help='number of processors to use',dest='pnum',type=int,default=1)
    parser.add_argument('-pool',help='pool interactions for all contigs which share the same name by averaging',action='store_true')
    
    args=parser.parse_args()
      
    infile=args.infile
    outfile=args.outfile
    nchr=args.nchr
    drop=args.drop
    included_chrs=args.included_chrs
    seed=args.seed
    rand_frac=args.rand_frac
    rand_n=args.rand_n
    evaluate=args.evaluate
    minnumchr=args.minnumchr
    maxnumchr=args.maxnumchr
    pnum=args.pnum
    pool=args.pool

    if len(included_chrs)==0:
        included_chrs=None

    d,bin_chr,bin_position=triangulation.load_data_txt(infile,remove_nans=True,chrs=included_chrs,retain=drop)

    sys.stderr.write("loaded "+str(bin_chr.shape[0])+" contigs\n")

    if pool:
        d = triangulation.func_reduce_2d(d,keys1=bin_chr,keys2=bin_chr,func=np.mean)
        bin_position = np.c_[triangulation.func_reduce_2d(bin_position,keys1=bin_chr,func=np.min)[:,0], triangulation.func_reduce_2d(bin_position,keys1=bin_chr,func=np.max)[:,1]]
        bin_chr = np.unique(bin_chr)
        
        sys.stderr.write("pooled to "+str(bin_chr.shape[0])+" contigs\n")

    transform=lambda x: np.log(np.max(x+1))-np.log(x+1)
    
    pred_nchr=False
    if nchr==0:
        nchr=(minnumchr,maxnumchr)
        pred_nchr=True
    
    n=d.shape[0]

    sys.stderr.write("karyotyping...")
    res=triangulation.predict_karyotype(d,nchr=nchr,pred_nchr=pred_nchr,transform=transform,shuffle=True,seed=seed,rand_frac=rand_frac,rand_n=rand_n)
    sys.stderr.write("done.\n")
    
    if pred_nchr:
        clust,Z,nchr,mean_step_len=res
        
        maxval=mean_step_len[-nchr+1]
        msl=len(mean_step_len)

        np.savetxt(outfile+'_avg_step_len.tab',np.c_[np.arange(msl+1,1,-1),mean_step_len],fmt='%s',delimiter='\t')

        plt.figure(figsize=(15,5))
        plt.plot(np.arange(msl+1,1,-1),mean_step_len,marker='o',color='b')
        plt.plot(nchr,maxval,marker='o',color='r')
        plt.gca().invert_xaxis()
        plt.xlabel('number of clusters')
        
        plt.vlines(minnumchr,0,maxval,color='r')
        plt.vlines(maxnumchr,0,maxval,color='r')

        plt.savefig(outfile+'_avg_step_len.png',dpi=600,format='png')
        
        plt.xlim(min(msl,nchr+30),max(0,nchr-30))
        plt.ylim(0,maxval*1.1)
        plt.savefig(outfile+'_avg_step_len_zoomed.png',dpi=600,format='png')
        
        sys.stderr.write("identified "+str(nchr)+" chromosomes.\n")
     


    else:
        clust,Z = res

        
    np.savetxt(outfile+'_clusteringZ.tab',Z,fmt='%s',delimiter='\t')
    
    with open(outfile+'_clusters.tab','w') as fh:
        nprint([bin_chr,bin_position.astype('int'),clust.astype('int')],fh=fh)

                    

    if evaluate:

        # match each cluster to the chromosome which most of its members belongs to
        
        chr_order=dict( zip(included_chrs,range(len(included_chrs))) )
      
        new_clust=np.zeros(n,dtype=bin_chr.dtype)
        new_clust_num=np.nan*np.ones(n)

        for i in range(nchr):

            new_clust[clust==i]=collections.Counter(bin_chr[clust==i]).most_common(1)[0][0]
            new_clust_num[clust==i]=chr_order[collections.Counter(bin_chr[clust==i]).most_common(1)[0][0]]

        sys.stderr.write("accuracy: "+str(np.sum(new_clust==bin_chr)/float(n))+"\n")

        plt.figure(figsize=(15,5))

        triangulation.chr_color_plot(np.mean(bin_position,1),bin_chr,new_clust_num,included_chrs)   

        plt.savefig(outfile+'_evaluation.png',dpi=600,format='png')
       
        with open(outfile+'_evaluation.tab','w') as fh:
            nprint([bin_chr,bin_position.astype('int'),new_clust.astype('int')],fh=fh)

                

if __name__=="__main__":
    main()

    
