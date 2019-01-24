
import sys
import numpy as np
import triangulation
import argparse
import matplotlib.pyplot as plt
import pdb
from mypy.lib import nprint
from random import randint

def main():
    
    parser=argparse.ArgumentParser(description='Scaffold chromosome de novo from contig interaction matrix.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-in',help='interaction frequency matrix file',dest='in_file',type=str,required=True)
    parser.add_argument('-out',help='out file prefix',dest='out_file',type=str,required=True)
    parser.add_argument('-it',help='number of times to rerun L-BFGS',dest='iterations',type=int,default=1)
    parser.add_argument('-p',help='number of processors to use',dest='pnum',type=int,default=0)
    parser.add_argument('-seed',help='seed for L-BFGS init',dest='init_seed',type=int,default=0)
    parser.add_argument('-shuffle_seed',help='seed for shuffle',dest='shuffle_seed',type=int,default=0)
    parser.add_argument('-realpos',help='file with actual contig positions (sorted same as interaction matrix). "contig\tstart\tend"',dest='realposfile',type=str,default=None)
    parser.add_argument('-best',help='sort by original positions to estimate best solution',dest='sort_by_realpos',action='store_true')
    parser.add_argument('-drop',help='leaves every nth bin in the data, ignoring the rest. 1 will use the whole dataset',dest='drop',type=int,default=1)
    parser.add_argument('-keep_unreal',help='keep contigs for which real position is not known',dest='keep_unreal',action='store_true')
    
    parser.add_argument('-lbfgs_pgtol',help='pgtol for lbfgs',dest='lbfgs_pgtol',type=float,default=1e-7)
    parser.add_argument('-lbfgs_factr',help='factr for lbfgs',dest='lbfgs_factr',type=float,default=1e4)
    parser.add_argument('-lbfgs_show',help='show lbfgs iterations (only with pnum=1)',dest='lbfgs_show',action='store_true')

    
    args=parser.parse_args()
      
    in_file=args.in_file
    out_file=args.out_file
    pnum=args.pnum
    iterations=args.iterations
    init_seed=args.init_seed
    shuffle_seed=args.shuffle_seed
    sort_by_realpos=args.sort_by_realpos
    drop=args.drop
    lbfgs_pgtol=args.lbfgs_pgtol
    lbfgs_factr=args.lbfgs_factr
    lbfgs_show=args.lbfgs_show
   
    realposfile=args.realposfile
    keep_unreal=args.keep_unreal

    sys.stderr.write("loading interactions from "+in_file+" ...\n")
      
    d,bin_chr,bin_position=triangulation.load_data_txt(in_file,retain=drop,remove_nans=False)
    nan_bins = np.all(np.isnan(d),1)
    d=d[:,~nan_bins][~nan_bins,:]
    bin_chr=bin_chr[~nan_bins]
    bin_position=bin_position[~nan_bins]
    

    sys.stderr.write("loaded matrix with "+str(d.shape[0])+" contigs.\n")

    if d.shape[0]==0:
        sys.exit('empty dataset')
        

    if realposfile!=None:

        sys.stderr.write("loading real positions from "+realposfile+" ...\n")
    
        contig_pos_dict={}
        with open(realposfile,"r") as fh:
            for line in fh:

                c_name,c_start,c_end = line.rstrip("\n").split("\t")[:3]
                contig_pos_dict[c_name] = (float(c_start),float(c_end))

        realpos=np.array([contig_pos_dict.get(i,(np.nan,np.nan)) for i in bin_chr])

        #realpos=realpos[:,0]+np.mean(bin_position,1)
        realpos=realpos[:,0]
        
        if not keep_unreal:
            sys.stderr.write("removing contigs without real positions...\n")
        
            relevant = ~np.isnan(realpos)
            realpos=realpos[relevant]
            d=d[relevant,:][:,relevant]
            bin_chr=bin_chr[relevant]

            sys.stderr.write(str(d.shape[0])+" contigs left.\n")


            
    # average contigs that share the same id

    sys.stderr.write("averaging contigs that share the same id...\n")
    
    #d=triangulation.func_reduce_2d(d,bin_chr)
    
    #if realposfile!=None:
    #    realpos=triangulation.func_reduce_2d(realpos,bin_chr)

    bin_chr=np.unique(bin_chr)
        
    sys.stderr.write(str(d.shape[0])+" contigs left.\n")

        
    shuffle=True
    if (sort_by_realpos):
        if realposfile==None:
            sys.exit('-best requires -realpos')
        if np.any(np.isnan(realpos)):
            sys.exit('-best requires real positions to be given for ALL contigs')
        
        rr=np.argsort(realpos)
        realpos=realpos[rr]
        d=d[rr,:][:,rr]
        bin_chr=bin_chr[rr]
        shuffle=False


    sys.stderr.write("scaffolding "+str(d.shape[0])+" contigs ...\n")

    for counter in range(1, 2):
        print("starting loop number  ",counter)
        init_seed = randint(0, 100)
        #init_seed = counter
        print("init_seed value for this loop ",init_seed)
        #scales,pos,x0,fvals=triangulation.assemble_chromosome(d,pnum=pnum,iterations=iterations,shuffle=shuffle,return_all=True,shuffle_seed=shuffle_seed,init_seed=init_seed,log_data=True,lbfgs_factr=lbfgs_factr,lbfgs_pgtol=lbfgs_pgtol,approx_grad=False,lbfgs_show=lbfgs_show)
        scales,pos,x0,fvals=triangulation.assemble_chromosome(d=d,pnum=pnum,iterations=iterations,log_data=False,approx_grad=False,shuffle=True,shuffle_seed=0,init_seed=init_seed,lbfgs_show=True)
        if (counter == 1):
            lowest_fvals = fvals
            best_solution = pos
        else: 
            if(fvals<lowest_fvals):
                lowest_fvals = fvals
                best_solution = pos
        
    #print(best_solution,"\n",lowest_fvals)
    print("saving with minimum score ", lowest_fvals) 
    sys.stderr.write("saving results ...\n")
        
    if realposfile!=None:
        with open(out_file+'_predpos.tab','w') as fh:
            nprint([bin_chr,realpos.astype('int'),best_solution[0,:]],fh=fh)
       
    else:
        with open(out_file+'_predpos.tab','w') as fh:
            nprint([bin_chr,best_solution],fh=fh)
        
    np.savetxt(out_file+'_pos_all.tab',best_solution,fmt='%s',delimiter='\t')

    np.savetxt(out_file+'_x0_all.tab',x0,fmt='%s',delimiter='\t')
        
    #np.savetxt(out_file+'_fvals_all.tab',lowest_fvals,fmt='%s',delimiter='\t')

    #np.savetxt(out_file+'_scales_all.tab',scales,fmt='%s',delimiter='\t')

    sys.stderr.write("done.\n")
    



if __name__=="__main__":
    main()

    
