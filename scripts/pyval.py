#!/usr/local/bin/python

import numpy as np
import scipy as sp
import matplotlib

import itertools as it
import sys
import time
import sklearn
import pickle
import argparse
import collections
import pdb
import signal

import scipy.stats
import scipy.spatial.distance
import scipy.ndimage
import scipy.signal
import scipy.linalg
import scipy.interpolate
import scipy.optimize
import types


import sklearn.cluster
import sklearn.svm
from hmmlearn import hmm
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.decomposition
import sklearn.feature_selection
import sklearn.manifold

import re
import h5py



def main():

    parser=argparse.ArgumentParser(description='Python command-line wrapper. Imports several modules (scipy.*, sklearn.*, numpy, mypy.lib, matplotlib.pyplot as plt, sys) and preloads data into numpy arrays for easy manipulation. Use npp() to pretty-print numpy arrays. Use list_imports() to see imported modules. Automatically runs pyplot.show() if needed.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d',help='data files (d[] float). use - for stdin.',dest='dfiles',nargs='+',type=str,default=[])
 
    parser.add_argument('-e',help='code to eval. can be given either as a single string or three strings (start_code,main_code,end_code - relevant only for line mode)',dest='code',nargs='+',type=str,default=[''])

    parser.add_argument('-hr',help='row headers (hr[] string)',dest='row_headers',nargs='+',type=int,default=[0])
    parser.add_argument('-hc',help='column headers (hc[] string)',dest='col_headers',nargs='+',type=int,default=[0])
    parser.add_argument('-delim',help='delimiter',dest='delim',type=str,default="\t")
    parser.add_argument('-dtype',help='numpy data type (list data type is always str)',dest='dtype',type=str,default='float32')
    parser.add_argument('-list',help='list mode (d[] will be list of lists instead of a numpy array)',dest='list_mode',action='store_true')
    parser.add_argument('-line',help='line mode (executes code for each line). simultaenously iterates over all files. header rows will be loaded into hr[] once.',dest='line_mode',action='store_true')
    parser.add_argument('-block',help='row block size',dest='row_block_size',type=int,default=1000)
    parser.add_argument('-pad',help='pad shorter lines with blank',dest='pad',action='store_true')
    
    parser.add_argument('-img',help='save last figure to a file, format will be decided by extension',dest='save_img',type=str,default=None)

    parser.add_argument('-dpi',help='figure dpi',dest='dpi',type=int,default=200)
    
    parser.add_argument('-m',help='multiline code. substitutes code semicolons with newlines.',dest='multiline_code',action='store_true')
    parser.add_argument('-p',help='before run, print all code to STDERR',dest='print_code',action='store_true')
    parser.add_argument('-mf',help='embedded makefile mode. trims tabs at starts of lines and spaces and backslash at ends of lines in code.',dest='makefile',action='store_true')
    parser.add_argument('-noterm',help='no terminal connected',dest='no_term',action='store_true')
    
        
    args=parser.parse_args()

    dfiles=args.dfiles
    col_headers=args.col_headers
    row_headers=args.row_headers
    code=args.code
    list_mode=args.list_mode
    line_mode=args.line_mode
    save_img=args.save_img
    dpi=args.dpi
    delim=args.delim
    dtype=args.dtype
    row_block_size=args.row_block_size
    pad=args.pad
    multiline_code=args.multiline_code
    print_code=args.print_code
    makefile=args.makefile
    no_term=args.no_term

    if no_term:
        matplotlib.use('Agg')
        
    import mypy.lib
    import matplotlib.pyplot as plt



    from mypy.lib import nprint
    
    if(len(col_headers)==1 and len(dfiles)>1):
        col_headers=[col_headers[0] for i in dfiles]


    if(len(row_headers)==1 and len(dfiles)>1):
        row_headers=[row_headers[0] for i in dfiles]


    if makefile:
        sys.stderr.write('ttt\n')
        code=[re.sub(r' *\\ *\n\t*','\n',c) for c in code]
        
    if multiline_code:
        code=[c.replace(";","\n") for c in code]

            
    start_code=''
    main_code=''
    end_code=''
    if (len(code)==1):
        main_code=code[0]
	#print("main", main_code)
    elif (len(code)==3):
        start_code=code[0]
        main_code=code[1]
        end_code=code[2]
	#print("start",main_code,start_code)
    else:
        sys.exit('code can be of length 1 or 3')

    if print_code:
        if start_code!='':
            print >> sys.stderr,'# start code\n'+start_code+'\n'
 
        if main_code!='':
            print >> sys.stderr,'# main code\n'+main_code+'\n'
         
        if end_code!='':
            print >> sys.stderr,'# end code\n'+end_code+'\n'
         
            

    if (line_mode): # line mode operates by calling load_matrix() on each line of each file
        
        #sys.exit('line mode not supported!')
        
        fh=[]
        hr=[[] for i in dfiles]
        
        for i,f in enumerate(dfiles):
            if f=='-':
                fh.append(sys.stdin)
            else:
                fh.append(open(f,'r'))

            if (row_headers[i]>0):
  
                hr[i]=mypy.lib.load_matrix(fh[i],hcols=col_headers[i],hrows=row_headers[i],np_dtype=dtype,numpy_mode=(not list_mode),max_rows=0,return_all=True,row_block_size=row_block_size)[1]
  
            if (len(dfiles)==1):
                hr=hr[0]

        exec (start_code)
                
        for line in it.izip(*fh): # iterate over all files jointly
            
            d=[[] for i in dfiles]
            hc=[[] for i in dfiles]
            for i,f in enumerate(dfiles):
                # iter([line[i]]) wraps line[i] with an iterator that has one element,
                # since load_matrix() assumes a file handle iterator

                d[i],_,hc[i]=mypy.lib.load_matrix(iter([line[i]]),hcols=col_headers[i],hrows=0,np_dtype=dtype,numpy_mode=(not list_mode),max_rows=1,return_all=True,row_block_size=row_block_size,pad=pad)
                d[i]=d[i][0]

            if(len(dfiles)==1):
                d=d[0]
                hc=hc[0]

            exec (main_code)

        for f in fh:
            try:
                f.close()
            except AttributeError:
                pass

        exec (end_code)
                    

    else:
    
        d=[[None] for i in dfiles]
        hr=[[None] for i in dfiles]
        hc=[[None] for i in dfiles]

        for i,f in enumerate(dfiles):

            d[i],hr[i],hc[i]=mypy.lib.load_matrix(f,hrows=row_headers[i],hcols=col_headers[i],np_dtype=dtype,numpy_mode=(not list_mode),return_all=True,row_block_size=row_block_size,pad=pad)

        if(len(dfiles)==1):
            d=d[0]
            hr=hr[0]
            hc=hc[0]
	
        exec (start_code)
        exec (main_code)
        exec (end_code)
  


        
    
    if(len(plt.get_fignums())>0):
        
        if(save_img):
            plt.savefig(save_img,dpi=dpi)
        else:
            plt.show()


def list_imports():
    for n,v in globals().items():
        if isinstance(v,types.ModuleType):
            print (n+"\t"+v.__name__)
           

if __name__=="__main__":
      main()
