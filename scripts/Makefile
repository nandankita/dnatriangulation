
SHELL = zsh

BINSIZE = 200000


data:
	cat Data/test-200kb.matrix | \
	sed 1d | \
	sed 's/|/\t/g' | \
	cut -f 3- | \
	sed 's/:/\t/' | \
	sed 's/-/\t/' \
	> Data/HiC_$(BINSIZE).tab

	cat Data/HiC_$(BINSIZE).tab | \
	cut -f 1 | \
        sort | \
	uniq -c | \
	perl -lane 'print $$F[1],"\t",$$F[0]' \
	> Data/Bincount_$(BINSIZE).tab


large_scaffolds:
	python pyval.py -d Data/Bincount_$(BINSIZE).tab Data/HiC_$(BINSIZE).tab -hc 1 3 -e 'x=dict(zip(hc[0],d[0].flatten()));v=np.array([x[i]>=1 for i in hc[1][0]]);nprint([np.array(hc[1])[:,v].T,d[1][v,:][:,v]])' \
	> Data/HiC_$(BINSIZE)_large.tab


karyotype:
	mkdir -p Output/Karyotype

	python karyotype.py -in Data/HiC_$(BINSIZE)_large.tab -out Output/Karyotype/Karyotype_$(BINSIZE)_large -ci -n 100 -f 0.9 -s 0 -p 10


.PHONY: scaffold
.NOTPARALLEL: scaffold

scaffold: $(foreach c, init $(shell cat Output/Karyotype/Karyotype_$(BINSIZE)_large_clusters.tab | cut -f 4 | sort | uniq -c | sed 's/  */\t/g' | sort -k 2,2nr | cut -f 3), scaffold_$(c) )


scaffold_init:
	mkdir -p Output/Scaffold

	-rm Output/Scaffold/Scaffold_VCs_$(BINSIZE)_predpos.tab

scaffold_%: c=$*

scaffold_%:
	# filter data
	cat Output/Karyotype/Karyotype_$(BINSIZE)_large_clusters.tab | \
	awk '$$4==$(c)' | \
	cut -f 1,2,3 | \
	sed 's/\t/__/g' | \
	python pyval.py -d - =(sed -e 's/\t/__/' -e 's/\t/__/' Data/HiC_$(BINSIZE)_large.tab) -hc 1 1 -e 's=set(hc[0]);v=np.array([(i in s) for i in hc[1]]);nprint([np.array(hc[1])[v],d[1][:,v][v,:]])' | \
	perl -lane '@x=split "__",$$F[0];print $$F[0],"\t1\t",$$x[2]-$$x[1]+1,"\t",join("\t",@F[1..$$#F])' \
	> Output/Scaffold/Data_VCs_$(BINSIZE)_$(c).tab

	# scaffold
	python3 chromosome_scaffold.py -in Output/Scaffold/Data_VCs_$(BINSIZE)_$(c).tab -out Output/Scaffold/Scaffold_VCs_$(BINSIZE)_$(c) -it 1000 -p 15

	cat Output/Scaffold/Scaffold_VCs_$(BINSIZE)_$(c)_predpos.tab | \
	sort -k 2,2g | \
	sed -e 's/$$/\t$(c)/' \
	>> Output/Scaffold/Scaffold_VCs_$(BINSIZE)_predpos.tab

	# plot
	cat Output/Scaffold/Scaffold_VCs_$(BINSIZE)_$(c)_predpos.tab | \
	sed 's/__/\t/g' | \
	python pyval.py -d - -hc 1 -m -e 'p=np.mean(d[:,:2],1);hc=np.array(hc);c=np.unique(hc);n=len(c);for i in range(n):; plt.subplot(n,1,i+1); plt.scatter(d[hc==c[i],2],p[hc==c[i]]); plt.xlim(np.min(d[:,2]),np.max(d[:,2]))' -img Output/Scaffold/Scaffold_VCs_$(BINSIZE)_$(c)_predpos.png -noterm

	python pyval.py -d Output/Scaffold/Data_VCs_$(BINSIZE)_$(c).tab Output/Scaffold/Scaffold_VCs_$(BINSIZE)_$(c)_predpos.tab -hc 3 1 -m -e 'np.fill_diagonal(d[0],np.nan);plt.subplot(1,2,1);plt.title("original");mypy.lib.heatmap(d[0],colorbar=False);plt.subplot(1,2,2);plt.title("predicted");v=[hc[0][0].index(i) for i in np.array(hc[1])[np.argsort(d[1][:,0])]];mypy.lib.heatmap(d[0][v,:][:,v],colorbar=False)' -img Output/Scaffold/Data_VCs_$(BINSIZE)_$(c)_scaffolded.png -noterm


scaffold_wrap:
	python pyval.py -d =(sed -e 's/\t/__/' -e 's/\t/__/' Data/HiC_$(BINSIZE)_large.tab) Output/Scaffold/Scaffold_VCs_$(BINSIZE)_predpos.tab -hc 1 1 -m -e 'np.fill_diagonal(d[0],np.nan);plt.subplot(1,2,1);plt.title("original");mypy.lib.heatmap(d[0],clip=0.001,colorbar=False);plt.subplot(1,2,2);plt.title("predicted");v=[hc[0].index(i) for i in hc[1]];mypy.lib.heatmap(d[0][v,:][:,v],clip=0.001,colorbar=False)' -img Output/Scaffold/Data_rearranged_VCs_$(BINSIZE).png -noterm
	
	python pyval.py -d =(sed -e 's/\t/__/' -e 's/\t/__/' Data/HiC_$(BINSIZE)_large.tab) Output/Scaffold/Scaffold_VCs_$(BINSIZE)_predpos.tab  -hc 1 1 -m -e 'x=collections.OrderedDict();for i,j in zip(d[1][:,1],hc[1]):; if i not in x:;  x[i]=[]; x[i].append(j);x=collections.OrderedDict(sorted(x.items(),key=(lambda y: len(y[1])),reverse=True));t=list(it.chain(*x.values()));p=np.array([hc[0].index(i) for i in t]);nprint([t,d[0][:,p][p,:]])' \
        > Output/Scaffold/Data_rearranged_VCs_$(BINSIZE).tab

