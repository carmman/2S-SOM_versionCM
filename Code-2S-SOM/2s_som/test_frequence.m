function [pvalue,phyper]=test_frequence(nkj,nk,nj,n)
%[pvalue]=test_frequence(nkj,nk,nj,n)
%calcule la pvalue avec la loi normale
%nkj : nombre d'individus de la modalité j dans la classe k
%nk : nombre d'individus dans la classe k
%nj : nombre d'individus de la modalité j dans l'échantillon
%n : nombre d'individu dans l'échantillon.

nkj;
Ekj=nk*nj/n; %nombre d'individus ayant la modalité j attendus dans la classe k (si équirépartition)
Skj=sqrt(nk*(n-nk)*(1/(n-1))*(nj/n)*(1-(nj/n)));

tkj= (nkj-Ekj)/Skj; %valeur test
 
 pvalue=1-normcdf(abs(tkj),0,1);
 phyper=hygecdf(nkj,n,nj,nk);
 if phyper>0.5
     phyper=1-phyper;
 end