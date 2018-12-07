%% Etude données mixtes
%[sMap bmus] = som_makeRTOM(D,'UpdateRule','mixed1','nb_var',nb_var,'nb_mod',nb_mod);
bmus=struct; sMap1=struct; Alpha_ck=struct; Beta=struct;clust=struct;Performance=struct;
DimData=[14 14 14 14];
DimBloc(1).Dim=[5 9];
DimBloc(2).Dim=[5 9];
DimBloc(3).Dim=[5 9];
DimBloc(4).Dim=[5 9];
D=[B1Q, B1T B2Q, B2T B3Q, B3T B4Q, B4T];
[sMap1 Bmus] = som_make(D,'init','randinit');
[sMap, bmus, Alpha Beta] = som_batchtrainRTOM(sMap1, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',2*i,'eta',2*j,'trainlen',50);
                    