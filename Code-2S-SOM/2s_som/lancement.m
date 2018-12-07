%[sMap bmus] = som_makeRTOM(D,'UpdateRule','mixed1','nb_var',nb_var,'nb_mod',nb_mod);
bmus=struct; sMap1=struct; Alpha_ck=struct; Beta=struct;clust=struct;Performance=struct;Result=struct;
D=som_normalize(D,'var');
Label=[ones(1,100) 2*ones(1,100) 3*ones(1,100) 4*ones(1,100)];
DimData=[5 5 5 5];

for h=1:6
    if h==1 
        param1=0.10;
        param2=0.10;
        fin1=10;fin2=10;fin=20;
        init=0.10;
    end 
    if h==2 
        param1=1;
        param2=1;
        fin1=10;fin2=10;fin=20;
        init=1;
    end 
    if h==3 
        param1=5;
        param2=5;
        fin1=10;fin2=10;fin=20;
        init=1;
    end 
    if h==4 
        param1=100;
        param2=2;
        fin1=2;fin2=10;fin=20;
        init=1;
    end 
    if h==5 
        param1=2;
        param2=100;
        fin1=10;fin2=2;fin=20;
        init=1;
    end
    if h==6 
        param1=100;
        param2=100;
        fin1=5;fin2=5;fin=10;
        init=1;
    end
    for i =0:fin1
        for j=0:fin2
            lambda=init+i*param1;
            eta=init+j*param2;
            ii=i+1; jj=j+1;
            for k=1:fin
                [sMap1 Bmus] = som_make(D,'init','randinit'); % Initialisation de la carte
                [sMap1 Bmus] = som_batchtrainRTOM(sMap1, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',lambda,'eta',eta,'trainlen',10); % in itialisation avec 2S-SOM
                disp([lambda,eta,h]); disp([ii,jj,k])
                [Result(h).sMap(ii,jj).res(k).sMap, Result(h).bmus(ii,jj).res(k).bmus, Result(h).Alpha_ck(ii,jj).res(k).Alpha_ck Result(h).Beta(ii,jj).res(k).Beta] = som_batchtrainRTOM(sMap1, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',lambda,'eta',eta,'trainlen',50);
                [Result(h).clust(ii,jj).res(k).clust Result(h).clcell Result(h).clcellConsol]=CAHOM(Result(h).sMap(ii,jj).res(k).sMap,Result(h).bmus(ii,jj).res(k).bmus,Nb_class,'My');
                if length(unique(Result(h).bmus(ii,jj).res(k).bmus))~=1
                    Result(h).Performance(ii,jj).res(k).Perf=comparaison_partitionOM(Label,Result(h).clust(ii,jj).res(k).clust(:,2)');
                    Result(h).Accuracy(ii,jj).res(k).Perf=accuracy(Label,Result(h).clust(ii,jj).res(k).clust(:,2));
                end 
            end
        end
    end
end 
%DimData=[76 216 64 240 47 6] 
    Index=[];
    for i=1:fin1,
        for j=1:fin2,
            for k=1:fin,
                Res(i,j).res(k)=Result(h).Performance(i,j).res(k).Perf.Performance.Precision;
            end
            Recall(i,j)=mean(Res(i,j).res);
        end
    end
for i=1:5,
    hold on 
    plot(Alpha_ck(1,5).res(31).Alpha_ck(:,i),['-' marker{i}],'MarkerFaceColor',couleurs{i},'Color',couleurs{i})
    xlabel('Cellules')
    ylabel('\alpha')
    legend('Bloc 1', 'Bloc 2','Bloc 3', 'Bloc 4','Bloc 5','Bloc 6','Location','Best')
    %saveas(gcf, ['AlphaVar100_',num2str(i)],'png');   
end 
marker={'^' 's' 'd' '*' 'o' 'd' '^' 'v' '>' '<' 'p''o' '>' '*' 'x' 's' 'd' '^'};
couleurs={'r' 'r' 'b' 'b' 'm' 'y' 'k' 'g'};
for l=5:5,
    figure
    compt=1;
    for b=1:7
        for vb=1:11
            if compt<DimBloc(l).Dim+1
                hold on
                plot(Beta(1,5).res(8).Beta(l).Beta_ck(:,compt),['-' marker{vb}],'MarkerFaceColor',couleurs{vb},'Color','black')  
            end
            compt=compt+1;
        end
    end
    xlabel('Cellules')
    ylabel('\beta')
    %legend('var 1', 'var 2','var 3','var 4', 'var 5','var 6', 'var 7','var 8','var 9', 'var 10','var 11', 'var 12','var 13','var 14', 'var 15','var 16', 'var 17','var 18','var 19', 'var 20','var 21', 'var 22','var 23','var 24', 'var 25','Location','Best')
    legend('var 1', 'var 2','var 3','var 4', 'var 5','Location','Best')
    %saveas(gcf, ['BetaVar20_',num2str(i)],'png');   
end 
ylim([0 0.40])

ORCLUS=KmeanRes
ORCLUS=EWKMRES
ORCLUS=ORCLUSRES
ORCLUS=EwkmMax

%%%% comparaison des performances 
for k=1:30
    Perf=comparaison_partitionOM(Labels,ORCLUS(:,k));
    PerfORCLUS.Precision.res(k)=Perf.Performance.Precision;
    PerfORCLUS.Recall.res(k)=Perf.Performance.Recall;
    PerfORCLUS.Fmeasure.res(k)=Perf.Performance.Fmeasure;
    PerfORCLUS.Accuracy.res(k)=Perf.Performance.Accuracy;
    PerfORCLUS.jaccard.res(k)=Perf.indice.jaccard;
    PerfORCLUS.rand.res(k)=Perf.indice.rand;
    PerfORCLUS.tanimoto.res(k)=Perf.indice.tanimoto;
    PerfORCLUS.VI.res(k)=Perf.indice.VI;
    PerfORCLUS.information_mutuelle.res(k)=Perf.entropie.information_mutuelle;
end 
PerfKmeansRes=PerfORCLUS
PerfEWKM=PerfORCLUS
PerfORCLUSRES=PerfORCLUS

for i=1:1
    for k=1:25
            [sMap1 BmusSOM] = som_make(D,'init','randinit','trainlen',100);
            [SOM(i).res(k).clust clcell clcellConsol]=CAHOM(sMap1,BmusSOM,8,'My');
            Perf=comparaison_partitionOM(Labels,SOM(i).res(k).clust(:,2));
            PerfSOM(i).Precision.res(k)=Perf.Performance.Precision;
            PerfSOM(i).Recall.res(k)=Perf.Performance.Recall;
            PerfSOM(i).Fmeasure.res(k)=Perf.Performance.Fmeasure;
            PerfSOM(i).Accuracy.res(k)=Perf.Performance.Accuracy;
            PerfSOM(i).jaccard.res(k)=Perf.indice.jaccard;
            PerfSOM(i).rand.res(k)=Perf.indice.rand;
            PerfSOM(i).tanimoto.res(k)=Perf.indice.tanimoto;
            PerfSOM(i).VI.res(k)=Perf.indice.VI;
            PerfSOM(i).information_mutuelle.res(k)=Perf.entropie.information_mutuelle;
    end
end 


for i=1:30,
        for k=1:25
            PartitionSOMR(i).res(k,:)=SOM(i).res(k).clust(:,2);
        end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% La matrice des bmus
for k=1:30
    BMU(:,k)=bmus(1,8).res(k).bmus;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Algorithme de consenus %%%%%%%%%
%% \lambda=12 et \eta=6 IS
bmusR=struct; sMap1R=struct; Alpha_ckR=struct;
BetaR=struct;clustR=struct;PerformanceR=struct;
sMapR=struct;
for i=1:30
    for k=1:10
            [sMap1R BmusR] = som_make(D,'init','randinit');
            [sMap1R] = som_batchtrainRTOM(sMap1R, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',1000,'eta',2*11,'trainlen',10);
            disp([i,k])
            [sMapR(i).res(k).sMap, bmusR(i).res(k).bmus, Alpha_ckR(i).res(k).Alpha_ck BetaR(i).res(k).Beta] = som_batchtrainRTOM(sMap1R, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',1000,'eta',2*11,'trainlen',50);
            [clustR(i).res(k).clust clcellR clcellConsolR]=CAHOM(sMapR(i).res(k).sMap,bmusR(i).res(k).bmus,7,'My');
            if length(unique(bmusR(i).res(k).bmus))~=1
                Perf=comparaison_partitionOM(Label,clustR(i).res(k).clust(:,2));
                PerformanceR.Precision(i).res(k)=Perf.Performance.Precision;
                PerformanceR.Recall(i).res(k)=Perf.Performance.Recall;
                PerformanceR.Fmeasure(i).res(k)=Perf.Performance.Fmeasure;
                PerformanceR.Accuracy(i).res(k)=Perf.Performance.Accuracy;
                PerformanceR.jaccard(i).res(k)=Perf.indice.jaccard;
                PerformanceR.rand(i).res(k)=Perf.indice.rand;
                PerformanceR.tanimoto(i).res(k)=Perf.indice.tanimoto;
                PerformanceR.VI(i).res(k)=Perf.indice.VI;
                PerformanceR.information_mutuelle(i).res(k)=Perf.entropie.information_mutuelle;
            end 
    end
end
%%%%%%%%% Les matrices des partitions
Data=PerformanceR.Accuracy; ResR=[];
for i=1:30,
        ResR(i)=mean(Data(i).res);
        for k=1:10
            PartitionR(i).res(k,:)=clustR(i).res(k).clust(:,2);
        end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Les applications des algorithmes de consensus %%%%%%%%%%
for i=1:30
    [VEES,bmus2R(i).Res,VEPU] = StatisFunction(PartitionR(i).res, nb_class );
    [VEESMO,bmus2(i).Res,VEPUSMO,VAPSMO] = StatisFunctionOM( PartitionR(i).res, nb_class );
    Purete(i).Res=comparaison_partitionOM(Label,bmus2R(i).Res);
    PureteR(i)= Purete(i).Res.Performance.Accuracy;
    PureteSMO(i).Res=comparaison_partitionOM(Label,bmus2(i).Res);
    PureteRSMO(i)= PureteSMO(i).Res.Performance.Accuracy;
    disp(i)
end

for i=1:30
    [VEES,bmus2R(i).Res,VEPU] = CSPAOM(PartitionR(i).res, nb_class );
    PureteCSPA(i).Res=comparaison_partitionOM(Label,bmus2R(i).Res);
    PureteCSPAR(i)= PureteCSPA(i).Res.Performance.Accuracy;
    disp(i)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Consensus avec NMF%%%%%%%%%%
for i=1:30
    [HNMF,DNMF, bmusRNMF(i).Res] = triNmfConsensus( PartitionR(i).res,7 );
    [W bmus ] = weightedExp( PartitionR(i).res,7  );
    PureteNMF(i).Res=comparaison_partitionOM(Label,bmusRNMF(i).Res);
    PureteNMFR(i)= PureteNMF(i).Res.Performance.Accuracy;
    [W bmusWNMF(i).Res ] = weightedExp( PartitionR(i).res,7);
    PureteWNMF(i).Res=comparaison_partitionOM(Label,bmusWNMF(i).Res);
    PureteWNMFR(i)= PureteWNMF(i).Res.Performance.Accuracy;
    disp(i)
end






for i=1:30
    [VEESOM,bmus2SOM(i).Res ] = StatisFunction(PartitionSOM.res, 4 );
    PureteSOM(i).Res=comparaison_partitionOM(Label,bmus2SOM(i).Res);
    PureteSOMR(i)= PureteSOM(i).Res.Performance.Accuracy;
    [HNMF,DNMF, bmusSOMNMFRes] = triNmfConsensus( PartitionSOM.res,4 );
    [W bmusWSOM ] = weightedExp( PartitionSOM.res,4  );
    PureteNMFSOMRes=comparaison_partitionOM(Label,bmusSOMNMFRes);
    PureteNMFSOMR(i)= PureteNMFSOMRes.Performance.Accuracy;
    [W bmusWNMFSOM(i).Res ] = weightedExp( PartitionSOM(i).res,4  );
    PureteWNMFSOM(i).Res=comparaison_partitionOM(Label,bmusWNMFSOM(i).Res);
    PureteWNMFSOMR(i)= PureteWNMFSOM(i).Res.Performance.Accuracy;
    disp(i)
end

