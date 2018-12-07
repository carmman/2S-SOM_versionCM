
bmus=struct; sMap1=struct; Alpha_ck=struct; Beta=struct;clust=struct;Performance=struct;

for i =1:10
    for j=1:10
        for k=1:10
            [sMap1 Bmus] = som_make(D,'init','randinit');
            [sMap1 Bmus] = som_batchtrainRTOM(sMap1, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',1000,'eta',2*j,'trainlen',10);
            disp([i,j,k])
            [sMap(i,j).res(k).sMap, bmus(i,j).res(k).bmus, Alpha_ck(i,j).res(k).Alpha_ck Beta(i,j).res(k).Beta] = som_batchtrainRTOM(sMap1, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',1000000,'eta',2*j,'trainlen',50);
            [clust(i,j).res(k).clust clcell clcellConsol]=CAHOM(sMap(i,j).res(k).sMap,bmus(i,j).res(k).bmus,Nb_class,'My');
            if length(unique(clust(i,j).res(k).clust(:,2)))~=1
            Perf=comparaison_partitionOM(Label,clust(i,j).res(k).clust(:,2));
            Performance.Precision(i,j).res(k)=Perf.Performance.Precision;
            Performance.Recall(i,j).res(k)=Perf.Performance.Recall;
            Performance.Fmeasure(i,j).res(k)=Perf.Performance.Fmeasure;
            Performance.Accuracy(i,j).res(k)=Accuracy(Label,clust(i,j).res(k).clust(:,2));
            Performance.jaccard(i,j).res(k)=Perf.indice.jaccard;
            Performance.rand(i,j).res(k)=Perf.indice.rand;
            Performance.tanimoto(i,j).res(k)=Perf.indice.tanimoto;
            %Performance.VI(i,j).res(k)=Perf.indice.VI;
            %Performance.information_mutuelle(i,j).res(k)=Perf.entropie.information_mutuelle;
            end
        end 
    end
end 

%DimData=[76 216 64 240 47 6] 
Data=Performance.Accuracy;Res=[];
for i=1:15,
    for j=1:20,
        Res(i,j)=mean(Data(i,j).res);
    end
end

for i=1:2,
    hold on 
    plot(Alpha_ck(6,12).res(4).Alpha_ck(:,i),['-' marker{i}],'MarkerFaceColor',couleurs{i})
    xlabel('Cellules')
    ylabel('\alpha')
    legend('Bloc 1', 'Bloc 2','Location','Best')
    %saveas(gcf, ['AlphaVar100_',num2str(i)],'png');   
end 

marker={'o' '>' 'o' 'o' 'o' 'o' 'o' 'o' '*' 'x' 's' 'd' '^' 'v' '>' '<' 'p'};
couleurs={'c' 'y' 'r' 'b' 'm' 'y' 'k' 'g'};
for l=1:2,
    [A B]=max(Performance.Accuracy(i,j).res);
    figure
    compt=1;
    for b=1:7
        for vb=1:11
            if compt<DimBloc(l).Dim+1
                hold on
                plot(Beta(i,j).res(B).Beta(l).Beta_ck(:,compt),['-' marker{vb}],'MarkerFaceColor',couleurs{vb},'Color','black')  
            end
            compt=compt+1;
        end
    end
    xlabel('Cellules')
    ylabel('\beta')
    %legend('var 1', 'var 2','var 3','var 4', 'var 5','var 6', 'var 7','var 8','var 9', 'var 10','var 11', 'var 12','var 13','var 14', 'var 15','var 16', 'var 17','var 18','var 19', 'var 20','var 21', 'var 22','var 23','var 24', 'var 25','Location','Best')
    %legend('var 1', 'var 2','var 3','var 4', 'var 5','Location','Best')
    %saveas(gcf, ['BetaVar20_',num2str(i)],'png');   
end 

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
            [SOM(i).res(k).clust clcell clcellConsol]=CAHOM(sMap1,BmusSOM,4,'My');
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
for i=19:30
    for k=1:10
        
            [sMap1R BmusR] = som_make(D,'init','randinit');
            [sMap1R] = som_batchtrainRTOM(sMap1R, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',1000,'eta',10,'trainlen',10);
            disp([i,k])
           [sMapR(i).res(k).sMap, bmusR(i).res(k).bmus, Alpha_ckR(i).res(k).Alpha_ck BetaR(i).res(k).Beta] = som_batchtrainRTOM(sMap1R, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',1000,'eta',10,'trainlen',50);
           [clustR(i).res(k).clust clcellR clcellConsolR]=CAHOM(sMapR(i).res(k).sMap,bmusR(i).res(k).bmus,Nb_class,'My');
            if length(unique(bmusR(i).res(k).bmus))~=1
                Perf=comparaison_partitionOM(Label,clustR(i).res(k).clust(:,2));
                PerformanceR.Precision(i).res(k)=Perf.Performance.Precision;
                PerformanceR.Recall(i).res(k)=Perf.Performance.Recall;
                PerformanceR.Fmeasure(i).res(k)=Perf.Performance.Fmeasure;
                PerformanceR.Accuracy(i).res(k)=Accuracy(Label,clustR(i).res(k).clust(:,2));
                PerformanceR.jaccard(i).res(k)=Perf.indice.jaccard;
                PerformanceR.rand(i).res(k)=Perf.indice.rand;
                PerformanceR.tanimoto(i).res(k)=Perf.indice.tanimoto;
                %PerformanceR.VI(i).res(k)=Perf.indice.VI;
                %PerformanceR.information_mutuelle(i).res(k)=Perf.entropie.information_mutuelle;
            end 
    end
end
%%%%%%%%% Les matrices des partitions
Data=PerformanceR.Accuracy; ResR=[];
for i=1:22,
        ResR(i)=mean(Data(i).res);
        for k=1:10
            PartitionR(i).res(k,:)=clustR(i).res(k).clust(:,2);
        end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Les applications des algorithmes de consensus %%%%%%%%%%
for i=1:30
    [VEES,bmus2R(i).Res,VEPU] = StatisFunction(PartitionR(i).res, Nb_class );
    %[VEESMO,bmus2(i).Res,VEPUSMO,VAPSMO] = StatisFunctionOM( PartitionR(i).res, Nb_class );
    Purete(i).Res=comparaison_partitionOM(Label,bmus2R(i).Res);
    PureteR(i)= Accuracy(Label,bmus2R(i).Res);
    %PureteSMO(i).Res=comparaison_partitionOM(Label,bmus2(i).Res);
   % PureteRSMO(i)= Accuracy(Label,bmus2(i).Res);
    disp(i)
end

for i=1:30
    [VEES,bmus2R(i).Res,VEPU] = CSPAOM(PartitionR(i).res, Nb_class );
    PureteCSPA(i).Res=comparaison_partitionOM(Label,bmus2R(i).Res);
    PureteCSPAR(i)= Accuracy(Label,bmus2R(i).Res);
    disp(i)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Consensus avec NMF%%%%%%%%%%
for i=1:30
    [HNMF,DNMF, bmusRNMF(i).Res] = triNmfConsensus( PartitionR(i).res,Nb_class );
    [W bmus ] = weightedExp( PartitionR(i).res,Nb_class  );
    PureteNMF(i).Res=comparaison_partitionOM(Label,bmusRNMF(i).Res);
    PureteNMFR(i)= Accuracy(Label,bmusRNMF(i).Res);
    [W bmusWNMF(i).Res ] = weightedExp( PartitionR(i).res,Nb_class);
    PureteWNMF(i).Res=comparaison_partitionOM(Label,bmusWNMF(i).Res);
    PureteWNMFR(i)= Accuracy(Label,bmusWNMF(i).Res);
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

