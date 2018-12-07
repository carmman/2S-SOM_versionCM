

%% Utilisation de 2S-SOM
data=IS(:,[1:2,4:19]);
D=som_normalize(data,'var');

[sMap1 Bmus]=som_make(D,'init','randinit');  %% Initialisation des paramètres de SOM
[cl cellOR ]=CAHOM(sMap1,Bmus,Nbclust,'My','ward');
Perf=comparaison_partitionOM(Labels,cl(:,1));

%Result=struct;
for i=1:M
    for j=1:M
        for k=1:k
            [sMap1 Bmus]=som_make(D,'init','randinit');  %% Initialisation des paramètres de SOM
            [Result(i,j).sMap(k).sMap Result(i,j).bmus(k).bmus Result(i,j).Alpha(k).Alpha Result(i,j).Beta(k).Beta]=SS_SOM_Fast(sMap1, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',(i-1)*10+1,'eta',(i-1)*10+1,'trainlen',50);
        end
    end
end 
%% Recupération de classes suite à une CAH sur la sortie 2S-SOM

CAHIndex='ward';

for i=1:M
    for j=1:M
        for k=1:5
            [Result(i,j).cl(k).cl Result(i,j).cellOR(k).cellOR]=CAHOM(Result(i,j).sMap(k).sMap,Result(i,j).bmus(k).bmus,Nbclust,'My',CAHIndex);
            Result(i,j).Perf(k).Perf=comparaison_partitionOM(Labels,Result(i,j).cl(k).cl(:,2));
        end
    end
end 

% Recupération des index particuliers 

for i=1:M
    for j=1:M
        for k=1:20
            PerfNMI1(k)=Result(i,j).Perf(k).Perf.indice.nmi;
            PerfAdjustedRand1(k)=Result(i,j).Perf(k).Perf.indice.AdusjtedRand;
            PerfAccu1(k)=Result(i,j).Perf(k).Perf.Performance.Accuracy;    
        end
            PerfNMI(i,j)=mean(PerfNMI1);
            PerfAdjustedRand(i,j)=mean(PerfAdjustedRand1);
            PerfAccu(i,j)=mean(PerfAccu1); 
            % 
            PerfNMI_Var(i,j)=std(PerfNMI1);
            PerfAdjustedRand_Var(i,j)=std(PerfAdjustedRand1);
            PerfAccu_Var(i,j)=std(PerfAccu1); 
    end
end 

% Apprentissage Kmeans

for k=1:20
    ClKmeans(:,k)=kmeans(D,4);
    PerfKmeans(k).Perf=comparaison_partitionOM(Labels,ClKmeans(:,k));
    PerfKmeansNMI(k)=PerfKmeans(k).Perf.indice.nmi;
    PerfKmeansAdjustedRand(k)=PerfKmeans(k).Perf.indice.AdusjtedRand;
    PerfKmeansAccu(k)=PerfKmeans(k).Perf.Performance.Accuracy;    
end

%  Apprentissage SOM

for k=1:20
    [SOM(k).sMap1 SOM(k).Bmus]=som_make(D,'init','randinit');  %% Initialisation des paramètres de SOM
    [SOM(k).cl SOM(k).cellOR ]=CAHOM(SOM(k).sMap1,SOM(k).Bmus,Nbclust,'My',CAHIndex);
    SOM(k).Perf=comparaison_partitionOM(Labels,SOM(k).cl(:,1));
    PerfSOMNMI(k)=SOM(k).Perf.indice.nmi;
    PerfSOMAdjustedRand(k)=SOM(k).Perf.indice.AdusjtedRand;
    PerfSOMAccu(k)=SOM(k).Perf.Performance.Accuracy;    
end


% Apprentissage EWKmeans

for k=1:20
    PerfEWKmeans(k).Perf=comparaison_partitionOM(Labels,ClEWKMeans(:,k));
    PerfEWKmeansNMI(k)=PerfEWKmeans(k).Perf.indice.nmi;
    PerfEWKmeansAdjustedRand(k)=PerfEWKmeans(k).Perf.indice.AdusjtedRand;
    PerfEWKmeansAccu(k)=PerfEWKmeans(k).Perf.Performance.Accuracy;    
end


PerfEWNMI=mean(PerfEWKmeansNMI);
PerfEWAdjustedRand=mean(PerfEWKmeansAdjustedRand);
PerfEWAccu=mean(PerfEWKmeansAccu); 
            % 
PerfEWNMI_Var=std(PerfEWKmeansNMI);
PerfEWAdjustedRand_Var=std(PerfEWKmeansAdjustedRand);
PerfEWAccu_Var=std(PerfEWKmeansAccu); 

            % Apprentissage FGKmeans

for k=1:20
    PerfFGKmeans(k).Perf=comparaison_partitionOM(Labels,ClFGkm(:,k));
    PerfFGKmeansNMI(k)=PerfFGKmeans(k).Perf.indice.nmi;
    PerfFGKmeansAdjustedRand(k)=PerfFGKmeans(k).Perf.indice.AdusjtedRand;
    PerfFGKmeansAccu(k)=PerfFGKmeans(k).Perf.Performance.Accuracy;    
end
PerfFGNMI=mean(PerfFGKmeansNMI);
PerfFGAdjustedRand=mean(PerfFGKmeansAdjustedRand);
PerfFGAccu=mean(PerfFGKmeansAccu); 
            % 
PerfFGNMI_Var=std(PerfFGKmeansNMI);
PerfFGAdjustedRand_Var=std(PerfFGKmeansAdjustedRand);
PerfFGAccu_Var=std(PerfFGKmeansAccu); 

            
% Create the correspondannce between the map

% Creation of Z_l data sets

% Structuraton de la base les noise en dernières position 

NoiseB(1).Dim=[6 8 10 12 13 17 19 22 23 24];
NoiseB(2).Dim=[1 4 5 6 10 11 14 15 17 24]+25;
NoiseB(3).Dim=[9 10 12 13 15 18 19 20 22 23]+50;
NoiseB(4).Dim=[1 4 8 11 12 16 19 22 23 24]+75;

    Variable=1:25;
    Relevance=Variables(~ismember(Variables(1:25),NoiseB(1).Dim));
    VarTot=[Relevance NoiseB(1).Dim];

for i=1:3
    Variable=(1:25)+25*i;
    Relevance=Variable(~ismember(Variable,NoiseB(i+1).Dim));
    VarTot1=[Relevance NoiseB(i+1).Dim];
    VarTot=[VarTot VarTot1];
end 

% VarTotF=[VarTot 101:110];
VarTotF=1:18;
Z_l=struct;


for l=1:M
    if l==1
        Index(:,l)=sort(randsample(2310, 2310, false));
        Z_l(l).D=D( Index(:,l),VarTotF);
        [sMap1 Bmus]=som_make(Z_l(l).D,'init','lininit');  %% Initialisation des paramètres de SOM
        [Z_l(l).sMap Z_l(l).bmus Z_l(l).Alpha Z_l(l).Beta ]=SS_SOM(sMap1, Z_l(l).D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',4*10+1,'eta',9*10+1,'trainlen',50);
    else
    Index(:,l)=sort(randsample(2310, 2310, true));
    Z_l(l).D=D( Index(:,l),VarTotF);
    [sMap1 Bmus]=som_make(Z_l(l).D,'init','lininit');  %% Initialisation des paramètres de SOM
    [Z_l(l).sMap Z_l(l).bmus Z_l(l).Alpha Z_l(l).Beta ]=SS_SOM(sMap1, Z_l(l).D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',2,'eta',6,'trainlen',50);
    end
end 

for l=1:M
    SSOM_BMUS(:,l)=Z_l(l).bmus;
end 

EnsCluster=Index;
            
Member=struct;
for k=1:M
    for kk=1:M
        Inter=intersect(EnsCluster(:,k),EnsCluster(:,kk)); % points belonging two bootstraps samples
        kMember=ismember(EnsCluster(:,k),Inter);           % Index points in sample k
        kkMember=ismember(EnsCluster(:,kk),Inter);         % Index points in sample kk
        Member(k, kk).Member=kMember & kkMember; 
    end 
end        


for k=1:M
    for kk=1:M
        MMember(k,kk)=abs(sum(Member(k,kk).Member([1 2]))-1);
    end 
end
            
% Cell contient la correspondance des cellules

[S  Cell C E B]=SimIJ_OM(SSOM_BMUS, Member, SSOM_BMUS, SSOM_BMUS,3);


% recuperationd des poids dans les cellules 

for l=1:M
    SSOM_BMUS(:,l)=Z_l(l).bmus;
end 

[ BS  AS] = BetaSample(Z_l ,Cell,2);

% BS contient les poids des cellules correspondante 
ResBeta=struct;
for k=1:5
    for i=1:Ncell
        ResWeight=FeatureSelectionOM( BS(i).bloc(k).W, 1/DimData(k), 'right', 0.05 );
        ResBeta(k).res(i,:)=ResWeight(:,4)';
    end
    [A B]=sort(median(ResBeta(k).res));
    ResBetaBoxplot=ResBeta(k).res(:,B);
    figure
    boxplot(ResBetaBoxplot,B,'plotstyle','compact')
    xlabel('Variables')
    ylabel('\beta')
    saveas(gcf, ['MeansBetaBlock', num2str(k)], 'png') 
end

for i=1:Ncell
        ResWeight=FeatureSelectionOM( AS(i).W, 1/size(DimData,2), 'right', 0.05 );
        ResWeightAlpha(i,:)=ResWeight(:,4)';
end

% AS  contient les poids des cellules correspondante 
      
boxplot(ResWeightAlpha,'plotstyle','compact','colors','r')   
xlabel('Blocks')
ylabel('\alpha')
saveas(gcf,'MeansAlphaBlock', 'png')

figure
PerfNMIEta=PerfNMI';
%% gap statistic 
Symbol={'h','o','x','+','s','d','v','^','p','<','>'};
plot(PerfNMIEta(1:10,1),['-k' Symbol{1}],'LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',3)
LEGENDE{1}=['\lambda','=',num2str(1)];
for  i=1:10
    hold on ;
    plot(PerfNMIEta(1:10,i),['-k' Symbol{i+1}],'LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',3);
    LEGENDE{i+1}=['\lambda','=',num2str(i)];
end 
ylabel('NMI Index');
xlabel('\eta');
legend(LEGENDE);


% apprentissage de 2S-SOM sur la base reduite
Dreduit=D(:,[2 4 5 6 7 8 11 12 13 18 19 20]);
for i=2:10
    for j=1:10
        for k=1:10
            [sMap1 Bmus]=som_make(Dreduit,'init','randinit');  %% Initialisation des paramètres de SOM
            [ResultReduit(i,j).Res(k).sMap ResultReduit(i,j).Res(k).bmus ResultReduit(i,j).Res(k).Alpha ResultReduit(i,j).Res(k).Beta]=SS_SOM(sMap1, Dreduit,'TypeAlgo','2SSOM','DimData',[3 3 3 3],'DimBloc',DimBloc,'lambda',2*i,'eta',2*j,'trainlen',50);
            [ResultReduit(i,j).Res(k).cl ResultReduit(i,j).Res(k).cellOR]=CAHOM(ResultReduit(i,j).Res(k).sMap,ResultReduit(i,j).Res(k).bmus,Nbclust,'My',CAHIndex);
            ResultReduit(i,j).Res(k).Perf=comparaison_partitionOM(Labels,ResultReduit(i,j).Res(k).cl(:,1));
            PerfNMIR1(k)=ResultReduit(i,j).Res(k).Perf.indice.nmi
            PerfAdjustedRandR1(k)=ResultReduit(i,j).Res(k).Perf.indice.AdusjtedRand;
            PerfAccuR1(k)=ResultReduit(i,j).Res(k).Perf.Performance.Accuracy;    
        end
        PerfNMIR(i,j)=mean(PerfNMIR1);
        PerfAdjustedRandR(i,j)=mean(PerfAdjustedRandR1);
        PerfAccuR(i,j)=mean(PerfAccuR1); 
             
        PerfNMI_VarR(i,j)=std(PerfNMIR1);
        PerfAdjustedRand_VarR(i,j)=std(PerfAdjustedRandR1);
        PerfAccu_VarR(i,j)=std(PerfAccuR1); 
    end 
end 
            
 

%  Sparse K-means
for k=1:20

    PerfSparseKmeans(k).Perf=comparaison_partitionOM(Labels,ClSparcl(:,k));
    PerfSparseKmeansNMI(k)=PerfSparseKmeans(k).Perf.indice.nmi;
    PerfSparseKmeansAdjustedRand(k)=PerfSparseKmeans(k).Perf.indice.AdusjtedRand;
    PerfSparseKmeansAccu(k)=PerfSparseKmeans(k).Perf.Performance.Accuracy;    
end
PerfSparseNMI=mean(PerfSparseKmeansNMI);
PerfSparseAdjustedRand=mean(PerfSparseKmeansAdjustedRand);
PerfSparseAccu=mean(PerfSparseKmeansAccu); 
            % 
PerfSparseNMI_Var=std(PerfSparseKmeansNMI);
PerfSparseAdjustedRand_Var=std(PerfSparseKmeansAdjustedRand);
PerfSparseAccu_Var=std(PerfSparseKmeansAccu);
        
