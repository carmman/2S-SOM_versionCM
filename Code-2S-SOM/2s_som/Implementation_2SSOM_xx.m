

%% Utilisation de 2S-SOM

% data : base de données à traiter  
% D : data normalisés
% DimData : Nombre de variables par blocs
% DimBloc : Structure contenant le nombre de variables par blocs 

D=som_normalize(data,'var');
DimData=[5 5 5 5 5];
DimBloc(1).Dim=5;
DimBloc(2).Dim=5;
DimBloc(3).Dim=5;
DimBloc(4).Dim=5;
DimBloc(5).Dim=5;
DimBloc(6).Dim=6;


% Initalisation des paramètres 

[sMap1 Bmus]=som_make(D,'init','randinit');  %% Initialisation des paramètres de SOM
%[cl,cellOR]=CAHOM(sMap1,Bmus,Nbclust,'My','ward');
 [cl cellOR clcellConsol DB_index]=CAHOM(sMap1,Bmus,Nbclust,'My','ward');
Perf=comparaison_partitionOM(Labels,cl(:,1));

% Result=struct; eta=1 et lambda=2 sont fixé libre à utilisateurs de les faire varier. 

for i=1:M
    for j=1:M
        for k=1:K
            [sMap1 Bmus]=som_make(D,'init','randinit');  %% Initialisation des paramètres de SOM
            [Result(i,j).sMap(k).sMap Result(i,j).bmus(k).bmus Result(i,j).Alpha(k).Alpha Result(i,j).Beta(k).Beta]=Som_batchtrainRTOM(sMap1, D,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda',2,'eta',1,'trainlen',50);
        end
    end
end 
%% Recupération de classes suite à une CAH sur la sortie 2S-SOM

CAHIndex='ward';

for i=1:M
    for j=1:M
        for k=1:K
            [Result(i,j).cl(k).cl Result(i,j).cellOR(k).cellOR]=CAHOM(Result(i,j).sMap(k).sMap,Result(i,j).bmus(k).bmus,Nbclust,'My',CAHIndex);
            Result(i,j).Perf(k).Perf=comparaison_partitionOM(Labels,Result(i,j).cl(k).cl(:,1));
        end
    end
end 

% Recupération des index particuliers 

for i=1:M
    for j=1:M
        for k=1:K
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





