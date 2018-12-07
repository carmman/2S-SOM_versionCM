%% Etude données mixtes
bmus=struct; sMap=struct; Alpha_ck=struct; Beta=struct;clust=struct;Performance=struct;
%% Préparation des données 
[B1 C1 D1]=transformDataOM(Logement(:,1:70),[32 38]);
[B2 C2 D2]=transformDataOM(Men(:,1:11),[5  6]);
[B3 C3 D3]=transformDataOM(Hab(:,1:45),[22 23]);
[B4 C4 D4]=transformDataOM(Pol1,[15 2]);
DD=[Log(:,1:32) LogQSOM Men(:,1:5) MenQSOM Hab(:,1:31) HabQSOM];% B3Q, B3T B4Q, B4T];
%Dans D vous disposez de 4 blocs de variables: B1Q représente les variables Quanti du bloc 1 et B1T la matrice disjonctive complète des variables catégorielles du bloc 1. De même pour D2  
DimData=[size(B1,2) size(B2,2) size(B3,2) size(B4,2) ]; %    dimB1= Dimension de B1Q + Dimension de B1T par exemple (5+9) 5 étant le nombre de variables Quanti et 9 le nombre total de modalités de deux variables quali 
DimBloc(1).Dim=size(B1,2); % NbVarB1 est le nombre de  variables quanti du bloc 1 ... tel que NbVarB1 + NbModB1=dimB1 soit [5  9] qui fait bien 14 
DimBloc(2).Dim=size(B2,2);
DimBloc(3).Dim=size(B3,2);
DimBloc(4).Dim=size(B4,2);
%% Le nombre de modalités par variable catégorielle et par bloc
DimCategorieVar(1).Dim=[NbrMod(C1)]; % Ce bloc contient deux variables catégorielles, la variable 1 contient 5 modalités et la variable 2 en contient 4. soit [5 4]  
DimCategorieVar(2).Dim=[NbrMod(C2)];
DimCategorieVar(3).Dim=[NbrMod(C3)];
DimCategorieVar(4).Dim=[NbrMod(C4)];
%% Implementation
[sMap1 Bmus] = som_make(D,'init','randinit'); % initialisation de carte 
i=1; j=1;
%[sMap bmus Alpha Beta]=som_makeRTOM(D,'TypeAlgo',TypeAlgo,'UpdateRule',UpdateRule,'DimCategorieVar',DimCategorieVar,'DimData',DimData, 'DimBloc',DimBloc,'lambda',1,'eta',1,'trainlen',50);
[sMap bmus Alpha Beta]=som_makeRTOM(D,'TypeAlgo',TypeAlgo,'UpdateRule',UpdateRule,'DimCategorieVar',DimCategorieVar,'DimData',DimData, 'DimBloc',DimBloc,'lambda',1,'eta',1,'trainlen',50);
[sMap bmus Alpha Beta]=som_batchtrainRTOM(sMap1, D,'TypeAlgo','2SSOM','numeric',[1 1],1,DimCategorieVar,'DimData',DimData,'DimBloc',DimBloc,'lambda',5,'eta',100,'trainlen',50);
% En sortie vous disposez d'une carte 'sMap', des neurones gagnant de chaque observation 'bmus', des poids 'Alpha' sur les blocs et des poids 'Beta' sur les variables.   
% vous fixer i=1 et j=1
%% Apprentissage multiple
for lambda=0:10
    for eta=0:10
        for k=1:10
            disp([lambda eta k])
            x=0.10*lambda+0.10; y=2*0.10+0.10;
            [sMap1(lambda+1,eta+1).res(k).sMap bmus1(lambda+1,eta+1).res(k).sMap Alpha1(lambda+1,eta+1).res(k).sMap Beta1(lambda+1,eta+1).res(k).sMap]=som_makeRTOM(D,'TypeAlgo',TypeAlgo,'UpdateRule',UpdateRule,'DimCategorieVar',DimCategorieVar,'DimData',DimData, 'DimBloc',DimBloc,'lambda',x,'eta',y,'trainlen',50);
        end 
    end 
end 
%% Calcule de la meilleure carte


for i=1:11
    for j=1:11
        Dist=0.0
        for k=1:10
            Dist=Dist+som_distortion(sMap(i,j).res(k).sMap,D)/257;
            [Precl(i,j).res(k).cl PreclcellOR(i,j).res(:,k) PreConsolOR(i,j).res(:,k) PreDB(i,j).res(:,k)]=CAHOM(sMap(lambda+1,eta+1).res(k).sMap,bmus(lambda+1,eta+1).res(k).sMap,Nbclust,'My',CAHIndex{6});
        end 
        Distorsion1(i,j)=Dist/10;
    end 
end 
figure
plot(Distorsion(1+2*[0:4],:)','-*')
figure
plot(Distorsion(:,1+2*[0:4]),'-*')
figure

for i=1:Appr, 
    disp(i)
    subplot(2,5,i)
    [c,p,err,ind(:,i)]=kmeans_clusters(D,10);
    plot(1:length(ind(:,i)),ind(:,i),'x-');
end

figure
Ind=mean(ind([2:10]',1:10));
plot(1:length(Ind),Ind,'x-');


%% consensus de SOM 

%% Consensus avec STatis sur ensemble de diversification
VEE=struct;bmus2=struct;W=struct;VEPU=struct;VAP=struct;
VEETilde=struct;bmus2VEETilde=struct;WVEETilde=struct;VEPUVEETilde=struct;VAPVEETilde=struct;
   

    for k=1:Appr
        [DiverEns(k).clustOR DiverEns(k).clcellOR DiverEns(k).clcellConsolOR]=CAHOM(sMap(1,4).res(k).sMap,bmus(1,4).res(k).sMap,Nbclust,TypeCAH,CAHIndex{6});
        DiverEns(k).Cluster=DiverEns(k).clustOR(:,2);
        DiverEns(ii).distorsion(k).res = som_distortion(sMap(1,4).res(k).sMap,D)/size(D,2);
        %PureteRSOM(k)= comparaison_partitionOM(Label,DiverEns(k).Cluster);
    end 
    % Calcul des mu_b
    for k=1:Appr
        DiverEns(ii).Distor(k) = 1/DiverEns(ii).distorsion(k).res;
    end
    FSOM(ii).Z_tilde=[];
    FSOM(ii).Z_Directe=[];

    for k=1:Appr
        DiverEns(ii).mu(k)=DiverEns(ii).Distor(k)/sum(DiverEns(ii).Distor);
        FSOM(ii).Z_tilde=[FSOM(ii).Z_tilde DiverEns(ii).mu(k)*sMap(1,4).res(k).sMap.codebook(bmus(1,4).res(k).sMap,:)];
        FSOM(ii).Z_Directe=[FSOM(ii).Z_Directe sMap(1,4).res(k).sMap.codebook(bmus(1,4).res(k).sMap,:)];
    end
    
    %% Apprentissage Fusion FSOM
   [FSOMRes(ii).sMapOR FSOMRes(ii).BmusOM]=som_make(FSOM(ii).Z_tilde, 'init', 'randinit','training','long' );
   [FSOMRes(ii).clustOR FSOMRes(ii).clcellOR FSOMRes(ii).clcellConsolOR]=CAHOM( FSOMRes(ii).sMapOR,FSOMRes(ii).BmusOM,Nbclust,TypeCAH,CAHIndex{6});
   [FSOMRes(ii).clustORD FSOMRes(ii).clcellORD FSOMRes(ii).clcellConsolORD]=CAHOM( FSOMRes(ii).sMapORD,FSOMRes(ii).BmusOMD,Nbclust,TypeCAH,CAHIndex{6});
    
    FSOMRes(ii).Cluster=FSOMRes(ii).clustOR(:,2);
    FSOMRes(ii).ClusterD=FSOMRes(ii).clustORD(:,2);
    PureteFSOM(ii)= comparaison_partitionOM(Label,FSOMRes(ii).Cluster);
    PureteFSOMD(ii)= comparaison_partitionOM(Label,FSOMRes(ii).ClusterD);
    %% Apprentissage Fusion STATIS
    for k=1:Appr
        EnsMatrix(k).matrix=DiverEns(k).sMapOR.codebook(DiverEns(k).BmusOM,:);
        EnsPartition(k,:)=DiverEns(k).Cluster;
    end
%     Initialisation 
    [VEE.App(ii).res,bmus2.App(ii).res,W.App(ii).res,VEPU.App(ii).res, VAP.App(ii).res] = StatisFunction(EnsPartition, Nb_class,'ward',1);
    PureteRGle(ii)= comparaison_partitionOM(Label,bmus2.App(ii).res);
    %Tilde
    [VEEVEETilde.App(ii).res,bmus2VEETilde.App(ii).res,WVEETilde.App(ii).res,VEPUVEETilde.App(ii).res, VAPVEETilde.App(ii).res] = StatisFunction(EnsMatrix, Nb_class,'ward',2);
    PureteRGleVEETilde(ii)= comparaison_partitionOM(Label,bmus2VEETilde.App(ii).res);
     EnsZ_tilde=[];
    for k=1:Appr
        EnsZ_tilde=[EnsZ_tilde sqrt(VEEVEETilde.App(ii).res(k))*DiverEns(k).sMapOR.codebook(DiverEns(k).BmusOM,:)];
    end
    ZZ(ii).EnsZ_tilde=EnsZ_tilde;  
    [sMapOR BmusOM] = som_make(ZZ(k).EnsZ_tilde, 'init', 'randinit','training','long' );
    
    for i=6:6,
        [clustOR1(i).res clcellOR(i).res clcellConsolOR(i).res]=CAHOM(sMapOR,BmusOM,Nbclust,TypeCAH,CAHIndex{i});
        PureteClasse(i).res(ii)= comparaison_partitionOM(Label,clustOR1(i).res(:,2));
    end

    %% Apprentissage Fusion NMF
     all_conMatrix = allConMatrix( EnsPartition );
     W=zeros(nb_partition,1);
       for i=1: nb_partition
           W(i)=1/nb_partition;
       end
    M_avg= agreConnMat( all_conMatrix,W);
    bmusNMF=NMFCluster( M_avg,Nbclust);
    PureteBlocNMF(ii)= comparaison_partitionOM(Label,bmusNMF);
    bmusWNMF=NMFClusterWNMF( M_avg,Nbclust);
    PureteBlocWNMF(ii)= comparaison_partitionOM(Label,bmusWNMF);
    %% Apprentissage Fusion CSPA
    Y = pdist(M_avg,'euclid'); 
    Z = linkage(Y,CAHIndex{6}); 
    T = cluster(Z,'maxclust',nb_class);
    PureteBlocCSPA(ii)=comparaison_partitionOM(Label,T);
    clear  EnsPartition;

PerfM=[StatIndex(PureteBlocNMF(1:10)) StatIndex(PureteBlocWNMF(1:10)) StatIndex(PureteBlocCSPA(1:10)); StatIndex(PureteFSOM(1:10)) StatIndex(PureteFSOMD(1:10))]
  
%Description des classes les variables pertinentes

for k=1:10, 
figure; 
BsMapk=sMap(1,4).res(k).sMap;
BsMapk.codebook(:,1:4)=Alpha(1,4).res(k).sMap;
som_show(BsMapk,'comp',1:4);
end  

%%% variable significativement
PoidsMoy=zeros(567,DimData(4));
for k=1:10,
    Idx=bmus(1,4).res(k).sMap;
    PoidsMoy=PoidsMoy+Beta(1,4).res(k).sMap(4).Beta_ck(Idx,:);
end

figure

X=1:5;
