% Fusion de SOM
%Base =D
%% DMU
%DimData=[76 216 64 240 47 6]
%Data(1).D=D(:,1:76);
%Data(2).D=D(:,77:292);
%Data(3).D=D(:,293:356);
%Data(4).D=D(:,357:596);
%Data(5).D=D(:,597:643);
%Data(6).D=D(:,644:649);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IS
% Data(1).D=D(:,1:9);
% Data(2).D=D(:,10:19);
%% CT3
%   Data(1).D=D(:,1:7);
%   Data(2).D=D(:,8:11);
%   Data(3).D=D(:,12:21);
%% FGKM data
%  Data(1).D=D(:,1:10);
%  Data(2).D=D(:,11:20);
%  Data(3).D=D(:,21:50);
 RFusion=struct;RSOM=struct;
 %% pré-traitement des données 
figure
for ii=1:Appr, 
    subplot(2,5,ii)
    [PresMap PreBmus] = som_make(D,'init', 'randinit','training','long' );
     [Precl(i).res PreclcellOR(:,i) PreConsolOR(:,i) PreDB(:,i)]=CAHOM(PresMap,PreBmus,Nbclust,TypeCAH,CAHIndex{6});
    for i=1:10,
        [c,p,err,ind(:,i)]=kmeans_clusters(PresMap,10);
    end
    Ind=mean(ind([2:10]',:));
    plot(1:length(Ind),Ind,'x-');
end

Nbclust=What;


for k=1:Appr
%% Etude générale 
    [RSOM(k).sMapOR RSOM(k).BmusOM] = som_make(D, 'init', 'randinit','training','long' );
    [RSOM(k).clustOR RSOM(k).clcellOR RSOM(k).clcellConsolOR]=CAHOM(RSOM(k).sMapOR,RSOM(k).BmusOM,Nbclust,TypeCAH,CAHIndex{6});
    RSOM(k).Cluster=RSOM(k).clustOR(:,2);
    
    for b=1:B
        %figure
        [RFusion(k).Result(b).sMap RFusion(k).Result(b).bmus] = som_make(Data(b).D,'init', 'randinit', 'training','long' );
        RFusion(k).Result(b).distorsion = (som_distortion(RFusion(k).Result(b).sMap,Data(b).D))/size(Data(b).D,2);
        RFusion(k).Result(b).Z_tilde=RFusion(k).Result(b).sMap.codebook(RFusion(k).Result(b).bmus,:);
        [RFusion(k).Result(b).clust RFusion(k).Result(b).clcell RFusion(k).Result(b).clcellConsol]=CAHOM(RFusion(k).Result(b).sMap,RFusion(k).Result(b).bmus,Nbclust,TypeCAH,CAHIndex{6});
        RSOM(k).Cluster=[RSOM(k).Cluster RFusion(k).Result(b).clust(:,2)];
   end

    % Calcul des mu_b
    for b=1:B
        RFusion(k).Distor(b) = 1/RFusion(k).Result(b).distorsion;
    end
    RFusion(k).Z_tilde=[];
    RFusion(k).Z_Directe=[];
    for b=1:B
        RFusion(k).Result(b).mu=RFusion(k).Distor(b)/sum(RFusion(k).Distor);
        RFusion(k).Z_tilde=[RFusion(k).Z_tilde RFusion(k).Result(b).mu*RFusion(k).Result(b).Z_tilde];
        RFusion(k).Z_Directe=[RFusion(k).Z_Directe RFusion(k).Result(b).Z_tilde];
    end
    %% creation de la table Z_tilde 
    [RFusion(k).Fusion RFusion(k).FusionBmus] = som_make(RFusion(k).Z_tilde, 'init', 'randinit','training','long' );
    [RFusion(k).Directe RFusion(k).DirecteBmus] = som_make(RFusion(k).Z_Directe, 'init', 'randinit','training','long');
%     %% Comparaison des partitions
    [RFusion(k).Fusionclust RFusion(k).Fusionclcell RFusion(k).FusionclcellConsol]=CAHOM(RFusion(k).Fusion,RFusion(k).FusionBmus,Nbclust,TypeCAH,CAHIndex{6});
    [RFusion(k).Directeclust RFusion(k).Directeclcell RFusion(k).DirecteclcellConsol]=CAHOM(RFusion(k).Directe,RFusion(k).DirecteBmus,Nbclust,TypeCAH,CAHIndex{6});
    %% creation de la matrice des partitions
    RSOM(k).Cluster=[RSOM(k).Cluster RFusion(k).Fusionclust(:,2)];
    RSOM(k).Cluster=[RSOM(k).Cluster RFusion(k).Directeclust(:,2)];
    %% Kmeans paritions
    [RFusion(k).KMclust]=kmeans(D,Nbclust);
    RSOM(k).Cluster=[RSOM(k).Cluster RFusion(k).KMclust];
    RSOM(k).Cluster=[RSOM(k).Cluster Label];
%for k=1:Appr
    for i=1:size(RSOM(k).Cluster,2)
        for j=1:size(RSOM(k).Cluster,2)
            Res=comparaison_partitionOM(RSOM(k).Cluster(:,i),RSOM(k).Cluster(:,j));
            RFusion(k).rand(i,j)=Res.indice.rand;
            RFusion(k).jaccard(i,j)=Res.indice.jaccard;
            RFusion(k). tanimoto(i,j)=Res.indice.tanimoto;
            RFusion(k).VI(i,j)=Res.indice.VI;
            RFusion(k).nmi(i,j)=Res.indice.nmi;
            RFusion(k).nvi(i,j)=Res.indice.nvi;
            RFusion(k).Precision(i,j)=Res.Performance.Precision;
            RFusion(k).Recall(i,j)=Res.Performance.Recall;
            RFusion(k).Fmeasure(i,j)=Res.Performance.Fmeasure;
            RFusion(k).Accuracy(i,j)=Res.Performance.Accuracy; 
        end 
    end
end

%%%%%%%% La moyenne et l'écart type des indices
B0B=size(RSOM(k).Cluster,2);
Perf=struct;
Perf.Rand=zeros(B0B,B0B); Perf.Jaccard=zeros(B0B,B0B);
Perf.Tanimoto=zeros(B0B,B0B); Perf.VI=zeros(B0B,B0B);
Perf.Precision=zeros(B0B,B0B); Perf.Recall=zeros(B0B,B0B);
Perf.Fmeasure=zeros(B0B,B0B); Perf.Accuracy=zeros(B0B,B0B);
Perf.nmi=zeros(B0B,B0B); Perf.nvi=zeros(B0B,B0B);

VarRand=zeros((B0B)*(B0B),1); VarJaccard=zeros((B0B)*(B0B),1);
VarTanimoto=zeros((B0B)*(B0B),1); VarVI=zeros((B0B)*(B0B),1);
VarPrecision=zeros((B0B)*(B0B),1); VarRecall=zeros((B0B)*(B0B),1);
VarFmeasure=zeros((B0B)*(B0B),1); VarAccuracy=zeros((B0B)*(B0B),1);
Varnmi=zeros((B0B)*(B0B),1); Varnvi=zeros((B0B)*(B0B),1);

for k=1:Appr
     Perf.Rand=Perf.Rand+RFusion(k).rand/Appr;
     Perf.Jaccard=Perf.Jaccard+RFusion(k).jaccard/Appr;
     Perf.Tanimoto=Perf.Tanimoto+RFusion(k). tanimoto/Appr;
     Perf.VI=Perf.VI+RFusion(k).VI/(B0B);
     Perf.Precision=Perf.Precision+RFusion(k).Precision/Appr;
     Perf.Recall=Perf.Recall+RFusion(k).Recall/Appr;
     Perf.Fmeasure=Perf.Fmeasure+RFusion(k).Fmeasure/Appr;
     Perf.Accuracy=Perf.Accuracy+RFusion(k).Accuracy/Appr;
     Perf.nmi=Perf.nmi+RFusion(k).nmi/Appr;
     Perf.nvi=Perf.nvi+RFusion(k).nvi/Appr;
     %% Les matrices des variances
     VarRand=[VarRand reshape(RFusion(k).rand,(B0B)*(B0B),1)];
     VarJaccard=[VarJaccard reshape(RFusion(k).jaccard,(B0B)*(B0B),1)];
     VarTanimoto=[VarTanimoto reshape(RFusion(k).tanimoto,(B0B)*(B0B),1)];
     VarVI=[VarVI reshape(RFusion(k).VI,(B0B)*(B0B),1)];
     VarPrecision=[VarPrecision reshape(RFusion(k).Precision,(B0B)*(B0B),1)];
     VarRecall=[VarRecall reshape(RFusion(k).Recall,(B0B)*(B0B),1)];
     VarFmeasure=[VarFmeasure reshape(RFusion(k).Fmeasure,(B0B)*(B0B),1)];
     VarAccuracy=[VarAccuracy reshape(RFusion(k).Accuracy,(B0B)*(B0B),1)];
     Varnmi=[Varnmi reshape(RFusion(k).nmi,(B0B)*(B0B),1)];
     Varnvi=[Varnvi reshape(RFusion(k).nvi,(B0B)*(B0B),1)];

end

%%%%%%%%%%%% écart-types
StdRand=reshape(std(VarRand'),B0B,B0B);
StdJaccard=reshape(std(VarJaccard'),B0B,B0B);
StdTanimoto=reshape(std(VarTanimoto'),B0B,B0B);
StdVI=reshape(std(VarVI'),B0B,B0B);
Stdnmi=reshape(std(Varnmi'),B0B,B0B);
Stdnvi=reshape(std(Varnvi'),B0B,B0B);
StdPrecision=reshape(std(VarPrecision'),B0B,B0B);
StdRecall=reshape(std(VarRecall'),B0B,B0B);
StdFmeasure=reshape(std(VarFmeasure'),B0B,B0B);
StdAccuracy=reshape(std(VarAccuracy'),B0B,B0B);

%% NMF et WNMF

% Creation et de l'ensemble des partitions pour les bases
for k=1:Appr
    clear EnsPartition;
    for b=1:B
        EnsPartition(b,:)=RSOM(k).Cluster(:,b+1);
    end
nb_partition=size(EnsPartition,1);
%tableau de structure pour les matrice de connectivité des partitions
all_conMatrix = allConMatrix( EnsPartition );
%W est le vecteur des poids à l'initialisation: ts les partitions ont le même
%poids
W=zeros(nb_partition,1);
  for i=1: nb_partition
      W(i)=1/nb_partition;
  end
M_avg= agreConnMat( all_conMatrix,W);
%% Apprentissage NMF et WNMF

[bmusNMF(:,k)]=NMFCluster( M_avg,Nbclust);
[bmusWNMF(:,k)]=NMFClusterWNMF( M_avg,Nbclust);
RNMF1(k)=comparaison_partitionOM(Label,bmusNMF(:,k)); 
RWNMF1(k)=comparaison_partitionOM(Label,bmusWNMF(:,k)); 

%% Apprentissage CSPA

%      'single'    --- nearest distance
%      'complete'  --- furthest distance
%      'average'   --- unweighted average distance (UPGMA) (also known as
%                      group average)
%      'weighted'  --- weighted average distance (WPGMA)
%      'centroid'  --- unweighted center of mass distance (UPGMC) (*)
%      'median'    --- weighted center of mass distance (WPGMC) (*)
%      'ward'      --- inner squared distance (min variance algorithm) (*)
  Y = pdist(M_avg,'euclid'); 
  for idx=6:6,
      Z = linkage(Y,CAHIndex{idx}); 
      T(k).CAHIndex(idx).res = cluster(Z,'maxclust',Nbclust);
      RCSPA.CAH(idx).res(k)=comparaison_partitionOM(Label, T(k).CAHIndex(idx).res);
  end 
 end 
 [ StatIndex( RCSPA.CAH(idx).res), StatIndex( RWNMF1) StatIndex(RNMF1)]
  
%% représentation graphique des cartes  
%som_plotplane(RFusion(k).Result(b).sMap, M, 'b');
%    STEP 4: VISUALIZING THE SELF-ORGANIZING MAP: SOM_SHOW_ADD
%    =========================================================
%    The SOM_SHOW_ADD function can be used to add markers, labels and
%    trajectories on top of SOM_SHOW created figures. The function
%    SOM_SHOW_CLEAR can be used to clear them away.
%    Here, the U-matrix is shown on the left, and an empty grid
%    named 'Labels' is shown on the right.
% Pour la table originiale
figure
for i=1:Appr, 
    subplot(2,5,i)
    [c,p,err,ind(:,i)]=kmeans_clusters(RFusion(i).Fusion,10);
    plot(1:length(ind(:,i)),ind(:,i),'x-');
end
figure
Ind=mean(ind([2:10]',:));
plot(1:length(Ind),Ind,'x-');



RFusion(k).Result(b).distorsion
for b=1:B
    figure
    sD=Data(b).D;
    sM1=RFusion(k).Result(b).sMap;
    sM2=RFusion(k).Result(b).sMap;
    comp=size(sM2.codebook,2);
    sM1.codebook(:,end)=RFusion(k).Result(b).clcellConsol;
    sD = som_data_struct(sD);
    sD=som_label(sD,'add',[1:length(Label)]',num2str(Label));
    sM2 = som_autolabel(sM2,sD,'vote');
    sM1 = som_autolabel(sM1,sD,'vote');
    som_show(sM1,'comp',comp);
    %som_show_add('label',sM2.labels,'textsize',5,'textcolor','k');
end

%fusion
figure
sD=RFusion(pos).Z_tilde;
sM1=RFusion(pos).Fusion;
sM2=RFusion(pos).Fusion;
comp=size(sM2.codebook,2);
sM1.codebook(:,end)=RFusion(pos).FusionclcellConsol;
sD = som_data_struct(sD);
sD=som_label(sD,'add',[1:length(Label)]',num2str(Label));
sM2 = som_autolabel(sM2,sD,'vote');
sM1 = som_autolabel(sM1,sD,'vote');
som_show(sM1,'comp',comp);
%som_show_add('label',sM2.labels,'textsize',8,'textcolor','k');

figure
%fusion
sD=RFusion(pos).Z_Directe;
sM1=RFusion(pos).Directe;
sM2=RFusion(pos).Directe;
comp=size(sM2.codebook,2);
sM1.codebook(:,end)=RFusion(pos).DirecteclcellConsol;
sD = som_data_struct(sD);
sD=som_label(sD,'add',[1:length(Label)]',num2str(Label));
sM2 = som_autolabel(sM2,sD,'vote');
sM1 = som_autolabel(sM1,sD,'vote');
som_show(sM1,'comp',comp);
%som_show_add('label',sM2.labels,'textsize',8,'textcolor','k');


%% Consensus avec STatis etude multi-blocs
VEE=struct;bmus2=struct;WW=struct;VEPU=struct;VAP=struct;
VEETilde=struct;bmus1Tilde=struct;WWTilde=struct;VEPUTilde=struct;VAPTilde=struct;
 
for k=1:Appr
    disp(k)
    %%
    for b=1:B
        EnsPartition(b,:)=RFusion(k).Result(b).clust(:,2);
    end
    %%initialisation 
    [VEE.App(k).res,bmus2.App(k).res,WW.App(k).res,VEPU.App(k).res, VAP.App(k).res] = StatisFunction(EnsPartition, nb_class,Critere, 1);
    PureteR(k)= comparaison_partitionOM(Label,bmus2.App(k).res);
    %% Consensus CSTATIS avec la matrice des z_tilde
    for b=1:B
            H_matrix(b).matrix=RFusion(k).Result(b).Z_tilde;
    end 
    [VEETilde(k).res, bmus1Tilde(k).res,WWTilde(k).W,VEPUTilde(k).res, VAPTilde(k).res, STilde(k).res]=StatisFunction( H_matrix, nb_class, Critere, 2 );
    PerfZtilde(k)=comparaison_partitionOM(Label,bmus1Tilde(k).res);
    % CSPA Tilde
    %YTilde = pdist(RFusion(k).Z_Directe,'euclid'); 
    %ZTilde = linkage(YTilde,'ward'); 
    %TTilde = cluster(ZTilde,'maxclust',nb_class);
    %PureteBlocCSPATilde(k)=comparaison_partitionOM(Label,TTilde);
    %% NMF et Weighted NMF
     all_conMatrix = allConMatrix( EnsPartition );
     W=zeros(nb_partition,1);
       for i=1: nb_partition
           W(i)=1/nb_partition;
       end
    M_avg= agreConnMat( all_conMatrix,W);
    bmusNMF(:,k)=NMFCluster( M_avg,Nbclust);
    PureteBlocNMF(k)= comparaison_partitionOM(Label,bmusNMF(:,k));
%% Blocs CSPA
    Y = pdist(M_avg,'euclid'); 
    Z = linkage(Y,'ward'); 
    T = cluster(Z,'maxclust',nb_class);
    PureteBlocCSPA(k)=comparaison_partitionOM(Label,T);
%% CSTATIS & SOM
%% Creation de l'image
    Z=[]; Z_tilde=[];
    for b=1:B
        Z=[Z sqrt(VEE.App(k).res(b))*Data(b).D];
        Z_tilde=[Z_tilde sqrt(VEETilde(k).res(b))*RFusion(k).Result(b).Z_tilde];
    end
    ZZ(k).Z=Z;
    ZZ(k).Z_tilde=Z_tilde;  
    [sMapOR BmusOM] = som_make(ZZ(k).Z_tilde, 'init', 'randinit','training','long' );
    [sMapORZ BmusOMZ] = som_make(ZZ(k).Z, 'init','randinit','training','long' );

    for i=1:6,
        [clustOR1(i).res clcellOR(i).res clcellConsolOR(i).res]=CAHOM(sMapOR,BmusOM,Nbclust,TypeCAH,CAHIndex{i});
        PureteClasse(i).res(k)= comparaison_partitionOM(Label,clustOR1(i).res(:,2));
        [clustOR1Z(i).res clcellORZ(i).res clcellConsolORZ(i).res]=CAHOM(sMapORZ,BmusOMZ,Nbclust,TypeCAH,CAHIndex{i});
        PureteClasseZ(i).res(k)= comparaison_partitionOM(Label,clustOR1Z(i).res(:,2));
    end
    clear  EnsPartition;
end

    mu=VEETilde(k).res;
    for k=2:10
        mu=mu+VEETilde(k).res;
    end 
    
PerfCSTATISR = StatIndex( PureteR );
PerfZtildeR = StatIndex( PerfZtilde );
%PureteBlocCSPAR=StatIndex( PureteBlocCSPA );
PerfCSTATISSOMR=StatIndex(PureteClasse(i).res);
[ StatIndex( PureteR(KK )), StatIndex( PerfZtilde(KK)) StatIndex(PureteClasse(i).res(KK)) StatIndex(PureteClasseZ(i).res(KK))]


%% Consensus avec STatis sur ensemble de diversification
VEE=struct;bmus2=struct;W=struct;VEPU=struct;VAP=struct;
VEETilde=struct;bmus2VEETilde=struct;WVEETilde=struct;VEPUVEETilde=struct;VAPVEETilde=struct;

for ii=8:Appr
    for k=1:Appr
        [DiverEns(k).sMapOR DiverEns(k).BmusOM] = som_make(D, 'init', 'randinit','training','long' );
        [DiverEns(k).clustOR DiverEns(k).clcellOR DiverEns(k).clcellConsolOR]=CAHOM(DiverEns(k).sMapOR,DiverEns(k).BmusOM,Nbclust,TypeCAH,CAHIndex{6});
        DiverEns(k).Cluster=DiverEns(k).clustOR(:,2);
        DiverEns(ii).distorsion(k).res = (som_distortion(DiverEns(k).sMapOR,D))/size(D,2);
        PureteRSOM(k)= comparaison_partitionOM(Label,DiverEns(k).Cluster);
    end 
    % Calcul des mu_b
%     for k=1:Appr
%         DiverEns(ii).Distor(k) = 1/DiverEns(ii).distorsion(k).res;
%     end
     FSOM(ii).Z_tilde=[];
      FSOM(ii).Z_Directe=[];

%     for k=1:Appr
%         DiverEns(ii).mu(k)=DiverEns(ii).Distor(k)/sum(DiverEns(ii).Distor);
%         FSOM(ii).Z_tilde=[FSOM(ii).Z_tilde DiverEns(ii).mu(k)*DiverEns(k).sMapOR.codebook(DiverEns(k).BmusOM,:)];
%         FSOM(ii).Z_Directe=[FSOM(ii).Z_Directe DiverEns(k).sMapOR.codebook(DiverEns(k).BmusOM,:)];
%     end
    
%%    Apprentissage Fusion FSOM
%     [FSOMRes(ii).sMapOR FSOMRes(ii).BmusOM]=som_make(FSOM(ii).Z_tilde, 'init', 'randinit','training','long' );
%     [FSOMRes(ii).sMapORD FSOMRes(ii).BmusOMD]=som_make(FSOM(ii).Z_Directe, 'init', 'randinit','training','long' );
%     [FSOMRes(ii).clustOR FSOMRes(ii).clcellOR FSOMRes(ii).clcellConsolOR]=CAHOM( FSOMRes(ii).sMapOR,FSOMRes(ii).BmusOM,Nbclust,TypeCAH,CAHIndex{6});
%     [FSOMRes(ii).clustORD FSOMRes(ii).clcellORD FSOMRes(ii).clcellConsolORD]=CAHOM( FSOMRes(ii).sMapORD,FSOMRes(ii).BmusOMD,Nbclust,TypeCAH,CAHIndex{6});
%     
%     FSOMRes(ii).Cluster=FSOMRes(ii).clustOR(:,2);
%     FSOMRes(ii).ClusterD=FSOMRes(ii).clustORD(:,2);
%     PureteFSOM(ii)= comparaison_partitionOM(Label,FSOMRes(ii).Cluster);
%     PureteFSOMD(ii)= comparaison_partitionOM(Label,FSOMRes(ii).ClusterD);
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
    [sMapOR BmusOM] = som_make(ZZ(ii).EnsZ_tilde, 'init', 'randinit','training','long' );
    
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
end


PerfM=[StatIndex(PureteBlocNMF(1:10)) StatIndex(PureteBlocWNMF(1:10)) StatIndex(PureteBlocCSPA(1:10)) StatIndex(PureteRGle(1:10)) StatIndex(PureteRGleVEETilde(1:10))]
  
PerfM([4,9,10],:)










  