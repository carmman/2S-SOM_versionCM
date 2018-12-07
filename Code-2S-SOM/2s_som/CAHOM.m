function [clust, clcell, clcellConsol, DB_index]=CAHOM(sMap,bmus,Nbclust,TypeCAH,CAHIndex)
% sMap : ta carte 
% msize:  les dimensions de ta cartes exemple : [10 10]
% name : un nom banale pour identifier nommer ta carte exemple : 'OM'
% bmus le bmus initial associ? ? sMap exemple : bah c'est le bmus quoi !!! 
% Nbclust le nombre de classe souhaite exemple : 7 
% CAH contrainte et consolid?e
% TypeCAH='My'; consolide

CahCluster1 = som_cllinkage(sMap,CAHIndex,'connect','neighbors');

%[Dendro1]=som_dendrogram(CahCluster1.tree,sMap,'colorthreshold','default'); 
%set(Dendro1,'LineWidth',2);

DB_index=[];
nb_clust=2;
while nb_clust<20,
    clust=cluster(CahCluster1.tree,'maxclust',nb_clust);
    DB_index(nb_clust)=db_index(sMap,clust);
    nb_clust=nb_clust+1;
end 

DB_index(1)=DB_index(2);
[A,B]=min(DB_index);
plot([1:19],DB_index,'--bs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',3);
%xlabel('Clustering');
%ylabel('Indice de Davie-Bouldin');
clust=cluster(CahCluster1.tree,'maxclust',Nbclust);
clcell=clust;
%%%%%%%%%%%%%% consolidation des cartes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

uniq=unique(clust);
MyCentroids=zeros(length(uniq),size(sMap.codebook,2));
for i=1:length(unique(clust))
MyCentroids(i,:)=mean(sMap.codebook(find(clust==uniq(i)),:));
end
[Clust ClConsol]=som_kmeans('batch',sMap,Nbclust,TypeCAH,MyCentroids);
ClConsol=ClConsol';
clcellConsol=ClConsol;
Carte=[clust,ClConsol]; 
%pour la structure carte%%%%%%% rajout des label%%%

for k=1:Nbclust
    pos1=find(clust==k);
    pos2=find(ClConsol==k);
    for i=1:size(pos1,1)
        clust_obs(bmus==pos1(i))=k; %#ok<AGROW>
    end 
    for j=1:size(pos2,1)
        clust_obsConsol(bmus==pos2(j))=k; %#ok<AGROW>
    end 
end
clust=[clust_obs; clust_obsConsol];
clust=clust';

