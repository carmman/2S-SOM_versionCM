function [distorsion Result] = Distorsion( Result, D,Nbclust)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
Dim=size(Result);distorsion=zeros(Dim(1),Dim(2));
for i=1:Dim(1)
    for j=1:Dim(2)
        Dist=0.0;
        for k=1:size(Result(i,j).res,2)
            Dist=Dist+som_distortion(Result(i,j).res(k).sMap,D)/size(D,2);
            [Result(i,j).res(k).cl Result(i,j).CellCl(:,k) Result(i,j).CellConSol(:,k)]=CAHOM(Result(i,j).res(k).sMap,Result(i,j).res(k).bmus,Nbclust,'My','ward');
        end 
        distorsion(i,j)=Dist/10;
    end 
end 
end

