function [ vec ] = Disjonctif(Bmus,Nb_Cell)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
%Var est un vecteur contenant la position de la modalité maximale pour
%chaque referent
DimX=length(Bmus);
Disj=zeros(Nb_Cell,DimX);
%modality=unique(Bmus);
for i=1:Nb_Cell
    vec=zeros(1,DimX);
    for j=1:DimX
        vec(Bmus==i)=1;
    end
    Disj(i,:)=vec;
end

end

