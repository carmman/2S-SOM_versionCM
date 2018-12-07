function [ D2 mat_modal FreqModal] = transformDataOM( D,vect)
%TRANSFORMDATA Summary of this function goes here
%   Detailed explanation goes here
%   'vect' contains the information concerning the categoricals data 
if(ne(length(vect),2))
    fprintf('error: vector must have only two elements\n');
    return;
end
[xDim yDim]=size(D);
Ordinal=D(:,yDim-vect(2)+1:end);
for i=1:vect(2)
    mat_modal(i).res=unique(Ordinal(:,i)); 
end 
 %number of quantitative   
nb_quant=vect(1);
%number of categorical variables
nb_cat=vect(2);
if(ne((nb_quant+nb_cat),size(D,2)))
    fprintf('error: total of variable exeeds matrix dimension\n');
    return
end

%transforming the original data matrix.

D2=som_normalize(D(:,1:nb_quant),'var');
%D2=D(:,1:nb_quant);
D3=D(:,nb_quant+1:end);
for i=1: size(D3,2)
    var=D3(:,i);
    modality=unique (var);
    nb_modality=length(modality);
    vec=zeros(length(var),nb_modality);
    for j=1:length(var)
        pos=find(modality==var(j));
        vec(j,pos)=1/sqrt(2);
    end
    D2=[D2 vec];
    FreqModal(i).res= sum(vec)*sqrt(2);
end
    
end

