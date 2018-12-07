function [ Res ] = compteur( Data )
%UNTITLED5 Summary of this function goes here
% Detailed explanation goes here
Uniq=unique(Data);Res=zeros(2,length(Uniq));
Res(1,:)=Uniq;
for i=1:length(Uniq)
    Res(2,i)=sum(Data==Uniq(i)); 
end 
end

