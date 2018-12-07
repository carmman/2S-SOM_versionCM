function [ Mordi ] = Mordinal( F, Vect2, mat_modal)
%MORDINAL Summary of this function goes here
%   Detailed explanation goes here
%mat_modal: elle presente les vraies modalit�s pour chaque variables
%ordinales, supposant que c'est une matrice(A*B) tel que A=nb variable ordinales
%et B est le nb modalit� de la variable ayant le maximum de nb modalit�
 Dim=size(F);
 Mordi=zeros(Dim(1),Dim(2));
 deb=1;
 for i =1:length(Vect2)
     vec=mat_modal(i).res/sum(mat_modal(i).res); %avec contient les valeurs de la modalit� de la variable i
     fin=sum(Vect2(1:i)); 
     mat=F(:,deb:fin)';
     mat2=zeros(Dim(1),length(vec)); %la remplir au fur et � mesure
   for j=1:Dim(1)%cette boucle pr�sente le nb de referents
       modal=0;
       for k=1:size(mat,1)%cette boucle pr�sente le nb-modalit�
         modal=modal+k*mat(k,j);
       end
       modal=round(modal);
       disp(modal)
       %ici il faut voir on est proche � quelle modalit�,
       %[a b]=min(vec2);% voir quelle modalit� est plus proche
       disp(j)
       mat2(j,modal)=1/sqrt(2);
   end
 Mordi(:,deb:fin)=mat2;
 deb=fin+1;
 end
end


