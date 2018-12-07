    function [ Mqual] = Mqualitative( F,H,Mqual2, Vect2 )
%   UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 Mqual=zeros(size(H,1),sum(Vect2));
 teta=0.5;
 deb=1;
 for i =1:length(Vect2)
     fin=sum(Vect2(1:i)); 
     mat=F(:,deb:fin)';
     %cette matrice est à utiliser dans la 3ème condition de l'article
     mat2=Mqual2(:,deb:fin);
     [A B]=max(mat);
     somme=sum(mat);
     
     for j=1: length(B)
        %chercher lesquels des neurones satisfont la première condition
       if((somme(j)-A(j))<A(j))
           %j represente le nombre de référent, B(j), la modalité maximale
           Mqual(j,B(j))=1/sqrt(2);
       else
           %générer un nombre aléatoire
       teta2=rand;
       if(teta2>teta)
           Mqual(j,B(j))=1/sqrt(2);
       else
           %ici il faut sauvegarder l'état ultérieur de la matrice M, donc
           %la modalité précédente qu'on trouve dans Mqual2
           Mqual(j,B(j))=mat2(j,B(j));
       end
       end
     end
            
    % Mqual(:,deb:fin)=Disjonctif( B )*(1/sqrt(2)); 
     deb=fin+1;
 end
end

