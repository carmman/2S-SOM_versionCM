    function [ Mqual] = Mqualitative( F,H,Mqual2, Vect2 )
%   UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 Mqual=zeros(size(H,1),sum(Vect2));
 teta=0.5;
 deb=1;
 for i =1:length(Vect2)
     fin=sum(Vect2(1:i)); 
     mat=F(:,deb:fin)';
     %cette matrice est � utiliser dans la 3�me condition de l'article
     mat2=Mqual2(:,deb:fin);
     [A B]=max(mat);
     somme=sum(mat);
     
     for j=1: length(B)
        %chercher lesquels des neurones satisfont la premi�re condition
       if((somme(j)-A(j))<A(j))
           %j represente le nombre de r�f�rent, B(j), la modalit� maximale
           Mqual(j,B(j))=1/sqrt(2);
       else
           %g�n�rer un nombre al�atoire
       teta2=rand;
       if(teta2>teta)
           Mqual(j,B(j))=1/sqrt(2);
       else
           %ici il faut sauvegarder l'�tat ult�rieur de la matrice M, donc
           %la modalit� pr�c�dente qu'on trouve dans Mqual2
           Mqual(j,B(j))=mat2(j,B(j));
       end
       end
     end
            
    % Mqual(:,deb:fin)=Disjonctif( B )*(1/sqrt(2)); 
     deb=fin+1;
 end
end

