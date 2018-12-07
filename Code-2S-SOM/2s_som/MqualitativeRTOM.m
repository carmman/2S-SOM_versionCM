    function [ Mqual] = MqualitativeRTOM( F,H, Vect2 )
%   UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 Mqual=zeros(size(H,1),sum(Vect2));
 teta=0.0005;
 deb=1;
 for i =1:length(Vect2)
     %disp(i)
     fin=sum(Vect2(1:i)); 
     mat=F(:,deb:fin)';
     Mpartiel= Mqual(:,deb:fin);
     %cette matrice est à utiliser dans la 3ème condition de l'article
     %mat2=Mqual2(:,deb:fin);
     [A B]=max(mat);
     somme=sum(mat);
     for j=1: length(B)
        %chercher lesquels des neurones satisfont la première condition
       if((somme(j)-A(j))<=A(j))
           %j represente le nombre de référent, B(j), la modalité maximale
           Mpartiel(j,B(j))=1/sqrt(2);
       else
       %générer un nombre aléatoire
       %teta2=rand;
       %if(teta2>teta)
          Mpartiel(j,B(j))=1/sqrt(2);
       %else
           %ici il faut sauvegarder l'état ultérieur de la matrice M, donc
           %la modalité précédente qu'on trouve dans Mqual2
           %pos=randi(Vect2(i),1);
           %Mpartiel(j,pos)=1/sqrt(2);
       %end
       end
     end
     Mqual(:,deb:fin)=Mpartiel;
     deb=fin+1;
 end
end

