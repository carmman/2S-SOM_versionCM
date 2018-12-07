function ctk_showindice(sMap)
%
% CTK_SHOWINDICE    : Permet d'ajouter les indices des referents sur
%    une carte topologique affichee par som_grid avec les coordonnees.
%
% ctk_showindice(sMap);
%
% Entrees :
% ---------
% sMap : Structure de la carte topologique

%_________________________________________________________	
% 2011 C. Sorror(bamanith, France)
%---------------------------------------------------------
W = sMap.codebook;
m=size(W,1);
for i=1:m
    t=text(W(i,1), W(i,2), int2str(i));
    set(t,'FontWeight','bold');
end
%---------------------------------------------------------
