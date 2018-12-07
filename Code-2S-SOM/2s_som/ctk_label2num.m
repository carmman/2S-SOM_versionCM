function indlabels = ctk_label2num(labels,namesorder)
%
% CTK_LABEL2NUM    : Traduit les labels en chiffres, comme le fait 
%    som_label2num, mais en faisant correspondre la numerotation 
%    avec l'ordre des noms de label passes dans namesorder.
%
% reclass = ctk_label2num(labels,namesorder);
%
% Entrees :
% ---------
% labels     : Tableau sur une colonne de chaines de caracteres
% namesorder : Tableau des noms des labels (en ligne) dans l'ordre 
%              ou on veut les numeroter.
% Sorties :
% ---------
% indlabels : Indices des labels suivant l'ordre defini par namesorder  
%
% Exemple :
%>> labels     = {'toto','tata','titi','toto','titi','tata'}';
%>> namesorder = {'titi','toto','tata'};
%>> ctk_label2num(labels,namesorder)
%ans = 2 3 1 2 1 3

% ________________________________________________________	
% 2011 C. Sorror(bamanith, France)
% --------------------------------------------------------
[class,names] = som_label2num(labels);
names         = names';
nbclass       = max(class);
indlabel      = zeros(length(class),1);
nbnames       = size(namesorder,2);
%
for i=1:nbnames
    if (isempty(namesorder{i})) continue; end
    for j=1:nbclass
        if (isempty(names{j})) continue; end
        if ( strcmp(names{j},namesorder{i}) ) 
           %ma classe i a été numéroté j
           J = find(class==j);
           indlabels(J)=i;
           break;
        end
    end    
end
%
indlabels = indlabels';
%
% --------------------------------------------------------
