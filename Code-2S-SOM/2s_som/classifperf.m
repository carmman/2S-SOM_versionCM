function perf = classifperf(sMap,sData,classnames)
%
% CLASSIFPERF       : Calcul la performance, exprimee en pourcentage,
%    de classification d'une carte topologique deja labellisee selon 
%    un vote majoritaire, pour des donnees deja labellisees.
%    
% perf = classifperf(sMap,sData,classnames);
%
% Entrees    :
% ------------
% sMap       : Stucture de la carte topologique deja labellisee
% sData      : Structure des donnees qui doivent deja etre labellisees
% classnames : Vecteur des noms des classes (ou labels)
%
% Sortie :
% --------
% perf   : La performance de classification en pourcentage

% _____________________________________________________________	
% 2011 C. Sorror
%--------------------------------------------------------------
class_ref     = ctk_label2num(sMap.labels,classnames);% N° de classe de referents suivant classnames
class_dataref = class_ref(som_bmus(sMap, sData));      % N° de classe des donnees selon leurs referents associes
class_datalab = ctk_label2num(sData.labels,classnames); % N° de classe des donnees telle quelles sont labellisees
%
Ieq           = find(class_dataref==class_datalab);  % Decomptage des donnees bien classees
perf          = length(Ieq) / size(class_datalab,1); % Calcul de la performance
%
%--------------------------------------------------------------
