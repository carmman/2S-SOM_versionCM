
function R = comparaison_partitionOM(p1, p2)

% COMPARAISON_PARTITION : calcule diff�rents indices de comparaison de
%                         partititions
% Input
%   p1    (vector)  groupe de chaque individu dans la partition 1
%                   (size dlen x 1)
%   p2    (vector)  groupe de chaque individu dans la partition 2
%                   (size dlen x 1)
% Note
%   Les numeros des groupes doivent etre strictement positifs.
%
% Output
%   R     (struct)  la structure utilisee pour le resultat comporte les
%                   champs suivants :
%                   - contingence  (matrix)
%                   - comptage     (struct)  comparaison des accords et des
%                                            desaccords entre les deux
%                                            partitions (nombre de paires
%                                            d'objets)
%                                            . N00 : separees dans les 2
%                                            . N11 : groupees dans les 2
%                                            . N01 : separees dans p1 mais
%                                                    groupees dans p2
%                                            . N10 : groupees dans p1 mais
%                                                    separees dans p2
%                   - entropie     (struct)  . partition1 : entropie de la
%                                                           partition p1
%                                            . partition2 : entropie de la
%                                                           partition p1
%                                            . information_mutuelle%
%                   - indice       (struct)  . jaccard  : indice de Jaccard
%                                            . rand     : indice de Rand
%                                            . tanimoto : indice de
%                                                         Tanimoto
%                                            . VI       : variation
%                                                         d'information
%                   - partition    (struct)  . partition1 : p1
%                                            . partition2 : p2
%
% References
% - Guerif S. "Reduction de Dimension en Apprentissage Numerique Non
%   Supervise". Thèse de doctorat, Université Paris 13, Dec. 2006.
% - Hubert L., Arabie P. "Comparing Partitions". Journal of Classification,
%   2(1): 193-218, Dec. 1985.
% - Meila M. "Comparing clusterings - an information based distance". in
%   print, 2006.
% - Meila M. "Comparing clusterings". UW Statistics Technical Report 418
%   and COLT 03, 2003.
%

R = struct( 'contingence', [], ...
            'comptage'   , struct('N00',        [], 'N11',        [],  ...
                                  'N01',        [], 'N10',        []), ...
            'entropie'   , struct('partition1', [], 'partition2', [],  ...
                                  'information_mutuelle',         []), ...
            'indice'     , struct('jaccard',    [], 'AdusjtedRand',       [], 'rand',       [],  ...
                                  'nmi',    [],'mi',    [], 'nvi',       [],  ...
                                  'tanimoto',   [], 'VI',         []), ...
            'partition'  , struct('partition1', [], 'partition2', []), ...
            'Performance'  ,struct ('Precision', [], 'Recall', [], 'Fmeasure',[],'Accuracy', []));



dlen = length(p1);

%
% table de contingence
T = zeros(max(p1),max(p2));
for i = 1:max(p1)
   x = find(p1 == i);
   for j = 1:max(p2(x))
       T(i,j) = sum(p2(x) == j);
   end
end
%
% utilisation des formules de linearisation (Hubert & Arabie, 1985)
N00 = 0.5 * ( dlen*dlen + sum(T(:).*T(:)) - sum(sum(T) .* sum(T)) - sum(sum(T') .* sum(T')) );
N11 = 0.5 * sum( T(:) .* T(:) - T(:) );
N01 = 0.5 * ( sum(sum(T').*sum(T')) - sum(T(:).*T(:)) );
N10 = 0.5 * ( sum(sum(T).*sum(T)) - sum(T(:).*T(:)) );
%
% calculs auxiliaire pour evaluer l'entropie et l'information mutuelle
proba1  = sum(T)  / dlen;
proba2  = sum(T') / dlen;
proba   = T       / dlen;       proba   = proba(:);
produit = (proba2' * proba1);   produit = produit(:);
%
% preparation des resultats
R.contingence  = T;

R.comptage.N00 = N00;
R.comptage.N11 = N11;
R.comptage.N01 = N01;
R.comptage.N10 = N10;

R.entropie.partition1 = - sum(proba1(proba1 ~= 0) .* log(proba1(proba1 ~= 0)));
R.entropie.partition2 = - sum(proba2(proba2 ~= 0) .* log(proba2(proba2 ~= 0)));

aux = proba .* produit;
R.entropie.information_mutuelle = ...
    sum(proba(aux ~= 0) .* log(proba(aux ~= 0) ./ produit(aux ~= 0)));

R.indice.jaccard  =     (N11      ) / ( N11          + N10 + N01);
R.indice.rand     =     (N11 + N00) / ( N11 + N00    + N10 + N01);
R.indice.AdusjtedRand=valid_RandIndex(p1, p2);
R.indice.tanimoto = 0.5*(N11 + N00) / ((N11 + N00)/2 + N10 + N01);
R.indice.VI       = R.entropie.partition1 + R.entropie.partition2 ...
                  - 2 * R.entropie.information_mutuelle;
[R.indice.nmi R.indice.mi]=nmi(p1, p2);  
R.indice.nvi=nvi(p1, p2); 

R.partition1 = p1;
R.partition2 = p2;

%% calcul de la performance d'un algorithme

R.Performance.Precision=   (N11) / (N11+ N01);
R.Performance.Recall=      (N11)/(N11+ N10);
R.Performance.Fmeasure=    (2*R.Performance.Precision*R.Performance.Recall)/(R.Performance.Precision+R.Performance.Recall); 
R.Performance.Accuracy=    sum(max(R.contingence))/dlen ; 

 
                 
