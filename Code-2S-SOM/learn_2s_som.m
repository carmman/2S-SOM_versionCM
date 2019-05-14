function [StsMap sMap_denorm Resultout sMapPTout] = learn_2s_som(A,nb_neurone,varargin)
% Cree la carte SOM ou S2-SOM Pour donnees cachees
%
% Usage:
%
%    [sMap, sMap_denorm, Result, sMapPT] = learn_2s_som(A, nb_neurone, <OPTIOS>)
%
%    St = learn_2s_som(A,nb_neurone, '-struct', <OPTIOS>)
%
% En entree obligatoire
%
%   A: les donnees cachees
%   nb_neurone: Nombre de neurones
%
% En option, elles sont specifies par couples de valeurs, p. exemple:
%
%         'radius', [ 5 1 ], 'trainlen', 20, 'tracking', 1, ...
%
%
%   radius: en forme de vecteur, chaque deux elements qui ce suivent constitue
%           une temperature [i..i+1],[i+1..i+2],....
%   trainlen: en forme de vecteur: chaque element constitue une itération de
%           l'entraienement. NB:vecteur radius doit avoir un element en plus
%           que le vecteur trainlen.
%   tracking: pour visualiser l'apprentissage.
%
%   'S2-SOM': pour faire l'apprentissage avec S2-SOM. Si 'S2-SOM' est
%           specifie alors il faut d'autres parametres:
%
%   DimData: vecteur contenant la dimention de chaque bloc.
%   lambda: vecteur, c'est un hyperparametre pour calculer le poids sur les
%           blocs.
%   eta: vecteur, c'est un hyperparametre pour calculer le poids sur les
%           variables.
%
% En sortie
%
%   sMap: La carte SOM ou S2-SOM au point de meilleur "Perf".
%
%   sMap_denorm: La carte SOM ou S2-SOM, denormalisee.
%
%   iBest = indice dans Result de la carte sMap retournee.
%
%   Result: structure (vecteur) avec les sorties ou resultats de chaque cas
%           entraine (avec une paire distincte de la combinaison entre lambda
%           et eta): sMap, bmus, Alpha, Beta, Perf.
%
%           Champs de Result:
%              sMap: La carte SOM ou S2-SOM du cas.
%              bmus: Bmus (best matching units) sur toute la zone.
%              Alpha: Coefficients Alpha multipliant les groupes.
%              Beta: Coefficients Alpha multipliant les variables au sans de la
%                  Carte Topologique.
%              Perf: parametre "distortion measure for the map", calcule par
%                  la fonction som_distortion.
%
%   (bmus_pixel:(best matching units) par pixel.)
% Detailed explanation goes here


% Valeurs par defaut
tracking       = 0;
init           = 'lininit';
lattice        = 'rect';

% flags et variables associees
bool_verbose        = false;
bool_return_struct  = false;
bool_norm           = false; type_norm     = 'simple';
bool_rad            = false; rad           = [5 1];
bool_trainlen       = false; trlen         = 20;
bool_rad_2s_som     = false; rad_2s_som    = [];
bool_trlen_2s_som   = false; trlen_2s_som  = [];
bool_2ssom          = false;
bool_DimData        = false; DimData       = [size(A,2)];
bool_lambda         = false; lambda        = 1;
bool_eta            = false; eta           = 1000;
bool_parcomp        = false; parcomp_workers = 8; % 8 workers for parallel computing by default, if activated (if bool_parcomp is true)

Result              = struct([]);

bool_init_with_make = true;
bool_pre_training   = true;

%recuperer les donnees
data.data=A;
label=[1:size(data.data,2)];

%Labelise les donnees (affectation apres boucle d'arguments (selon la valeur de DimBloc)
ListVar={};

data_casename='simulation';

% --- CM pour ajouter les arguments 'data_name' et 'comp_names'
i=1;
while (i<=length(varargin))
    if ischar(varargin{i})
        switch lower(varargin{i}),
            case { 'verbose', '-verbose' },
                bool_verbose = true;
            case { 'returnstruct', 'return-struct', 'struct', '-return-struct', '-struct' },
                bool_return_struct = true;
            case { 'data_name' },
                data_casename = varargin{i+1}; i=i+1;
            case { 'comp_names' },
                ListVar = varargin{i+1}; i=i+1;
            case { 'norm' },
                bool_norm = true;
                type_norm = varargin{i+1}; i=i+1;
            case { 'init' },
                init = varargin{i+1}; i=i+1;
            case { 'tracking' },
                tracking = varargin{i+1}; i=i+1;
            case { 'lattice' },
                lattice = varargin{i+1}; i=i+1;
            case 'radius'
                bool_rad = true;
                rad = varargin{i+1}; i=i+1;
            case 'trainlen'
                bool_trainlen = true;
                trlen = varargin{i+1}; i=i+1;
            case 'radius-2s-som'
                bool_rad_2s_som = true;
                rad_2s_som = varargin{i+1}; i=i+1;
            case 'trainlen-2s-som'
                bool_trlen_2s_som = true;
                trlen_2s_som = varargin{i+1}; i=i+1;
            case {'s2-som', '2s-som'},
                disp('** S2-SOM Active **');
                bool_2ssom = true;
            case {'no-s2-som', 'no-2s-som'},
                disp('** S2-SOM Inactive, only SOM training **');
                bool_2ssom = false;
            case 'dimdata'
                DimData = varargin{i+1}; i=i+1;
                for di=1:length(DimData)
                    DimBloc(di).Dim = DimData(di);
                end
                bool_DimData = true;
            case 'lambda'
                lambda=varargin{i+1}; i=i+1;
                if length(lambda) < 1
                    error('lambda est de longueur nulle !  Il doit y avoir au moins une valeur')
                end
                if isscalar(lambda) && lambda <= 0
                    bool_lambda = false;
                else
                    bool_lambda = true;
                end
            case 'eta'
                eta = varargin{i+1}; i=i+1;
                if length(eta) < 1
                    error('eta est de longueur nulle !  Il doit y avoir au moins une valeur')
                end
                if isscalar(eta) && eta <= 0
                    bool_eta = false;
                else
                    bool_eta = true;
                end
            case 'parcomp'
                bool_parcomp = true;
                parcomp_workers = varargin{i+1}; i=i+1;
            case 'ini-with-make'
                bool_init_with_make = true;
            case 'no-ini-with-make'
                bool_init_with_make = false;
            case 'pre-training'
                bool_pre_training   = true;
            case 'no-pre-training'
                bool_pre_training   = false;
            otherwise
                error(sprintf(' *** %s error: argument(%d) ''%s'' inconnu ***\n', ...
                    mfilename, i, varargin{i}));
        end
    else
        error(sprintf(' *** %s error: argument non-string inattendu (en %d-iemme position) ***\n', ...
            mfilename, i));
    end
    i=i+1;
end

if isempty(ListVar),
    kVar = 1;
    for iG = 1:length(DimData),
        szG = DimData(iG);
        for l = 1:szG
            ListVar{kVar,1} = sprintf('Gr%dVar%d', iG, l);
            kVar = kVar + 1;
        end
    end
end

data.colheaders = ListVar;

sD = som_data_struct(data.data,'name', data_casename,'comp_names', upper(ListVar));
% i=1;
% while (i<=length(varargin) && bool_norm==0)
%   if strcmp(varargin{i},'norm')
%     bool_norm=1;
%     type_norm=varargin{i+1};
%   end
%   i=i+1;
% end

%normalisation des donnees
if bool_norm
    fprintf(1,'\n-- Normalisation des donnees selon ''%s'' ...\n', type_norm);
    if strcmp(type_norm,'simple')
        sD_norm=som_normalize(sD);
    else
        sD_norm=som_normalize(sD,type_norm);
    end
else
    fprintf(1,'\n** Pas de normalisation des donnees **\n');
    sD_norm = sD;
end


% if ~isempty(varargin)
%   i=1;
%   while i<=length(varargin)
%     if strcmp(varargin{i},'init')
%       init=varargin{i+1};
%     end
%     if strcmp(varargin{i},'tracking')
%       tracking=varargin{i+1};
%     end
%     if strcmp(varargin{i},'lattice')
%       lattice=varargin{i+1};
%     end
%     i=i+1;
%   end
% end

fprintf(1,[ '\n-- ------------------------------------------------------------------\n', ...
    '-- New 2S-SOMTraining function:\n', ...
    '--   %s (''%s'', ''%s'', ''%s'', ... )\n', ...
    '-- ------------------------------------------------------------------\n' ], ...
    mfilename, init, lattice, data_casename);

%SOM initialisation
if bool_init_with_make
    fprintf(1,'\n-- Initialisation avec SOM_MAKE ... ')
    sMap=som_make(sD_norm, ...
        'munits',   nb_neurone, ...
        'lattice',  lattice, ...
        'init',     init, ...
        'tracking', tracking); % creer la carte initiale avec et effectuer un entrainenemt
    
else
    if strcmp(init,'randinit')
        fprintf(1,'\n-- Initialisation avec SOM_RANDINIT ... ')
        sMap=som_randinit(sD_norm, ...
            'munits',   nb_neurone, ...
            'lattice',  lattice, ...
            'tracking', tracking); % creer la carte initiale
        
    elseif strcmp(init,'lininit')
        fprintf(1,'\n-- Initialisation avec SOM_LININIT ... ')
        sMap=som_lininit(sD_norm, ...
            'munits',   nb_neurone, ...
            'lattice',  lattice, ...
            'tracking', tracking); % creer la carte initiale
        
    else
        error(sprintf(['\n *** %s error: invalid ''init'' option ''%s'' ***\n', ...
            '     Shoud be one between { ''lininit'', ''randinit'' } ***\n' ], ...
            mfilename, init));
    end
    fprintf(1,' <som init END>.\n')
end

% bool_rad=0;
% bool_trainlen=0;
% if ~isempty(varargin)
%
%   i=1;
%   while i<=length(varargin)
%     if ischar(varargin{i})
%       switch varargin{i}
%         case 'radius'
%           bool_rad=1;
%           loc_rad=i;
%           rad=varargin{loc_rad+1};
%           i=i+1;
%         case 'trainlen'
%           bool_trainlen=1;
%           loc_trainlen=i;
%           trlen=varargin{loc_trainlen+1};
%           i=i+1;
%         otherwise
%           i=i+1;
%       end
%     else
%       i=i+1;
%     end
%   end

if bool_pre_training
    pretrain_tracking = tracking;
    %pretrain_tracking = 1;
    
    % batchtrain avec radius ...
    if (bool_rad && ~bool_trainlen)
        fprintf(1,'\n-- BATCHTRAIN initial avec radius ... ')
        if pretrain_tracking, fprintf(1,'\n'); end
        j=1;
        while j<length(rad)
            
            sMap=som_batchtrain(sMap, sD_norm, ...
                'radius',[rad(j) rad(j+1)], ...
                'tracking',pretrain_tracking);
            j=j+1;
            
        end
    end
    % batchtrain avec trainlen ...
    if (~bool_rad && bool_trainlen)
        fprintf(1,'\n-- BATCHTRAIN initial avec trainlen ... ')
        if pretrain_tracking, fprintf(1,'\n'); end
        j=1;
        while j<=length(trlen)
            
            sMap=som_batchtrain(sMap, sD_norm, ...
                'trainlen',trlen(j), ...
                'tracking',pretrain_tracking);
            j=j+1;
            
        end
    end
    % batchtrain avec radius et trainlen
    if (bool_rad && bool_trainlen)
        fprintf(1,'\n-- BATCHTRAIN initial avec radius et trainlen ... \n')
        if pretrain_tracking, fprintf(1,'\n'); end
        if length(rad)==length(trlen)+1
            
            j=1;
            while j<length(rad)
                
                sMap=som_batchtrain(sMap, sD_norm, ...
                    'radius',[rad(j) rad(j+1)], ...
                    'trainlen',trlen(j), ...
                    'tracking',pretrain_tracking);
                j=j+1;
                
            end
        else
            error('vecteur radius doit avoir un element en plus que le vecteur trainlen ')
        end
    end
    sMapPT = sMap;
    
    current_perf = som_distortion(sMap,sD_norm);
    fprintf(1,'--> som_distortion apres entrainement initiale = %s\n', num2str(current_perf));
    
else
    fprintf(1,'** batchtrain initial (pre-training) non active **\n')
end
% end

% %S2-SOM
% bool_2ssom=0;
% bool_DimData=0;
% bool_lambda=0;
% bool_eta=0;
%
% if ~isempty(varargin)
%   i=1;
%   while i<=length(varargin)
%     if ischar(varargin{i})
%       switch varargin{i}
%
%         case 'S2-SOM'
%           disp('** S2-SOM Active **');
%           bool_2ssom=1;
%           i=i+1;
%           %mettre en bloc
%         case 'DimData'
%           i=i+1;
%           DimData=varargin{i};
%           for di=1:length(DimData)
%             DimBloc(di).Dim=DimData(di);
%           end
%           bool_DimData=1;
%         case 'lambda'
%           i=i+1;
%           lambda=varargin{i};
%           if length(lambda) < 1
%             error('lambda est de longueur nulle !  Il doit y avoir au moins une valeur')
%           end
%           bool_lambda=1;
%         case 'eta'
%           i=i+1; eta=varargin{i};
%           if length(eta) < 1
%             error('eta est de longueur nulle !  Il doit y avoir au moins une valeur')
%           end
%           bool_eta=1;
%         otherwise
%           i=i+1;
%
%       end
%     else
%       i=i+1;
%     end
%   end

if (bool_2ssom)
    %if (bool_lambda && bool_eta && bool_DimData)
    if (bool_lambda && bool_eta)
        
        n_lambda = length(lambda);
        n_eta    = length(eta);
        i_train = 1;
        n_train = n_lambda*n_eta;
        
        if ~bool_rad_2s_som
            rad_2s_som =  [rad(round(length(rad)/2)) ...
                rad((round(length(rad)/2))+1)];
        end
        if ~bool_trlen_2s_som
            trlen_2s_som = trlen(round(length(trlen)/2));
        end
        
        fprintf(1,[ '\n-- batchtrainRTOM loop for %d lambda and %d eta values:\n', ...
            '-- ------------------------------------------------------------------\n' ], ...
            n_lambda, n_eta);
        if tracking > 1,
            fprintf(1,'   ... trainlen_2s_som ... %s\n', num2str(trlen_2s_som))
            fprintf(1,'   ... radius_2s_som ..... [%s]\n', join(string(rad_2s_som),', '))
        end
        if bool_parcomp, ticBytes(gcp); end   % POUR CALCUL PARALLELE
        if bool_parcomp
            parcomp_M = parcomp_workers;
        else
            parcomp_M = 1;
        end
        parfor (i=1:n_lambda,parcomp_M)
            for j=1:n_eta
                fprintf(1,'-- batchtrainRTOM (%d/%d) with lambda=%s and eta=%s ... ',(i - 1) * n_eta + j, ...
                    n_train, num2str(lambda(i)),num2str(eta(j)));
                if tracking, fprintf(1,'\n'); end
                
                [Result(i,j).sMap Result(i,j).bmus Result(i,j).Alpha Result(i,j).Beta] = som_batchtrainRTOM( ...
                    sMap, sD_norm, ...
                    'TypeAlgo', '2SSOM', ...
                    'DimData',  DimData, ...
                    'DimBloc',  DimBloc, ...
                    'lambda',   lambda(i), ...
                    'eta',      eta(j), ...
                    'radius',   rad_2s_som, ...
                    'trainlen', trlen_2s_som, ...
                    'tracking', tracking);
                
                Result(i,j).lambda  = lambda(i);
                Result(i,j).eta     = eta(j);
                Result(i,j).DimData = DimData;
                
                current_perf = som_distortion(Result(i,j).sMap,sD_norm);
                fprintf(1,'   --> som_distortion=%s\n', num2str(current_perf));
                
                Result(i,j).Perf = current_perf;
            end
        end
        if bool_parcomp, tocBytes(gcp), end   % POUR CALCUL PARALLELE
        
        % best_i   = 0;
        % best_j   = 0;
        % bestperf = inf;
        % for i=1:n_lambda
        %   for j=1:n_eta
        %       %  end
        %       %end
        %       % best_i=0;
        %       % best_j=0;
        %       % bestperf=inf;
        %       % for i=1:n_lambda
        %       %   for j=1:n_eta
        %       %
        %       if Result(i,j).Perf < bestperf
        %           best_i = i;
        %           best_j = j;
        %           bestperf = Result(i,j).Perf;
        %       end
        %   end
        % end
        %
        % sMap = Result(best_i,best_j).sMap;
        
    else
        error('manque de parametre: specifier les valeurs pour LAMBDA, pour ETA ou pour les deux!')
    end
elseif (bool_lambda || bool_eta)
    error([ '*** %s: PAS DE 2SSOM SPECIFIE MAIS FLAGS (LAMBDA ou ETA) ACTIVE ***\n', ...
        '    mentionnez si vous voulez ''S2-SOM''\n' ], mfilename)
else
    fprintf(1,[ '*** %s: PAS DE 2SSOM SPECIFIE ***\n', ...
        '    mentionnez si vous voulez ''S2-SOM''\n' ], mfilename)
end

% end

clear St
if (bool_2ssom)
    % si 2S-SOM alors best perf sMap
    [BestPerf,iBest] = min(cell2mat({Result.Perf}));
    
    sMap = Result(iBest).sMap;
else
    % sinon, si pas 2S-SOM
    sMap = sMapPT;
end

% denormalisation de la Map
if bool_norm
    sMap_dnrm = som_denormalize(sMap,sD_norm.comp_norm);
else
    sMap_dnrm = sMap;
end

if bool_return_struct
    % Si retour STRUCT
    St.sMap     = sMap;
    St.sD       = sD;
    
    if ~bool_2ssom
        St.bmus = som_bmus(sMap,sD);
    end
    
    if bool_norm
        St.sMap_denorm = sMap_dnrm;
        St.sD_norm = sD_norm;
    end
    
    if (bool_2ssom)
        St.lambda  = Result(iBest).lambda;
        St.eta     = Result(iBest).eta;
        St.bmus    = Result(iBest).bmus;
        St.Alpha   = Result(iBest).Alpha;
        St.Beta    = Result(iBest).Beta;
        St.DimData = Result(iBest).DimData;
        
        St.sMapPT   = sMapPT;
        if bool_norm
            St.sMapPT_denorm = som_denormalize(sMapPT,sD_norm.comp_norm);
        end
        
        St.Result   = Result;
        St.iBest    = iBest;
    end
    
    StsMap = St; % variable de retour
else
    % Sinon, alors retour variables  ... (sMap sMap_denorm Result)
    StsMap      = sMap;
    sMap_denorm = sMap_dnrm;
    Resultout   = Result;
    sMapPTout   = sMapPT;
end

return
