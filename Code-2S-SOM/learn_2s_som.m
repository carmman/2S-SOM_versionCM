function [sMap sMap_denorm Result] = learn_2s_som(A,nb_neurone,varargin)
% Cree la carte SOM ou S2-SOM Pour donnees cachees
%
% En entree obligatoire
%
%   A: les donnees cachees
%   nb_neurone: Nombre de neurones 
%
% En option
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
%   sMap: La carte SOM ou S2-SOM.
%
%   sMap_denorm: La carte SOM ou S2-SOM, denormalisee.
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
  
  bool_verbose   = false;
  bool_norm      = false;
  bool_rad       = false;
  bool_trainlen  = false;
  bool_2ssom     = false;
  bool_DimData   = false;
  bool_lambda    = false;
  bool_eta       = false;

  Result         = [];
  
  init_with_make = 1;
  pre_train      = 1;
  
  %recuperer les donnees
  data.data=A;
  label=[1:size(data.data,2)];
  
  %Labelise les donnees
  ListVar={};
  for l=1:length(label)
    ListVar{l}=char(strcat('v ',int2str(label(l))));
  end
  data.colheaders=ListVar;
  
  data_casename='simulation';
  
  % --- CM pour ajouter les arguments 'data_name' et 'comp_names'
  i=1;
  while (i<=length(varargin))
    if ischar(varargin{i})
      switch varargin{i},
        case { 'verbose', '-verbose' },
          bool_verbose = true;
        case { 'data_name' },
          data_casename = varargin{i+1}; i=i+1;
        case { 'comp_names' },
          data.colheaders = varargin{i+1}; i=i+1;
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
        case 'S2-SOM'
          disp('** S2-SOM Active **');
          bool_2ssom = true;
        case 'DimData'
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
          bool_lambda = true;
        case 'eta' 
          eta = varargin{i+1}; i=i+1;
          if length(eta) < 1
            error('eta est de longueur nulle !  Il doit y avoir au moins une valeur')
          end
          bool_eta = true;
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
  if init_with_make
    fprintf(1,'\n-- Initialisation avec SOM_MAKE ... ')
    sMap=som_make(sD_norm.data, ...
                  'munits',   nb_neurone, ...
                  'lattice',  lattice, ...
                  'init',     init, ...
                  'tracking', tracking); % creer la carte initiale avec et effectuer un entrainenemt

  else
    if strcmp(init,'randinit')
      fprintf(1,'\n-- Initialisation avec SOM_RANDINIT ... ')
      sMap=som_randinit(sD_norm.data, ...
                        'munits',   nb_neurone, ...
                        'lattice',  lattice, ...
                        'tracking', tracking); % creer la carte initiale

    elseif strcmp(init,'lininit')
      fprintf(1,'\n-- Initialisation avec SOM_LININIT ... ')
      sMap=som_lininit(sD_norm.data, ...
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

  if pre_train
    tracking_ini = tracking;
    %tracking_ini = 1;
    
    % batchtrain avec radius ...
    if (bool_rad && ~bool_trainlen)
      fprintf(1,'\n-- BATCHTRAIN initial avec radius ... ')
      if tracking_ini, fprintf(1,'\n'); end
      j=1;
      while j<length(rad)
        
        sMap=som_batchtrain(sMap,sD_norm.data,'radius',[rad(j) rad(j+1)],'tracking',tracking_ini);
        j=j+1;
        
      end
    end
    % batchtrain avec trainlen ...
    if (~bool_rad && bool_trainlen) 
      fprintf(1,'\n-- BATCHTRAIN initial avec trainlen ... ')
      if tracking_ini, fprintf(1,'\n'); end
      j=1;
      while j<=length(trlen)
        
        sMap=som_batchtrain(sMap,sD_norm.data,'trainlen',trlen(j),'tracking',tracking_ini);
        j=j+1;
        
      end
    end
    % batchtrain avec radius et trainlen             
    if (bool_rad && bool_trainlen)
      fprintf(1,'\n-- BATCHTRAIN initial avec radius et trainlen ... \n')
      if tracking_ini, fprintf(1,'\n'); end
      if length(rad)==length(trlen)+1
        
        j=1;
        while j<length(rad)
          
          sMap=som_batchtrain(sMap,sD_norm.data,'radius',[rad(j) rad(j+1)],'trainlen',trlen(j),'tracking',tracking_ini);
          j=j+1;
          
        end
      else
        error('vecteur radius doit avoir un element en plus que le vecteur trainlen ')
      end
    end
    
    current_perf = som_distortion(sMap,sD_norm);
    fprintf(1,'--> som_distortion apres entrainement initiale = %s\n', num2str(current_perf));
    
  else
    fprintf(1,'** batchtrain initial non active **\n')
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
    if (bool_lambda && bool_eta && bool_DimData)
      
      best_i   = 0;
      best_j   = 0;
      bestperf = inf;
      
      i_train = 1;
      n_train = length(lambda)*length(eta);
      fprintf(1,[ '\n-- batchtrainRTOM loop for %d lambda and %d eta values:\n', ... 
                  '-- ------------------------------------------------------------------\n' ], ...
              length(lambda), length(eta));
      for i=1:length(lambda)
        for j=1:length(eta)
          fprintf(1,'-- batchtrainRTOM (%d/%d) with lambda=%s and eta=%s ... ',i_train, ...
                  n_train, num2str(lambda(i)),num2str(eta(j)));
          if tracking, fprintf(1,'\n'); end
          
          [Result(i,j).sMap Result(i,j).bmus Result(i,j).Alpha Result(i,j).Beta] = som_batchtrainRTOM( ...
              sMap, sD_norm, ...
              'TypeAlgo','2SSOM', ...
              'DimData',DimData, ...
              'DimBloc',DimBloc, ...
              'lambda', lambda(i), ...
              'eta',eta(j), ...
              'radius',[rad(round(length(rad)/2)) ...
                        rad((round(length(rad)/2))+1)], ...
              'trainlen',trlen(round(length(trlen)/2)), ...
              'tracking',tracking);
          
          current_perf = som_distortion(Result(i,j).sMap,sD_norm);
          fprintf(1,'   --> som_distortion=%s\n', num2str(current_perf));
          %  end
          %end
          % best_i=0;
          % best_j=0;
          % bestperf=inf;
          % for i=1:length(lambda)
          %   for j=1:length(eta)
          %         
          Result(i,j).Perf = current_perf;
          if Result(i,j).Perf < bestperf
            best_i = i;
            best_j = j;
          end
          
          i_train = i_train + 1;
        end
      end
      
      sMap = Result(best_i,best_j).sMap;
      
    else
      error('manque de parametre')
    end
  elseif (bool_lambda || bool_eta || bool_DimData)
    error('mentionnez si vous voulez S2-SOM')
  end
  
  % end
  
  % denormalisation de la Map
  if bool_norm
    sMap_denorm=som_denormalize(sMap,sD_norm.comp_norm);
  else
    sMap_denorm=sMap;
  end
  
  return
