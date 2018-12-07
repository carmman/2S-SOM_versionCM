function [sMap sMap_denorm Result] = learn_som_carlos(A,nb_neurone,varargin)
    % Cree la carte SOM ou S2-SOM Pour donnees cachees
    % En entree obligatoire
    %   A: les donnees cachees
    %   grille: les Coordonnes geographiques
    %   nb_neurone: Nombre de neurones 
    % En option
    %   radius: en forme de vecteur, chaque deux elements qui ce suivent
    %           constitue une temperature [i..i+1],[i+1..i+2],....
    %   trainlen: en forme de vecteur: chaque element constitue une
    %             itération de l'entraienement. NB:vecteur radius doit 
    %             avoir un element en plus que le vecteur trainlen
    %   tracking: pour visualiser l'apprentissage
    %   S2-SOM: pour faire l'apprentissage avec S2-SOM il faut d'autre 
    %           parametre
    %        DimData: vecteur contenant la dimention de chaque bloc
    %        lambda: vecteur, c'est un hyperparametre pour calculer 
    %                le poids sur les blocs    
    %        eta: vecteur, c'est un hyperparametre pour calculer 
    %             le poids sur les variables    
    % En sortie
    %   sMap: La carte OM ou S2-SOM
    %   bmus: Bmus (best matching units) sur toute la zone
    %   bmus_pixel:(best matching units) par pixel
    %   Detailed explanation goes here
    
    
    tracking=0;
    init='lininit';
    lattice='rect'
    bool_norm=0;   

    Result=[];
    
    
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
    
    sD = som_data_struct(data.data,'name', data_casename,'comp_names', upper(ListVar));
    i=1;
    while (i<=length(varargin) && bool_norm==0)
        if strcmp(varargin{i},'norm')
            bool_norm=1; 
            type_norm=varargin{i+1};

        end
        i=i+1;
   end
    %normalisation
    if bool_norm
        if strcmp(type_norm,'simple')
            sD_norm=som_normalize(sD);
        else
            sD_norm=som_normalize(sD,type_norm);
        end
    else           
        sD_norm =sD;
    end
    
    
    if ~isempty(varargin)
        i=1;
        while i<=length(varargin)
            if strcmp(varargin{i},'init')
                init=varargin{i+1};
            end
            if strcmp(varargin{i},'tracking')
                tracking=varargin{i+1};
            end
            if strcmp(varargin{i},'lattice')
                lattice=varargin{i+1};
            end
            i=i+1;
        end
    end

    %SOM initialisation
    sMap=som_make(sD_norm.data,'munits',nb_neurone, 'lattice', lattice,'init',init,'tracking',tracking);%creer la carte initiale
    bool_rad=0;
    bool_trainlen=0;
    if ~isempty(varargin)
        
        i=1;
        while i<=length(varargin)
            if ischar(varargin{i})
                switch varargin{i}
                    case 'radius'
                       bool_rad=1;
                       loc_rad=i;
                       i=i+1;
                    case 'trainlen' 
                        bool_trainlen=1;
                        loc_trainlen=i;
                        i=i+1;
                    otherwise
                        i=i+1;
                end
            else
                i=i+1;
            end
        end
                    
        % batchtrain avec radius et  trainlen             
        if (bool_rad && ~bool_trainlen) 
            rad=varargin{loc_rad+1};
                    j=1;
                    while j<length(rad)

                        sMap=som_batchtrain(sMap,sD_norm.data,'radius',[rad(j) rad(j+1)]);
                        j=j+1;

                    end
        end
        if (~bool_rad && bool_trainlen) 
            trlen=varargin{loc_trainlen+1};
            j=1;
            while j<=length(trlen)
             sMap=som_batchtrain(sMap,sD_norm.data,'trainlen',trlen(j));
            end

        end
        if (bool_rad && bool_trainlen) 

             rad=varargin{loc_rad+1};
             trlen=varargin{loc_trainlen+1};
             if length(rad)==length(trlen)+1

                    j=1;
                    while j<length(rad)

                        sMap=som_batchtrain(sMap,sD_norm.data,'radius',[rad(j) rad(j+1)],'trainlen',trlen(j));
                        j=j+1;

                    end
             else
                 error('vecteur radius doit avoir un element en plus que le vecteur trainlen ')
             end
        end     
                
                
                
        
        
        
    end
    %S2-SOM
    bool_2ssom=0;
    bool_DimData=0;
    bool_lambda=0;
    bool_eta=0;
    
    if ~isempty(varargin)
        i=1;
        while i<=length(varargin)
            if ischar(varargin{i}) 
                switch varargin{i} 
                    
                    case 'S2-SOM'
                        fprintf('Lancement S2-SOM');
                        bool_2ssom=1;
                        i=i+1;
                        %mettre en bloc
                    case 'DimData'
                        i=i+1;
                        DimData=varargin{i};
                        for di=1:length(DimData)
                            DimBloc(di).Dim=DimData(di) 
                          
                        end
                        bool_DimData=1;
                    case 'lambda' 
                        i=i+1; 
                        lambda=varargin{i};
                        bool_lambda=1;
                    case 'eta' 
                        i=i+1; eta=varargin{i};
                        bool_eta=1;
                    otherwise
                        i=i+1;
                   

                end
            else
                i=i+1;
            end
        end
        
        if (bool_2ssom)
            if(bool_lambda && bool_eta && bool_DimData)
            
            
                for i=1:length(lambda)

                    for j=1:length(eta)
                        [Result(i,j).sMap Result(i,j).bmus Result(i,j).Alpha Result(i,j).Beta]=som_batchtrainRTOM(sMap, sD_norm,'TypeAlgo','2SSOM','DimData',DimData,'DimBloc',DimBloc,'lambda', lambda(i),'eta',eta(j),'radius',[rad(round(length(rad)/2)) rad((round(length(rad)/2))+1)],'trainlen',trlen(round(length(trlen)/2)));

                    end
                end
                best_i=0;
                best_j=0;
                bestperf=inf;
                for i=1:length(lambda)
                    for j=1:length(eta)
        %         
                        Result(i,j).Perf=som_distortion(Result(i,j).sMap,sD_norm);
                        if Result(i,j).Perf<bestperf
                            best_i=i;
                            best_j=j;
                        end
        %        
                    end
                end
                sMap=Result(best_i,best_j).sMap;
            else
                error('manque de parametre')
            end
        else if (bool_lambda || bool_eta || bool_DimData)
                error('mentionnez si vous voulez S2-SOM')
            end
            
        end
        

    end
    %denormalisation
    if bool_norm
        sMap_denorm=som_denormalize(sMap,sD_norm.comp_norm);
    else
        sMap_denorm=sMap;
    end
    
  

    

    
end
