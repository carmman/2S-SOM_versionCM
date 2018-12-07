function [sMap, bmus, Alpha_ck Beta sTrain, qe] = som_batchtrainRTOM(sMap, D, varargin)

%SOM_BATCHTRAIN  Use batch algorithm to train the Self-Organizing Map.
%          OM=som_batchtrainRTOM(sMap, D,'mixed1',[5 2],[4 2])
%          OM=som_batchtrainRTOM(sMap, D,'mixed2',[5 2 2],[4 2],[4 2])
% [sM,sT] = som_batchtrain(sM, D, [argID, value, ...])
%  sM     = som_batchtrain(sM,D);
%  sM     = som_batchtrain(sM,sD,'radius',[10 3 2 1 0.1],'tracking',3);
%  [M,sT] = som_batchtrain(M,D,'ep','msize',[10 3],'hexa');
%  Input and output arguments ([]'s are optional): 
%   sM      (struct) map struct, the trained and updated map is returned
%           (matrix) codebook matrix of a self-organizing map
%                    size munits x dim or  msize(1) x ... x msize(k) x dim
%                    The trained map codebook is returned.
%   D       (struct) training data; data struct
%           (matrix) training data, size dlen x dim
%   [argID, (string) See below. The values which are unambiguous can 
%    value] (varies) be given without the preceeding argID.
%   sT      (struct) learning parameters used during the training
% Here are the valid argument IDs and corresponding values. The values which
% are unambiguous (marked with '*') can be given without the preceeding argID.
%   'mask'       (vector) BMU search mask, size dim x 1
%   'msize'      (vector) map size
%   'radius'     (vector) neighborhood radiuses, length 1, 2 or trainlen
%   'radius_ini' (scalar) initial training radius             
%   'radius_fin' (scalar) final training radius
%   'tracking'   (scalar) tracking level, 0-3 
%   'trainlen'   (scalar) training length in epochs
%   'train'     *(struct) train struct, parameters for training
%   'sTrain','som_train'  = 'train'
%   'neigh'     *(string) neighborhood function, 'gaussian', 'cutgauss',
%                         'ep' or 'bubble'
%   'topol'     *(struct) topology struct
%   'som_topol','sTopol'  = 'topol'
%   'lattice'   *(string) map lattice, 'hexa' or 'rect'
%   'shape'     *(string) map shape, 'sheet', 'cyl' or 'toroid'
%   'weights'    (vector) sample weights: each sample is weighted 
%
% For more help, try 'type som_batchtrain' or check out online documentation.
% See also  SOM_MAKE, SOM_SEQTRAIN, SOM_TRAIN_STRUCT.
%%%%%%%%%%% DETAILED DESCRIPTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% som_batchtrain
%
% PURPOSE
%
% Trains a Self-Organizing Map using the batch algorithm. 
%
% SYNTAX
%
%  sM = som_batchtrain(sM,D);
%  sM = som_batchtrain(sM,sD);
%  sM = som_batchtrain(...,'argID',value,...);
%  sM = som_batchtrain(...,value,...);
%  [sM,sT] = som_batchtrain(M,D,...);
%
% DESCRIPTION
%
% Trains the given SOM (sM or M above) with the given training data
% (sD or D) using batch training algorithm.  If no optional arguments
% (argID, value) are given, a default training is done. Using optional
% arguments the training parameters can be specified. Returns the
% trained and updated SOM and a train struct which contains
% information on the training.
%
% REFERENCES
%
% Kohonen, T., "Self-Organizing Map", 2nd ed., Springer-Verlag, 
%    Berlin, 1995, pp. 127-128.
% Kohonen, T., "Things you haven't heard about the Self-Organizing
%    Map", In proceedings of International Conference
%    on Neural Networks (ICNN), San Francisco, 1993, pp. 1147-1156.
%
% KNOWN BUGS
%
% Batchtrain does not work correctly for a map with a single unit. 
% This is because of the way 'min'-function works. 
%
% REQUIRED INPUT ARGUMENTS
%
%  sM          The map to be trained. 
%     (struct) map struct
%     (matrix) codebook matrix (field .data of map struct)
%              Size is either [munits dim], in which case the map grid 
%              dimensions (msize) should be specified with optional arguments,
%              or [msize(1) ... msize(k) dim] in which case the map 
%              grid dimensions are taken from the size of the matrix. 
%              Lattice, by default, is 'rect' and shape 'sheet'.
%  D           Training data.
%     (struct) data struct
%     (matrix) data matrix, size [dlen dim]
%  
% OPTIONAL INPUT ARGUMENTS 
%
%  argID (string) Argument identifier string (see below).
%  value (varies) Value for the argument (see below).
%
%  The optional arguments can be given as 'argID',value -pairs. If an
%  argument is given value multiple times, the last one is
%  used. The valid IDs and corresponding values are listed below. The values 
%  which are unambiguous (marked with '*') can be given without the 
%  preceeding argID.
%
%  Below is the list of valid arguments: 
%   'mask'       (vector) BMU search mask, size dim x 1. Default is 
%                         the one in sM (field '.mask') or a vector of
%                         ones if only a codebook matrix was given.
%   'msize'      (vector) map grid dimensions. Default is the one
%                         in sM (field sM.topol.msize) or 
%                         'si = size(sM); msize = si(1:end-1);' 
%                         if only a codebook matrix was given. 
%   'radius'     (vector) neighborhood radius 
%                         length = 1: radius_ini = radius
%                         length = 2: [radius_ini radius_fin] = radius
%                         length > 2: the vector given neighborhood
%                                     radius for each step separately
%                                     trainlen = length(radius)
%   'radius_ini' (scalar) initial training radius
%   'radius_fin' (scalar) final training radius
%   'tracking'   (scalar) tracking level: 0, 1 (default), 2 or 3
%                         0 - estimate time 
%                         1 - track time and quantization error 
%                         2 - plot quantization error
%                         3 - plot quantization error and two first 
%                             components 
%   'trainlen'   (scalar) training length in epochs
%   'train'     *(struct) train struct, parameters for training. 
%                         Default parameters, unless specified, 
%                         are acquired using SOM_TRAIN_STRUCT (this 
%                         also applies for 'trainlen', 'radius_ini' 
%                         and 'radius_fin').
%   'sTrain', 'som_topol' (struct) = 'train'
%   'neigh'     *(string) The used neighborhood function. Default is 
%                         the one in sM (field '.neigh') or 'gaussian'
%                         if only a codebook matrix was given. Other 
%                         possible values is 'cutgauss', 'ep' and 'bubble'.
%   'topol'     *(struct) topology of the map. Default is the one
%                         in sM (field '.topol').
%   'sTopol', 'som_topol' (struct) = 'topol'
%   'lattice'   *(string) map lattice. Default is the one in sM
%                         (field sM.topol.lattice) or 'rect' 
%                         if only a codebook matrix was given. 
%   'shape'     *(string) map shape. Default is the one in sM
%                         (field sM.topol.shape) or 'sheet' 
%                         if only a codebook matrix was given. 
%   'weights'    (vector) weight for each data vector: during training, 
%                         each data sample is weighted with the corresponding
%                         value, for example giving weights = [1 1 2 1] 
%                         would have the same result as having third sample
%                         appear 2 times in the data
%
% OUTPUT ARGUMENTS
% 
%  sM          the trained map
%     (struct) if a map struct was given as input argument, a 
%              map struct is also returned. The current training 
%              is added to the training history (sM.trainhist).
%              The 'neigh' and 'mask' fields of the map struct
%              are updated to match those of the training.
%     (matrix) if a matrix was given as input argument, a matrix
%              is also returned with the same size as the input 
%              argument.
%  sT (struct) train struct; information of the accomplished training
%
% EXAMPLES
%
% Simplest case:
%  sM = som_batchtrain(sM,D);  
%  sM = som_batchtrain(sM,sD);  
%
% To change the tracking level, 'tracking' argument is specified:
%  sM = som_batchtrain(sM,D,'tracking',3);
%
% The change training parameters, the optional arguments 'train','neigh',
% 'mask','trainlen','radius','radius_ini' and 'radius_fin' are used. 
%  sM = som_batchtrain(sM,D,'neigh','cutgauss','trainlen',10,'radius_fin',0);
%
% Another way to specify training parameters is to create a train struct:
%  sTrain = som_train_struct(sM,'dlen',size(D,1));
%  sTrain = som_set(sTrain,'neigh','cutgauss');
%  sM = som_batchtrain(sM,D,sTrain);
%
% By default the neighborhood radius goes linearly from radius_ini to
% radius_fin. If you want to change this, you can use the 'radius' argument
% to specify the neighborhood radius for each step separately:
%  sM = som_batchtrain(sM,D,'radius',[5 3 1 1 1 1 0.5 0.5 0.5]);
%
% You don't necessarily have to use the map struct, but you can operate
% directly with codebook matrices. However, in this case you have to
% specify the topology of the map in the optional arguments. The
% following commads are identical (M is originally a 200 x dim sized matrix):
%  M = som_batchtrain(M,D,'msize',[20 10],'lattice','hexa','shape','cyl');
%   or
%  M = som_batchtrain(M,D,'msize',[20 10],'hexa','cyl');
%   or
%  sT= som_set('som_topol','msize',[20 10],'lattice','hexa','shape','cyl');
%  M = som_batchtrain(M,D,sT);
%   or
%  M = reshape(M,[20 10 dim]);
%  M = som_batchtrain(M,D,'hexa','cyl');
%
% The som_batchtrain also returns a train struct with information on the 
% accomplished training. This struct is also added to the end of the 
% trainhist field of map struct, in case a map struct was given.
%  [M,sTrain] = som_batchtrain(M,D,'msize',[20 10]);
%  [sM,sTrain] = som_batchtrain(sM,D); % sM.trainhist{end}==sTrain
%
% SEE ALSO
% 
%  som_make         Initialize and train a SOM using default parameters.
%  som_seqtrain     Train SOM with sequential algorithm.
%  som_train_struct Determine default training parameters.
% Copyright (c) 1997-2000 by the SOM toolbox programming team.
% http://www.cis.hut.fi/projects/somtoolbox/
% Version 1.0beta juuso 071197 041297
% Version 2.0beta juuso 101199
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check arguments 
%%
error(nargchk(2, Inf, nargin));  % check the number of input arguments
%RT initialisation de updateRule
UpdateRule='numeric'; %valeur par defaut
% map 
%disp(sMap.codebook)
struct_mode = isstruct(sMap);
if struct_mode, 
  sTopol = sMap.topol;
else  
  orig_size = size(sMap);
  if ndims(sMap) > 2, 
    si = size(sMap); dim = si(end); msize = si(1:end-1);
    M = reshape(sMap,[prod(msize) dim]);
  else
    msize = [orig_size(1) 1]; 
    dim = orig_size(2);    
  end
  sMap   = som_map_struct(dim,'msize',msize);
  sTopol = sMap.topol;
end
[munits dim] = size(sMap.codebook);
% data
if isstruct(D), 
  data_name = D.name; 
  D = D.data;
else 
  data_name = inputname(2); 
end
nonempty = find(sum(isnan(D),2) < dim);
D = D(nonempty,:);                    % remove empty vectors from the data
[dlen ddim] = size(D);                % check input dimension
if dim ~= ddim, 
  error('Map and data input space dimensions disagree.'); 
end
% varargin
sTrain = som_set('som_train','algorithm','batch','neigh', ...
		 sMap.neigh,'mask',sMap.mask,'data_name',data_name);
radius     = [];
tracking   = 1;
weights    = 1; 
i=1; 
while i<=length(varargin), 
  argok = 1; 
  if ischar(varargin{i}), 
    switch varargin{i}, 
     % argument IDs
        case 'TypeAlgo', i=i+1; TypeAlgo = varargin{i};
        case 'mixed1',i=i+3; 
            vect1=varargin{i-2};
            if(ne(length(vect1),2))
                fprintf('error: if mixed1, vect1 must have exactly two elements');
                return;
            end
            vect2=varargin{i-1};
            if(ne(length(vect2),vect1(2)))
             fprintf('error: element in vect2 not equal to vect1(2) ');
                return;
            end
            DimCategorieVar=varargin{i};
            UpdateRule='mixed1';
        case 'mixed2',i=i+4;
            vect1=varargin{i-3};
         if(ne(length(vect1),3))
             fprintf('error: if mixed3, vect1 must have exactly three elements');
             return;
         end
            vect2=varargin{i-2};
            if(ne(length(vect2),vect1(2)))
                fprintf('error: element in vect2 not equal to vect1(2) ');
                return;
            end
            vect3=varargin{i-1};
            if(ne(length(vect3),vect1(3)))
                fprintf('error: element in vect3 not equal to vect1(3) ');
                return;
            end
            mat_modal=varargin{i};
            UpdateRule='mixed2';
            case 'DimData', i=i+1; DimData = varargin{i};
            case 'DimBloc', i=i+1; DimBloc = varargin{i};
            case 'lambda', i=i+1; lambda=varargin{i};
            case 'eta', i=i+1; eta=varargin{i};
            case 'msize', i=i+1; sTopol.msize = varargin{i}; 
            case 'lattice', i=i+1; sTopol.lattice = varargin{i};
            case 'shape', i=i+1; sTopol.shape = varargin{i};
            case 'mask', i=i+1; sTrain.mask = varargin{i};
            case 'neigh', i=i+1; sTrain.neigh = varargin{i};
            case 'trainlen', i=i+1; sTrain.trainlen = varargin{i};
            case 'tracking', i=i+1; tracking = varargin{i};
            case 'weights', i=i+1; weights = varargin{i}; 
            case 'radius_ini', i=i+1; sTrain.radius_ini = varargin{i};
            case 'radius_fin', i=i+1; sTrain.radius_fin = varargin{i};
            case 'radius', 
            i=i+1; 
            l = length(varargin{i}); 
            if l==1, 
                sTrain.radius_ini = varargin{i}; 
            else 
        sTrain.radius_ini = varargin{i}(1); 
        sTrain.radius_fin = varargin{i}(end);
        if l>2, radius = varargin{i}; end
      end 
     case {'sTrain','train','som_train'}, i=i+1; sTrain = varargin{i};
     case {'topol','sTopol','som_topol'}, 
      i=i+1; 
      sTopol = varargin{i};
      if prod(sTopol.msize) ~= munits, 
        error('Given map grid size does not match the codebook size.');
      end
      % unambiguous values
     case {'hexa','rect'}, sTopol.lattice = varargin{i};
     case {'sheet','cyl','toroid'}, sTopol.shape = varargin{i}; 
     case {'gaussian','cutgauss','ep','bubble'}, sTrain.neigh = varargin{i};
     otherwise argok=0; 
    end
  elseif isstruct(varargin{i}) & isfield(varargin{i},'type'), 
    switch varargin{i}(1).type, 
     case 'som_topol', 
      sTopol = varargin{i}; 
      if prod(sTopol.msize) ~= munits, 
        error('Given map grid size does not match the codebook size.');
      end
     case 'som_train', sTrain = varargin{i};
     otherwise argok=0; 
    end
  else
    argok = 0; 
  end
  if ~argok, 
    disp(['(som_batchtrain) Ignoring invalid argument #' num2str(i+2)]); 
  end
  i = i+1; 
end

% take only weights of non-empty vectors
if length(weights)>dlen, weights = weights(nonempty); end
% trainlen
if ~isempty(radius), sTrain.trainlen = length(radius); end

% check topology
if struct_mode, 
  if ~strcmp(sTopol.lattice,sMap.topol.lattice) | ...
	~strcmp(sTopol.shape,sMap.topol.shape) | ...
	any(sTopol.msize ~= sMap.topol.msize), 
    warning('Changing the original map topology.');
  end
end
sMap.topol = sTopol; 

% complement the training struct
sTrain = som_train_struct(sTrain,sMap,'dlen',dlen);
if isempty(sTrain.mask), sTrain.mask = ones(dim,1); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initialize

M        = sMap.codebook;
mask     = sTrain.mask;
trainlen = sTrain.trainlen;
% neighborhood radius
if trainlen==1, 
  radius = sTrain.radius_ini; 
elseif length(radius)<=2,  
  r0 = sTrain.radius_ini; r1 = sTrain.radius_fin;
  radius = r1 + fliplr((0:(trainlen-1))/(trainlen-1)) * (r0 - r1);
else
  % nil
end                                
%  distance between map units in the output space
%  Since in the case of gaussian and ep neighborhood functions, the 
%  equations utilize squares of the unit distances and in bubble case
%  it doesn't matter which is used, the unitdistances and neighborhood
%  radiuses are squared.
Ud = som_unit_dists(sTopol);
Ud = Ud.^2;
radius = radius.^2;
% zero neighborhood radius may cause div-by-zero error
radius(find(radius==0)) = eps; 
% The training algorithm involves calculating weighted Euclidian distances 
% to all map units for each data vector. Basically this is done as
%   for i=1:dlen, 
%     for j=1:munits, 
%       for k=1:dim
%         Dist(j,i) = Dist(j,i) + mask(k) * (D(i,k) - M(j,k))^2;
%       end
%     end
%   end
% where mask is the weighting vector for distance calculation. However, taking 
% into account that distance between vectors m and v can be expressed as
%   |m - v|^2 = sum_i ((m_i - v_i)^2) = sum_i (m_i^2 + v_i^2 - 2*m_i*v_i)
% this can be made much faster by transforming it to a matrix operation:
%   Dist = (M.^2)*mask*ones(1,d) + ones(m,1)*mask'*(D'.^2) - 2*M*diag(mask)*D'
% Of the involved matrices, several are constant, as the mask and data do 
% not change during training. Therefore they are calculated beforehand.
% For the case where there are unknown components in the data, each data
% vector will have an individual mask vector so that for that unit, the 
% unknown components are not taken into account in distance calculation.
% In addition all NaN's are changed to zeros so that they don't screw up 
% the matrix multiplications and behave correctly in updating step.
 switch TypeAlgo,
     case 'NCSOM',
        Known = ~isnan(D);
        W1 = (mask*ones(1,dlen)) .* Known'; 
        D(~Known) = 0;  
        % constant matrices
        WD = 2*diag(mask)*D';    % constant matrix
        dconst = ((D.^2)*mask)'; % constant in distance calculation for each data sample 
        W2 = ones(munits,1)*mask'; 
        D2 = (D'.^2);
        % initialize tracking
        start = clock;
        qe = zeros(trainlen,1); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Action
        % With the 'blen' parameter you can control the memory consumption 
        % of the algorithm, which is in practive directly proportional
        % to munits*blen. If you're having problems with memory, try to 
        % set the value of blen lower. 
        blen = min(munits,dlen);
        % reserve some space
        bmus = zeros(1,dlen); 
        ddists = zeros(1,dlen); 
        for t = 1:trainlen,  
             %#ok<ALIGN>
        % batchy train - this is done a block of data (inds) at a time
        % rather than in a single sweep to save memory consumption. 
        % The 'Dist' and 'Hw' matrices have size munits*blen
        % which - if you have a lot of data - would be HUGE if you 
        % calculated it all at once. A single-sweep version would 
        % look like this: 
        %  Dist = (M.^2)*W1 - M*WD; %+ W2*D2
        %  [ddists, bmus] = min(Dist);
        % (notice that the W2*D2 term can be ignored since it is constant)
        % This "batchy" version is the same as single-sweep if blen=dlen. 
        %% Affectation des observations aux cellules
        i0 = 0;     
        while i0+1<=dlen, 
            inds = [(i0+1):min(dlen,i0+blen)]; i0 = i0+blen;      
            Dist = (M.^2)*W1(:,inds) - M*WD(:,inds)+ W2*D2(:,inds);
            [ddists(inds), bmus(inds)] = min(Dist);
        end  
        %% Complexité de l'algorithme
        if tracking > 0,
            ddists = ddists+dconst; % add the constant term
            ddists(ddists<0) = 0;   % rounding errors...
            qe(t) = mean(sqrt(ddists));
            trackplot(M,D,tracking,start,t,qe);
        end
        %% Calcul des voisinages 
        % neighborhood 
        % notice that the elements Ud and radius have been squared!
        % note: 'bubble' matches the original "Batch Map" algorithm
        switch sTrain.neigh, 
            case 'bubble',   H = (Ud<=radius(t)); 
            case 'gaussian', H = exp(-Ud/(2*radius(t))); 
            case 'cutgauss', H = exp(-Ud/(2*radius(t))) .* (Ud<=radius(t));
            case 'ep',       H = (1-Ud/radius(t)) .* (Ud<=radius(t));
        end 
        %% Actualisation
        P = sparse(bmus,[1:dlen],weights,munits,dlen);
        switch UpdateRule,
            case 'numeric', [ M ] = UpdateNumericOM( H,P,D,Known);
            case 'mixed1',
                % Update qualitative part of M
                Dquant=D(:,1:vect1(1));
                [ Mquant ] = UpdateNumericOM( H,P,Dquant,Known(:,1:vect1(1))); 
                % OK Update qualitative part of M
                Dqual=D(:,(vect1(1)+1):end);
                Squal = H*(P*Dqual); 
                Aqual = H*(P*Known(:,vect1(1)+1:end));
                nonzero = find(Aqual > 0); 
                F(nonzero) = Squal(nonzero)./ Aqual(nonzero); %% les fréquences de chaque modalité
                F=reshape(F,size(H,1),size(Dqual,2));
                [Mqual] = MqualitativeRTOM( F, H, vect2 );
                Mquant=reshape(Mquant,size(H,1),size(Dquant,2));
                Mqual=reshape(Mqual,size(H,1),size(Dqual,2));
                M=[Mquant Mqual];
        end
        end
     case '2SSOM',
	 % pré traitement : creation des différetentes blocs de variables  
         dconst = ((D.^2)*mask)';
         deb=1; BlocData=struct; % structure contenant les blocs de données
         for i=1:size(DimData,2) % iteration sur les  blocs de variables
             fin=sum(DimData(1:i));
            % disp(deb); disp(fin );
             BlocData(i).D=D(:,deb:fin);
             BlocData(i).M=M(:,deb:fin);
             BlocData(i).Dim=DimBloc(i).Dim;
             BlocData(i).mask=mask(deb:fin);
             BlocData(i).Known=~isnan(BlocData(i).D);
             debBloc=1;
             for k=1:length(DimBloc(i).Dim), % Création des blocs de variables
                 finBloc=sum(DimBloc(i).Dim(1:k)); %ok
                 BlocData(i).SousBlocData(k).D=BlocData(i).D(:,debBloc:finBloc); %ok
                 BlocData(i).SousBlocData(k).M=BlocData(i).M(:,debBloc:finBloc); % ok
                 BlocData(i).SousBlocData(k).Known=~isnan(BlocData(i).SousBlocData(k).D); % ok 
                 BlocData(i).SousBlocData(k).mask=BlocData(i).mask(debBloc:finBloc); % ok
                 BlocData(i).SousBlocData(k).W1=(BlocData(i).SousBlocData(k).mask*ones(1,dlen)).* BlocData(i).SousBlocData(k).Known'; % ok 
                 BlocData(i).SousBlocData(k).D(~BlocData(i).SousBlocData(k).Known) = 0; % ok 
                 BlocData(i).SousBlocData(k).WD = 2*diag(BlocData(i).SousBlocData(k).mask)*BlocData(i).SousBlocData(k).D'; % ok 
                 BlocData(i).SousBlocData(k).dconst = (((BlocData(i).SousBlocData(k).D).^2)*BlocData(i).SousBlocData(k).mask)'; % ok 
                 BlocData(i).SousBlocData(k).W2 = ones(munits,1)*BlocData(i).SousBlocData(k).mask'; 
                 BlocData(i).SousBlocData(k).D2 = ((BlocData(i).SousBlocData(k).D)'.^2);
                 debBloc=finBloc+1;
             end
             deb=fin+1;
         end
         % initialize tracking
         start = clock;
         qe = zeros(trainlen,1);  
         blen = min(munits,dlen);
         % reserve some space
         bmus = zeros(1,dlen); 
         ddists = zeros(1,dlen); 
         H=diag(ones(1,blen));
     
         %% Apprentissage
         for t = 1:trainlen,
             % t=1 initialisation
             if t==1,
                 %% Affectation des observations aux cellules
                 %% verification OK
                 % calcul des distances entre les observations et vecteurs référents r
                 i0 = 0;     
                 while i0+1<=dlen, 
                     inds = (i0+1):min(dlen,i0+blen); i0 = i0+blen;
                     Dist=zeros(blen, length(inds));
                     for i=1:size(DimData,2) % distance 
                         for k=1:length(DimBloc(i).Dim),
                             if k==1
                                 Dist =Dist+(((BlocData(i).SousBlocData(k).M).^2)* BlocData(i).SousBlocData(k).W1(:,inds) -  BlocData(i).SousBlocData(k).M* BlocData(i).SousBlocData(k).WD(:,inds)+ BlocData(i).SousBlocData(k).W2*(BlocData(i).SousBlocData(k).D2(:,inds)))/DimBloc(i).Dim(k);
                             else
                               Dist =Dist+(((BlocData(i).SousBlocData(k).M).^2)* BlocData(i).SousBlocData(k).W1(:,inds) -  BlocData(i).SousBlocData(k).M* BlocData(i).SousBlocData(k).WD(:,inds)+ BlocData(i).SousBlocData(k).W2*(BlocData(i).SousBlocData(k).D2(:,inds)))*DimBloc(i).Dim(k)/DimBloc(i).Dim(k);  
                             end 
                         end 
                     end 
                     %Dist=(Dist'*H)';
                     
                     [ddists(inds), bmus(inds)] = min(Dist);
                 end  
                %% Complexité de l'algorithme
                if tracking > 0,
                    ddists = ddists+dconst; % add the constant term
                    ddists(ddists<0) = 0;   % rounding errors...
                    qe(t) = mean(sqrt(ddists));
                    trackplot(M,D,tracking,start,t,qe);
                end
                %% Calcul des voisinages
                % neighborhood 
                % notice that the elements Ud and radius have been squared!
                % note: 'bubble' matches the original "Batch Map" algorithm
                switch sTrain.neigh, 
                    case 'bubble',   H = (Ud<=radius(t)); 
                    case 'gaussian', H = exp(-Ud/(2*radius(t))); 
                    case 'cutgauss', H = exp(-Ud/(2*radius(t))) .* (Ud<=radius(t));
                    case 'ep',       H = (1-Ud/radius(t)) .* (Ud<=radius(t));
                end
                %% Actualisation des poids 
                P = sparse(bmus,1:dlen,weights,munits,dlen);
                deb=1; PoidsBeta=struct; PoidsAlpha=struct;
                for i=1:size(DimData,2)
                    fin=sum(DimData(1:i));
                    switch UpdateRule,
                            case 'numeric', [ BlocData(i).M ] = UpdateNumericOM( H,P,BlocData(i).D,BlocData(i).Known, BlocData(i).M);
                                PoidsBeta(i).SB(1).beta=repmat(ones(1,BlocData(i).Dim(1))./DimData(i),blen,1);
                                PoidsAlpha(i).SB(1).alpha=repmat(ones(1,BlocData(i).Dim(1))./size(DimData,2),blen,1);
                                disp('numeric')
                            case 'mixed1',
                                %disp(i)
                                PoidsBeta(i).SB(1).beta=repmat(ones(1,BlocData(i).Dim(1))./DimData(i),blen,1);
                                PoidsBeta(i).SB(2).beta=repmat(ones(1,BlocData(i).Dim(2))./DimData(i),blen,1);
                                PoidsAlpha(i).SB(1).alpha=repmat(ones(1,BlocData(i).Dim(1))./size(DimData,2),blen,1);
                                PoidsAlpha(i).SB(2).alpha=repmat(ones(1,BlocData(i).Dim(2))./size(DimData,2),blen,1);
                                DebCat=1;
                                % Normalisation par rapport aux modalités
%                                 if DimCategorieVar(i).Dim==0
%                                 else
%                                     for Cat=1:length(DimCategorieVar(i).Dim)
%                                         FinCat=sum(DimCategorieVar(i).Dim(1:Cat));
%                                         PoidsBeta(i).SB(2).beta(:,DebCat:FinCat)=(1/DimCategorieVar(i).Dim(Cat))*PoidsBeta(i).SB(2).beta(:,DebCat:FinCat);
%                                         DebCat=FinCat+1;
%                                     end 
%                                 end
                                [ BlocData(i).SousBlocData(1).M ] = UpdateNumericOM( H,P,BlocData(i).SousBlocData(1).D,BlocData(i).SousBlocData(1).Known,BlocData(i).SousBlocData(1).M);  % quantitative Part
                                Squal = H*(P*BlocData(i).SousBlocData(2).D); 
                                Aqual = H*(P*BlocData(i).SousBlocData(2).Known);
                                nonzero = find(Aqual > 0);
                                clear F;
                                F(nonzero) = Squal(nonzero)./Aqual(nonzero); %% les fréquences de chaque modalité
                                F=reshape(F,size(H,1),DimBloc(i).Dim(2));
                                [BlocData(i).SousBlocData(2).M] = MqualitativeRTOM( F, H, DimCategorieVar(i).Dim);
                                %BlocData(i).SousBlocData(1).M=reshape(BlocData(i).SousBlocData(1).M,size(H,1),size(BlocData(i).SousBlocData(1).D,2));
                                %BlocData(i).SousBlocData(2).M=reshape(BlocData(i).SousBlocData(2).M,size(H,1),size(BlocData(i).SousBlocData(2).D,2));
                                BlocData(i).M=[BlocData(i).SousBlocData(1).M BlocData(i).SousBlocData(2).M];
                    end
                    M(:,deb:fin)=BlocData(i).M;
                   % disp(M)
                    deb=fin+1;
                end
             end
             if t>1,
                 %% Affectation des observations aux cellules en tenant
                 %% compte des poids 
                 i0 = 0;   
                 while i0+1<=dlen, 
                     inds = (i0+1):min(dlen,i0+blen); i0 = i0+blen;
                     Dist=zeros(blen, length(inds));
                     for i=1:size(DimData,2) % distance 
                         if strcmp(UpdateRule,'mixed1')
                             for k=1:2,
                                 if k==1
                                     Dist =Dist+((((PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha)).*(BlocData(i).SousBlocData(k).M).^2)* BlocData(i).SousBlocData(k).W1(:,inds) -  (PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha).*BlocData(i).SousBlocData(k).M* BlocData(i).SousBlocData(k).WD(:,inds)+ (PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha).*BlocData(i).SousBlocData(k).W2*(BlocData(i).SousBlocData(k).D2(:,inds)))/DimBloc(i).Dim(k);
                                 else
                                     Dist =Dist+((((PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha)).*(BlocData(i).SousBlocData(k).M).^2)* BlocData(i).SousBlocData(k).W1(:,inds) -  (PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha).*BlocData(i).SousBlocData(k).M* BlocData(i).SousBlocData(k).WD(:,inds)+ (PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha).*BlocData(i).SousBlocData(k).W2*(BlocData(i).SousBlocData(k).D2(:,inds)))/DimBloc(i).Dim(k);  
                                 end 
                              end  
                         end
                         if strcmp(UpdateRule,'numeric')
                                 k=1;
                                 Dist =Dist+((((PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha)).*(BlocData(i).SousBlocData(k).M).^2)* BlocData(i).SousBlocData(k).W1(:,inds) -  (PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha).*BlocData(i).SousBlocData(k).M* BlocData(i).SousBlocData(k).WD(:,inds)+ (PoidsBeta(i).SB(k).beta.*PoidsAlpha(i).SB(k).alpha).*BlocData(i).SousBlocData(k).W2*(BlocData(i).SousBlocData(k).D2(:,inds)))/DimBloc(i).Dim(k);
                         end
                     end
                   %Dist=(Dist'*H)';
                     [ddists(inds), bmus(inds)] = min(Dist);
                 end  
                %% Complexité de l'algorithme
                if tracking > 0,
                    ddists = ddists+dconst; % add the constant term
                    ddists(ddists<0) = 0;   % rounding errors...
                    qe(t) = mean(sqrt(ddists));
                    trackplot(M,D,tracking,start,t,qe);
                end
                %% Calcul des voisinages
                % neighborhood 
                % notice that the elements Ud and radius have been squared!
                % note: 'bubble' matches the original "Batch Map" algorithm
                switch sTrain.neigh, 
                    case 'bubble',   H = (Ud<=radius(t)); 
                    case 'gaussian', H = exp(-Ud/(2*radius(t))); 
                    case 'cutgauss', H = exp(-Ud/(2*radius(t))) .* (Ud<=radius(t));
                    case 'ep',       H = (1-Ud/radius(t)) .* (Ud<=radius(t));
                end
                %% Actualisation
             
                P = sparse(bmus,1:dlen,weights,munits,dlen);
                deb=1; 
                for i=1:size(DimData,2)
                    fin=sum(DimData(1:i));
                    %Beta(:,deb:fin)=repmat(ones(1,DimData(i))./DimData(i),blen,1);
                    switch UpdateRule,
                            case 'numeric', [ BlocData(i).M ] = UpdateNumericOM( H,P,BlocData(i).D,BlocData(i).Known,BlocData(i).M); %ok
                            case 'mixed1',
                                [ BlocData(i).SousBlocData(1).M ] = UpdateNumericOM( H,P,BlocData(i).SousBlocData(1).D,BlocData(i).SousBlocData(1).Known,BlocData(i).SousBlocData(1).M );  % quantitative Part
                                Squal = H*(P*BlocData(i).SousBlocData(2).D); 
                                Aqual = H*(P*BlocData(i).SousBlocData(2).Known);
                                nonzero = find(Aqual > 0); 
                                clear F;
                                F(nonzero) = Squal(nonzero)./Aqual(nonzero); %% les fréquences de chaque modalité
                                F=reshape(F,size(H,1),DimBloc(i).Dim(2));
                                [BlocData(i).SousBlocData(2).M] = MqualitativeRTOM( F, H, DimCategorieVar(i).Dim);
                                BlocData(i).SousBlocData(1).M=reshape(BlocData(i).SousBlocData(1).M,size(H,1),size(BlocData(i).SousBlocData(1).D,2));
                                BlocData(i).SousBlocData(2).M=reshape(BlocData(i).SousBlocData(2).M,size(H,1),size(BlocData(i).SousBlocData(2).D,2));
                                BlocData(i).M=[BlocData(i).SousBlocData(1).M BlocData(i).SousBlocData(2).M];
                    end
                    M(:,deb:fin)=BlocData(i).M;
                    deb=fin+1;
                end
                %% Poids Alpha et Beta
                %% Calcul des psi_ck et phi_ckj
                %% Distance entre w_c et z_i
                DistAlpha=zeros(blen, dlen);Psi_ck=zeros(blen,size(DimData,2)); Beta=struct;Beta_ck=struct;
                for i=1:size(DimData,2) % distance 
                         % Les alpha
                           % disp(i)
                         %DistBeta=zeros(blen, DimData(i));
                         if strcmp(UpdateRule,'mixed1')
                             for k=1:2,
                                 DistAlpha = DistAlpha+((PoidsBeta(i).SB(k).beta).*(BlocData(i).SousBlocData(k).M).^2)* BlocData(i).SousBlocData(k).W1 -  (PoidsBeta(i).SB(k).beta).*BlocData(i).SousBlocData(k).M* BlocData(i).SousBlocData(k).WD+ (PoidsBeta(i).SB(k).beta).*BlocData(i).SousBlocData(k).W2*(BlocData(i).SousBlocData(k).D2)/DimBloc(i).Dim(k); 
                             end 
                         end
                         if strcmp(UpdateRule,'numeric')
                                 k=1;
                                 DistAlpha = DistAlpha+((PoidsBeta(i).SB(k).beta).*(BlocData(i).SousBlocData(k).M).^2)*BlocData(i).SousBlocData(k).W1 -  (PoidsBeta(i).SB(k).beta).*BlocData(i).SousBlocData(k).M* BlocData(i).SousBlocData(k).WD+ (PoidsBeta(i).SB(k).beta).*BlocData(i).SousBlocData(k).W2*(BlocData(i).SousBlocData(k).D2)/DimBloc(i).Dim(k); 
                         end
                         Psi_ck(:,i)=sum((DistAlpha.*H(:,bmus))');
                         DistAlpha=zeros(blen,dlen);
                         % les beta
                         for c=1:blen,
                            if strcmp(UpdateRule,'mixed1')
                                for k=1:2
                                    if(DimCategorieVar(i).Dim==0) 
                                        Phi=(PoidsAlpha(i).SB(1).alpha(c,i)*((repmat(H(c,bmus),DimData(i),1))').*([BlocData(i).SousBlocData(1).D]-repmat([BlocData(i).SousBlocData(1).M(c,:)],dlen,1)).^2)/DimBloc(i).Dim(1); 
                                    else
                                        Phi=(PoidsAlpha(i).SB(k).alpha(c,i)*((repmat(H(c,bmus),DimData(i),1))').*([BlocData(i).SousBlocData(1).D BlocData(i).SousBlocData(2).D]-repmat([BlocData(i).SousBlocData(1).M(c,:) BlocData(i).SousBlocData(2).M(c,:)],dlen,1)).^2)/DimBloc(i).Dim(k); 
                                    end
                                end 
                                Beta_ck(i).Beta_ck(c,:)=exp(-sum(Phi)/eta)/sum(exp(-sum(Phi)/eta));
                            end
                            if strcmp(UpdateRule,'numeric')
                                Phi=PoidsAlpha(i).SB(k).alpha(c,i)*((repmat(H(c,bmus),DimData(i),1))').*([BlocData(i).SousBlocData(1).D]-repmat([BlocData(i).SousBlocData(1).M(c,:)],dlen,1)).^2; 
                                Beta_ck(i).Beta_ck(c,:)=exp(-sum(Phi)/eta)/sum(exp(-sum(Phi)/eta));
                            end 
                         end
                         Beta(i).Beta_ck=Beta_ck(i).Beta_ck;
                end
                     Alpha_ck=exp(-Psi_ck/lambda)./repmat(sum((exp(-Psi_ck/lambda))'),size(DimData,2),1)';
                     deb=1; 
                     for i=1:size(DimData,2)
                         fin=sum(DimData(1:i));
                         debBloc=1;
                         for k=1:length(DimBloc(i).Dim), % Création des blocs de variables
                             finBloc=sum(DimBloc(i).Dim(1:k)); %ok
                             PoidsBeta(i).SB(k).beta=Beta(i).Beta_ck(:,debBloc:finBloc);
                             PoidsAlpha(i).SB(k).alpha=repmat(Alpha_ck(:,i),1,DimBloc(i).Dim(k));
                             debBloc=finBloc+1;
                         end
                         deb=fin+1;
                     end
             end
         end; % for t = 1:trainlen
 end 

%disp(sMap.codebook)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build / clean up the return arguments
% tracking
if tracking > 0, fprintf(1,'\n'); end
% update structures
sTrain = som_set(sTrain,'time',datestr(now,0));
if struct_mode, 
  sMap = som_set(sMap,'codebook',M,'mask',sTrain.mask,'neigh',sTrain.neigh);
  tl = length(sMap.trainhist);
  sMap.trainhist(tl+1) = sTrain;
else
  sMap = reshape(M,orig_size);
end

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% subfunctions
%%%%%%%%
function [] = trackplot(M,D,tracking,start,n,qe)

  l = length(qe);
  elap_t = etime(clock,start); 
  tot_t = elap_t*l/n;
  fprintf(1,'\rTraining: %3.0f/ %3.0f s',elap_t,tot_t)  
  switch tracking
   case 1, 
   case 2,   
    plot(1:n,qe(1:n),(n+1):l,qe((n+1):l))
    title('Quantization error after each epoch');
    drawnow
   otherwise,
    subplot(2,1,1), plot(1:n,qe(1:n),(n+1):l,qe((n+1):l))
    title('Quantization error after each epoch');
    subplot(2,1,2), plot(M(:,1),M(:,3),'ro',D(:,1),D(:,3),'b+'); 
    title('First two components of map units (o) and data vectors (+)');
    drawnow
  end
  % end of trackplot

  %trackplot(M,D,2,start,n,qe)