function [sMap, bmus] = SoftSubpsace_SOM( D, varargin)
%UNTITLED2 Summary of this function goes here
%Detailed explanation goes here
%Author : OM
% Data Preprocessing 
[xDim yDim] = size(D);
% varargin
i=1; 
while i<=length(varargin), 
  argok = 1; 
  if ischar(varargin{i}), 
    switch varargin{i}, 
     case 'nb_var', i=i+1; nb_var= varargin{i}; 
     case 'UpdateRule', i=i+1; UpdateRule = varargin{i};
         if strcmp(UpdateRule,'mixed1')
             D(:,1:nb_var(1))=som_normalize(D(:,1:nb_var(1)),'var'); % normalisation des variables N(0,1)
             [ D mat_modal ] = transformData( D,nb_var);              
         end
         for j=1:nb_var(2),
             nb_mod(j)=length(mat_modal(j).res);
         end 
     case 'mask',       i=i+1; mask = varargin{i}; 
     case 'munits',     i=i+1; munits = varargin{i}; 
     case 'msize',      i=i+1; sTopol.msize = varargin{i}; 
                        munits = prod(sTopol.msize); 
     case 'mapsize',    i=i+1; mapsize = varargin{i}; 
     case 'name',       i=i+1; name = varargin{i};
     case 'comp_names', i=i+1; comp_names = varargin{i}; 
     case 'lattice',    i=i+1; sTopol.lattice = varargin{i};
     case 'shape',      i=i+1; sTopol.shape = varargin{i}; 
     case {'topol','som_topol','sTopol'}, 
                        i=i+1; sTopol = varargin{i}; munits = prod(sTopol.msize); 
     case 'neigh',      i=i+1; neigh = varargin{i};
     case 'tracking',   i=i+1; tracking = varargin{i};
     case 'algorithm',  i=i+1; algorithm = varargin{i}; 
     case 'init',       i=i+1; initalg = varargin{i};
     case 'training',   i=i+1; training = varargin{i}; 
      % unambiguous values
     case {'hexa','rect'}, sTopol.lattice = varargin{i};
     case {'sheet','cyl','toroid'}, sTopol.shape = varargin{i}; 
     case {'gaussian','cutgauss','ep','bubble'}, neigh = varargin{i};
     case {'seq','batch','sompak'}, algorithm = varargin{i}; 
     case {'small','normal','big'}, mapsize = varargin{i}; 
     case {'randinit','lininit'}, initalg = varargin{i};
     case {'short','default','long'}, training = varargin{i}; 
     otherwise argok=0; 
    end
  elseif isstruct(varargin{i}) & isfield(varargin{i},'type'), 
    switch varargin{i}(1).type, 
     case 'som_topol', sTopol = varargin{i}; 
     otherwise argok=0; 
    end
  else
    argok = 0; 
  end
  if ~argok, 
    disp(['(som_make) Ignoring invalid argument #' num2str(i+1)]); 
  end
  i = i+1; 
end

%% Apprentissage  
[sMap, bmus]=som_makeRTOM(D,'UpdateRule',UpdateRule, 'nb_var', nb_var,'nb_mod',nb_mod);
%[clust clcell clcellConsol]=CAHOM(sMap,sMap.topol.msize,'name',bmus,2,'My');

end

