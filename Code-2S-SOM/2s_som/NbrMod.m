function [ ModLog ] = NbrMod( mat_modalLog )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
for i=1:length(mat_modalLog), 
    ModLog(i)=length(mat_modalLog(i).res); 
end

end

