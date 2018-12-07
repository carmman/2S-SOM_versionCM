function [A Perf] = StatIndex( data1 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% moyenne et variance d'indices de validations
for j=1:size(data1,2)
            data.rand(j)=data1(j).indice.rand;
            data.jaccard(j)=data1(j).indice.jaccard;
            data. tanimoto(j)=data1(j).indice.tanimoto;
            data.VI(j)=data1(j).indice.VI;
            data.nmi(j)=data1(j).indice.nmi;
            data.nvi(j)=data1(j).indice.nvi;
            data.Precision(j)=data1(j).Performance.Precision;
            data.Recall(j)=data1(j).Performance.Recall;
            data.Fmeasure(j)=data1(j).Performance.Fmeasure;
            data.Accuracy(j)=data1(j).Performance.Accuracy;
end

Perf=[data.Precision' data.Recall' data.Fmeasure' data.Accuracy' data.rand' data.jaccard' data.tanimoto' data.VI' data.nmi' data.nvi'];
            A=[mean(data.Precision) var(data.Precision);
            mean(data.Recall) var(data.Recall);
            mean(data.Fmeasure) var(data.Fmeasure) ;
            mean(data.Accuracy) var(data.Accuracy);
            mean(data.rand) var(data.rand);
            mean(data.jaccard) var(data.jaccard);
            mean(data.tanimoto) var(data.tanimoto);
            mean(data.VI) var(data.VI);
            mean(data.nmi) var(data.nmi);
            mean(data.nvi) var(data.nvi)];

            
            
            
            
            
            
            
            
            
            
            
            

