%%
function [ Result ] = LearningProcess2SSOM( PasLambda, PasEta, EndLambda, EndEta,D,TypeAlgo,UpdateRule,DimCategorieVar,DimData, DimBloc,trainlen)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Result=struct;
for lambda=0:EndLambda
    for eta=0:EndEta
        for k=1:10
            disp([lambda eta k])
            x=PasLambda*lambda+1; y=PasEta*eta+1;
            [Result(lambda+1,eta+1).res(k).sMap Result(lambda+1,eta+1).res(k).bmus Result(lambda+1,eta+1).res(k).Alpha Result(lambda+1,eta+1).res(k).Beta]=som_makeRTOM(D,'TypeAlgo',TypeAlgo,'UpdateRule',UpdateRule,'DimCategorieVar',DimCategorieVar,'DimData',DimData, 'DimBloc',DimBloc,'lambda',x,'eta',y,'trainlen',trainlen);
        end 
    end 
end 
end

%%