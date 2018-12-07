% comparaison multiple
AnovaTest1=reshape(Res,100,1);
AnovaTest2=reshape(Res',100,1);
subplot(211)
[p,t,st1] = anova1(AnovaTest1,label,'off');
[c,m,h,nms] = multcompare(st1,'estimate','kruskalwallis','display','on');
subplot(212)
[p,t,st2] = anova1(AnovaTest2,label,'off');
[c,m,h,nms] = multcompare(st2,'estimate','kruskalwallis','display','on');

