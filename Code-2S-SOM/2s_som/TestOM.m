function  [MM] = TestOM( H,P,Dquant,Dqual,Known,vect )

  % Then the sum of vectors in each Voronoi set are calculated (P*D) and the
  % neighborhood is taken into account by calculating a weighted sum of the
  % Voronoi sum (H*). The "activation" matrix A is the denominator of the 
  % equation above.
  %% Update quantitative part of M
  Squant = H*(P*Dquant); 
  Aquant = H*(P*Known(:,1:vect(1)));
  nonzero = find(Aquant > 0); 
  Mquant(nonzero) = Squant(nonzero) ./ Aquant(nonzero); 
  
  %% Update qualitative part of M
  Squal = H*(P*Dqual); 
  Aqual = H*(P*Known(:,vect(1)+1:end));
  nonzero = find(Aqual > 0); 
  F(nonzero) = Squal(nonzero)./ Aqual(nonzero);
  % Update rule on nominale features OM : 
  Mqual=[];
  Vect2=[4 2];
  deb=1; compt=1;
  while compt<length(Vect2)+1,
          fin=sum(Vect2(1:compt));Mq=[];
          [A B]=max(F(:,deb:fin)');
          for i=1:size(Mquant,1)
              Mq(i,B(i))=1/sqrt(2);
          end 
          Mqual= [Mqual Mq];
          compt=compt+1;deb=fin+1;
  end
  MM=[Mquant Mqual];
end

