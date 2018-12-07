function  [Mquant Mqual] = UpdateNumericNominalOM( H,P,Dquant,Dqual,Known,vect )

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
  Vect2=[4 2];
  [ Mqual] = Mqualitaive( F,H, Vect2 )
  Mquant=reshape(Mquant,size(H,1),size(Dquant,2));
  % Mqual=reshape(Mqual,size(H,1),sum(Vect2));
  % M=[Mquant Mqual];

end

