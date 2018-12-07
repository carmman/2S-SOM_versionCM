function [MM] = UpdateNumericOM( H,P,D,Known,M)

 % In principle the updating step goes like this: replace each map unit 
 % by the average of the data vectors that were in its neighborhood.
          % The contribution, or activation, of data vectors in the mean can 
          % be varied with the neighborhood function. This activation is given 
          % by matrix H. So, for each map unit the new weight vector is
          %
          %      m = sum_i (h_i * d_i) / sum_i (h_i),  
          % where i denotes the index of data vector.  Since the values of
          % neighborhood function h_i are the same for all data vectors belonging to
          % the Voronoi set of the same map unit, the calculation is actually done
          % by first calculating a partition matrix P with elements p_ij=1 if the
          % BMU of data vector j is i.
          % Then the sum of vectors in each Voronoi set are calculated (P*D) and the
          % neighborhood is taken into account by calculating a weighted sum of the
          % Voronoi sum (H*). The "activation" matrix A is the denominator of the 
          % equation above.
          S = H*(P*D); 
          A = H*(P*Known);
          MM=(S./A); 
          MM(isnan(MM))=M(isnan(MM));
          %M=reshape(M,size(H,1),size(D,2));
end

