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
    subplot(2,1,2), plot(M(:,1),M(:,2),'ro',D(:,1),D(:,2),'b+'); 
    title('First two components of map units (o) and data vectors (+)');
    drawnow
  end
  % end of trackplot