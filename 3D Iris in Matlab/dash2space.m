function txtOut = dash2space(txtIn)
% This function changes all dashes to spaces. It is usefull to show figure titles.
    
    ind = txtIn=='_' | txtIn=='-';
    
    txtIn(ind) = ' ';
    
    txtOut = txtIn;
end

