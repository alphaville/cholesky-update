% GIST:
% https://gist.github.com/alphaville/4459f416c3d790b43502


%% First test
clear;

m = 7;
for r=1:100,
    A = randn(150,m);                 % random rectangular matrix
    L = chol(A'*A,'lower');           % cholesky of original matrix
    
    
    for idx = 1:m,
        % remove just one column at a time
        [L11bar, L31bar, L33bar] = chol_ata_remove_col(L, idx);
        
        L_updated= [L11bar      zeros(idx-1,size(A,2)-idx);
            L31bar      L33bar];
        
        A_removed_col = [A(:,1:idx-1) A(:,idx+1:end)];
        L_correct = chol(A_removed_col'*A_removed_col,'lower');
        
        assert(norm(L_updated-L_correct)<1e-10);
    end
end


%% Now with a fat matrix

for r = 1:100,
    A = randn(50,2000);
    alpha = [24 1:5:21 3 7 2 19 23 2000 1999];
    L = chol(A(:, alpha)'*A(:, alpha),'lower');
    
    idx = [5 6 9 11 13];
    
    [L11bar, L31bar, L33bar] = chol_ata_remove_col(L, idx);
    
    L_updated= [L11bar      zeros(size(L11bar,1),size(L33bar,2));
        L31bar      L33bar];
    
    new_alpha = alpha; new_alpha(idx) = [];
    A_removed_col = A(:,new_alpha);
    
    L_correct = chol(A_removed_col'*A_removed_col,'lower');
    
    assert(norm(L_updated-L_correct)<1e-10);
end
%% Remove absolutely everything

for r = 1:100,
    A = randn(50,2000);
    alpha = [24 1:5:20 19 23];
    L = chol(A(:, alpha)'*A(:, alpha),'lower');
    [L11bar, L31bar, L33bar] = chol_ata_remove_col(L, 1:length(alpha));
    assert(isempty(L11bar));
    assert(isempty(L31bar));
    assert(isempty(L33bar));
end
%% Remove many columns
clear
for r=1:100,
    A = randn(150,12);
    L = chol(A'*A,'lower');
    cols_to_remove = [1 3 5 10];
    cols_to_remove = sort(cols_to_remove);
    sd = setdiff((1:12), cols_to_remove);
    Anew = A(:, sd);
    L_updated = L;
    idx_rem = 0;
    for idx=cols_to_remove,
        [L11bar, L31bar, L33bar] = chol_ata_remove_col(L_updated, idx-idx_rem);
        L_updated= [L11bar      zeros(size(L11bar,1), size(L33bar, 2));
            L31bar      L33bar];
        idx_rem = idx_rem+1;
    end
    assert( norm(Anew'*Anew - L_updated*L_updated')   < 1e-10);
end

%% Remove many one command
clear

for r=1:100,
    A = randn(150,12);
    L = chol(A'*A,'lower');
    cols_to_remove = [3 1 5 10];
    [L11bar, L31bar, L33bar] = chol_ata_remove_col(L, cols_to_remove);
    L_updated= [L11bar      zeros(6, 2);
        L31bar      L33bar];
    
    
    sd = setdiff((1:12), cols_to_remove);
    Anew = A(:, sd);
    assert(    norm(Anew'*Anew - L_updated*L_updated')   < 1e-10 );
end