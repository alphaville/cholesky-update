%% Test update: only remove columns
% This test is to make sure that for an initial set of indices alpha (in
% shuffled order) and a target set of indices alpha_ (again, not ordered),
% we produce the correct Cholesky factorization.

m = 60;
n = 3000;
for r=1:50,
    A = randn(m,n);
    alpha  = randperm(n, m);
    % Initial cholesky factorization (using columns alpha)
    L = chol(A(:, alpha)'*A(:, alpha),'lower');
    % The target set alpha_ contains some of the original indices, and not
    % in the same order as in alpha.
    alpha_ = alpha(randperm(length(alpha)));
    
    % Check with various lengths of removed columns (remove 1 up to all)
    j=3;
    for j=0:length(alpha)
        [L_, idx_new, out] = chol_ata_update(A, L, alpha, alpha_(1:end-j));
        
        if j==0,
            assert(norm(L-L_, Inf) < 1e-14);
        end
        
        % Check the output status flag
        assert(out.flag==0, 'flag');
        
        % Check that there were not any additions
        assert(out.additions==0, 'additions');
        
        % Check that the number of removals of columns is the one expected
        assert(out.removals==j, ['removals not ' num2str(j+1)]);
        
        % Check whether the resulting updated Choleksy factorization is
        % correct. It should be that L_ is the factorization of
        % A(:,idx_new)'*A(:,idx_new), where idx_new is produced by
        % chol_ata_update.
        assert(norm(L_*L_' - A(:,idx_new)'*A(:,idx_new), Inf) < 1e-9);
    end
end

%% Test update: only add columns
clear;

m = 30;
n = 2000;

for r=1:100,
    A = randn(m,n);
    
    % Select some columns of A (original set: alpha) and factorize
    % A(:,alpha)'*A(:, alpha)
    alpha  = randperm(n, m-10);
    L = chol(A(:, alpha)'*A(:, alpha),'lower');
    
    % Introduce some more columns in alpha. Make sure that none of the columns
    % in alpha are repeated in alpha_
    alpha_c = setdiff(1:2000, alpha);
    alpha_c = alpha_c(randperm(length(alpha_c)));
    
    for n_add = 1 : m-length(alpha)+10,    
        alpha_ = [alpha alpha_c(1:n_add)];        
        [L_, idx_new, out] = chol_ata_update(A, L, alpha, alpha_);
        
        % Check that there were not any removals
        assert(out.removals==0, 'removals');
        
        % Make sure that the number of additions are properly reported
        assert(out.additions==min(n_add, m), 'additions');
        
        % Verify that the result is correct
        assert(norm(L_*L_' - A(:, idx_new)'*A(:, idx_new), Inf)<1e-10);
    end
end