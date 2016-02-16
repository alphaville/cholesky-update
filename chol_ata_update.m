function [L_, idx_new, out] = chol_ata_update(A, L, alpha, alpha_)
%CHOL_ATA_UPDATE updated the Cholesky factorisation of A(:,alpha)'A(:,alpha)
%when alpha is replaced by the new set of indices alpha_.
%
%
%This method returns the factorisation of the matrix A_'*A_ where
%
%                   A_ = A(:, idx_new),
%
%where idx_new is a permutation (reordering) of the given target set alpha_,
%given the Cholesky factorisation of A(:, alpha)'*A(:, alpha). Usually it
%is not important to have the factorisation with the indices in alpha_ is a
%particular order.
%
%
%Syntax:
%[L_, idx_new, flag] = CHOL_ATA_UPDATE(A, L, alpha, alpha_)
%[L_, idx_new, flag, flops] = CHOL_ATA_UPDATE(A, L, alpha, alpha_)
%
%
%Input arguments
%A      The original matrix A
%L      The lower-triangular matrix which factorises A(:,alpha)'*A(:,alpha),
%       i.e., A(:,alpha)'*A(:,alpha) = L*L'
%alpha  The set of original indices alpha
%alpha_ The target set of indices
%
%
%Output arguments
%L_         The updated factorisation of A_'*A_
%idx_new    The set of indices which defines A_ = A(:, idx_new). This set
%           is a permutation of alpha_
%flag       A status flag which is 0 if the update has succeeded and -1 if
%           the matrix A_'A_ is singular
%flops      The total flop count for the update
%
%
%See also:
%chol_ata_remove, chol_ata_append

narginchk(4,4)
nargoutchk(2,3)

if size(alpha,2)==1,
    alpha = alpha';
end

if size(alpha_,2)==1,
    alpha_ = alpha_';
end


dbg = true;

if dbg,
    assert( norm(A(:,alpha)'*A(:,alpha) - L*L', Inf)<1e-7, 'Given L is wrong');
end

flag = 0;

idx_to_add = setdiff(alpha_, alpha);
[~, ikill] = setdiff(alpha, alpha_);
idx_new = alpha;
idx_new(ikill) = [];

out.rejected_cols = 0;

%% Estimate flops for removals and additions
out.flops_remove = 0;
s = length(alpha);
for i=length(ikill):-1:1,
    out.flops_remove = out.flops_remove + 2 * (s - ikill(i))^2 + 4 * (s - ikill(i));
    s = s - 1;
end

out.flops_add = 0;
s = length(alpha);
for i = 1:length(idx_to_add),
    out.flops_add = out.flops_add + s^2 + 3*s;
    s = s + 1;
end

out.flops = out.flops_add + out.flops_remove;

% Estimated flops to do the factorization from scratch
% total flops = cholesky factorization + multiplicatio A_alpha'*A_alpha
out.full_chol_flops = length(alpha_)/3 + size(A,1)^2*(2*length(alpha_)-1);

%% Remove columns
t0 = cputime;
out.removals = length(ikill);
if ~isempty( ikill),
    [L11bar, L31bar, L33bar] = chol_ata_remove_col(L, ikill');
    L_ = [ L11bar   zeros(size(L11bar,1), size(L33bar,2))
        L31bar   L33bar ];
else
    L_ = L;
end

A_ = A(:, idx_new);
n_final = length(idx_new) + length(idx_to_add);
j = length(idx_new);

L_ = [L_ zeros(j, n_final-j);
    zeros(n_final-j, n_final)];
t1 = cputime;
% At this point L_ is a lower triangular matrix which factorizes:
% A(:, idx_new)'*A(:, idx_new). See test_chol_ata_update.m (first block)

%% Add columns
out.additions = length(idx_to_add);
for i = 1:length(idx_to_add),
    % for all columns to be added
    col = A(:, idx_to_add(i));
    [l1, l2, flag] = chol_ata_append_col(L_(1:j, 1:j), A_, col);
    if flag==0,
        % If column can be added without making the matrix singular, add it
        % else proceed.
        idx_new = [idx_new idx_to_add(i)];
        L_(j+1, 1:j+1) = [l1', l2];
        j = j + 1;
        A_ = [A_ col];
    else
        out.rejected_cols = out.rejected_cols + 1;
    end
end

m = length(idx_new);
L_ = L_(1:m, 1:m);
t2 = cputime;


% Runtimes
out.time.remove = t1-t0;
out.time.add = t2-t1;
out.time.total = t2-t0;