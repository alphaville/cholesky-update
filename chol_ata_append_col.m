function [l1, l2, flag, P_tr]=chol_ata_append_col(L, A, c, Ptr)
%CHOL_ATA_APPEND_COL updates the cholesky factorization of matrix A'A when a
%column is appended at the end of A, i.e., 
%
%knowning that the cholesky factorization of A'A is
%
% A'A = LL'
%
%we can efficiently compute the Cholesky factorization of the matrix
%
% [A c]'[A c]
%
%provided that it exists. Denote by A_ = [A c]. The Cholesky factor of
%A_'A_ will have the form
%
%  L_ = [L   0 
%        l1' l2]
%
%This function returns l1 and l2 which are used to construct matrix L_.
%
%The total number of flops needed to perform this update is n^2+3n, where n
%is the column dimension of A.
%
%Syntax:
%[l1, l2] = CHOL_ATA_APPEND_COL(L, A, c);
%[l1, l2, flag] = CHOL_ATA_APPEND_COL(L, A, c);
%[l1, l2, flag, P_tr] = CHOL_ATA_APPEND_COL(L, A, c, Ptr);
%
%Input arguments:
%c     The column which is appended at the end of matrix A
%A     The original matrix A
%L     The Cholesky factor of A'A
%Ptr   Ptr is an optional argument which, when provided, means that for the 
%      original matrix A'*A there is available a permuted Cholesky factorisation 
%      of the form A'*A = PLL'P', where L is a lower diagonal matrix and P is a
%      permutation matrix. Here P is provided as a permutation vector.
%
%Output arguments:
%l1, l2    Vectors which are used to update the Cholesky factorization of 
%          A'A when a column is appended in A. The updated Cholesky factor
%          will be L_ = [L   0 ; l1' l2]
%flag      This flag is set to 0 if the computation has succeded and to 
%          -1 if the new matrix [A c]'[A c] is not positive definite.
%P_tr      When `Ptr` is provided as an input argument (see above) then
%          P_tr is a permutation vector so that the updated perturbed
%          Cholesky factorization will be A_'A_ = P_ L_ L_' P_' and
%          P_tr=P_' (the transpose of P_ is returned).
%
%Example of use:
%
% A = randn(150,7);
% p = [2 4 5 1 6 7 3];
% [~, ptr] = sort(p);
% P=perm_mat(p);
% L = chol(A(:,ptr)'*A(:,ptr),'lower');
% c = rand(150,1);
% [l1, l2, flag, p_tr] = chol_ata_append_col(L, A, c, ptr);
% L_ = [L      zeros(size(L,1),1) 
%       l1'    l2];
% A_ = [A c];
% 
%
%
%See also:
%chol_ata_insert_col, chol_ata_remove_col

% Pantelis Sopasakis

% GIST:
% https://gist.github.com/alphaville/4459f416c3d790b43502

narginchk(3, 4);
nargoutchk(2,4);
if nargin<4,
    P_tr = [];
    l1 = L\(A'*c);
else
    P_tr = [Ptr length(Ptr)+1];
    l1 = L\(A(:,Ptr)'*c);
end


l2 = c'*c - l1'*l1;
if l2>1e-8,
    flag = 0;
    l2 = sqrt(l2);
else
    flag = -1;
    l2 = [];
end