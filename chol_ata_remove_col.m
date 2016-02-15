function [L11bar, L31bar, L33bar] = chol_ata_remove_col(L, idx) %#codegen
%CHOL_ATA_REMOVE_COL updated the Cholesky factorisation of A'A when a
%column is removed from matrix A. We assume that A has the form
%
% A = [a(1) a(2) ... a(idx-1) a(idx) a(idx+1) ... a(n)]
%
%where a(i) are column-vectors and a(idx) is removed from A to form A_.
%Here is an example in MATLAB code:
%
% > A = rand(140, 12);
% > idx = 5;
% > A_ = [A(:,1:idx-1) A(:,idx+1:end)];
%
%The resulting matrix A_ leads to the following factorisation of A_'A_
%
% A_'A_ = [ L11bar  0
%           L31bar  L33bar ];
%
%This function computes matrices L11bar, L31bar and L33bar given the
%Cholesky factorisation of the original matrix.
%
%The total number of flops to perform this update is 2(n-idx)^2+4(n-idx).
%
%Syntax
%[L11bar, L31bar, L33bar] = CHOL_ATA_REMOVE_COL(L, idx)
%
%Input arguments:
% L     Cholesky factorisation of A'*A (lower triangular)
% idx   Integer index or indices of the columns in A to be removed
%
%
%Example:
%
%  A = randn(150,12);
%  L = chol(A'*A,'lower');
%  cols_to_remove = [3 1 5 10];
%  [L11bar, L31bar, L33bar] = chol_ata_remove_col(L, cols_to_remove);
%  L_updated= [L11bar      zeros(6, 2);
%              L31bar      L33bar];
%
%
%See also:
% chol_ata_append_col, chol_ata_insert_col

% Pantelis Sopasakis

% GIST:
% https://gist.github.com/alphaville/4459f416c3d790b43502

narginchk(2,2);

idx = sort(idx);

L11bar=[];
L31bar=[];
L33bar=[];
L_updated = L;
for i = 1:length(idx),
    [L11bar, L31bar, L33bar] = chol_3587(L_updated, idx(i)-i+1);
    L_updated= [L_updated(1:idx(i)-i,1:idx(i)-i)   zeros(size(L11bar,1), size(L33bar, 2));
                L31bar      L33bar];
end


function [L11bar, L31bar, L33bar] = chol_3587(L, idx)
L11bar=L(1:idx-1, 1:idx-1);
l32 = L(idx+1:end, idx);
L31bar=L(idx+1:end, 1:idx-1);
if ~isempty(l32)
    L33bar=cholupdate2(L(idx+1:end, idx+1:end), l32);
else
    L33bar = [];
end

function L = cholupdate2(L,x)
n = length(x);
for k=1:n
    r = sqrt(L(k,k)^2 + x(k)^2);
    c = r / L(k, k);
    s = x(k) / L(k, k);
    L(k, k) = r;
    if (k<=n-1),
        L(k+1:n,k) = (L(k+1:n,k) + s*x(k+1:n)) / c;
        x(k+1:n) = c*x(k+1:n) - s*L(k+1:n,k);
    end
end


