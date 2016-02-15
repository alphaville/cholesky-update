% GIST:
% https://gist.github.com/alphaville/4459f416c3d790b43502

! rm *~ ./tests/*~ 2> /dev/null



%% Update when a new column is appended in A --> [A c]
clear;
for r=1:80,
    A=randn(150,7);             % random rectangular matrix
    L = chol(A'*A,'lower');     % cholesky of original matrix
    
    
    for i=1:20,
        c=rand(size(A,1), 1);       % column to add
        
        Ac = [A c];                 % new matrix
        
        % compute new Cholesky (provide only L, A and c)
        [l1, l2, flag] = chol_ata_append_col(L, A, c);
        assert(flag==0);       % flag:0 (success)
        assert(~isempty(l2));
        assert(~isempty(l1));
        
        % Construct the updated Cholesky
        L_update = [L zeros(size(L,1),1); l1' l2];
        
        % test
        assert(norm(Ac'*Ac - L_update*L_update')<1e-10);
        A = Ac;
        L = L_update;
    end
end

%% Make matrix singular by adding column
clear;
A = rand(10,100);
alpha = 1:9;
AtA = A(:, alpha)'*A(:, alpha);
L = chol(AtA, 'lower');
assert(norm(L*L' - AtA, Inf)< 1e-10);

c = randn(10,1);
[l1, l2, flag] = chol_ata_append_col(L, A(:, alpha), c);
assert(flag == 0);

L = [L zeros(size(L,1),1); l1' l2];
[~, l2, flag] = chol_ata_append_col(L, [A(:, alpha) c], c);
assert(isempty(l2),'l2 should be empty here');
assert(flag == -1, 'should have failed here!');
%% Update when row is added (just a mini test)
A=randn(20,7);
L = chol(A'*A);
w = rand(size(A,2),1);
L_row = cholupdate(L, w,'+')';
assert(norm([A; w']'*[A; w']-L_row*L_row')<1e-10)

%%
clear;
N=5000;
A=rand(N,100);
L=chol(A'*A,'lower');

c=rand(N,1);
[l1, l2,flag] = chol_ata_append_col(L,A,c);
L_update = [L zeros(size(L,1),1); l1' l2];
assert(flag==0);

d=c+0.0001*randn(N,1);
[~, ~, flag] = chol_ata_append_col(L_update,[A c],d);
assert(flag==0);

%% Appending columns using a permutation matrix
clear
A = randn(150,7);
p = [2 4 5 1 6 7 3];
[~, ptr] = sort(p);
P=perm_mat(p);

assert(norm(P'*(A'*A)*P-A(:,ptr)'*A(:,ptr))<1e-10);
L = chol(A(:,ptr)'*A(:,ptr), 'lower');

c = rand(150,1);

[l1,l2,flag,p_tr]=chol_ata_append_col(L,A,c,ptr);
assert(flag==0);
L_ = [L zeros(size(L,1),1); l1' l2];
[~, p_tr] = sort(p_tr);
P_tr=perm_mat(p_tr);
P_=P_tr;


A_ = [A c];

assert(   norm(  (A_'*A_) - P_*(L_*L_')*P_', Inf )  < 1e-10);


T=L_*L_';
assert( norm((A_'*A_) - T(p_tr,p_tr)',Inf)  < 1e-10);
assert( norm((A_'*A_) - L_(p_tr,:)*L_(p_tr,:)',Inf)  < 1e-10);


%% Another example
clear;
N = 250;
n = 7;
A = randn(N,n);
p = [2 4 5 1 6 7 3];
[~, ptr] = sort(p);

L = chol(A(:,ptr)'*A(:,ptr),'lower');
c = rand(N,1);
[l1, l2, flag, p_tr] = chol_ata_append_col(L, A, c, ptr);
L_ = [L      zeros(size(L,1),1)
    l1'    l2];
A_ = [A c];


P_tr = perm_mat(p_tr);
P_ = P_tr';
% A_' A_ = P_ L_ L_' P_'
% A_' A_ x =b
% P_ L_ L_' P_' x = b
% L_ L_' P_' x = P_' b
% y = P_' x   --->  L_ L_' y = P_' b

AtA = A_'*A_;
AtA2 = (L_'*P_tr)'*(L_'*P_tr);
assert(norm(AtA-AtA2)<1e-10);

%       P_ L_ L_' P_' x = b
% <-->  L_ L_' P_' x = P_' b
% <-->  L_ L_' y = P_' b
%
b = rand(n+1,1);
[~, p_] = sort(p_tr);
y = (L_')\(L_\b(p_tr));
x = P_*y;

xx = chol_ata_solve(L_,b,p_tr);
assert(norm((A_'*A_)\b - x)<1e-10);
assert(norm(xx-x)<1e-12);
