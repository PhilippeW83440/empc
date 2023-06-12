using SparseArrays

include("afti16.jl")

#N = 10 # Number of steps (using zero-order hold every 0.05s)
N= 100
x_init = [0.0; 0; 0; 0] # Initial State
x_ref = get_x_ref(N) # Reference trajectory to track

@time H, Ae, be, Ai, bi = getAfti16Model(N, x_init, x_ref)

Ai = sparse(Ai)
Ae = sparse(Ae)

# -----------------------
# Interior Point method: just started ...
# -----------------------

n_vars = size(H)[1]
n_eqs = size(Ae)[1]
n_ineqs = size(Ai)[1]

z0 = zeros(n_vars)
nu0 = zeros(n_eqs)

function f(z; t=1) 
	return (t/2)*z'*H*z-sum(log.(bi - Ai*z))
end

function gradf(z; t=1) 
	d = 1 ./ (bi - Ai*z)
	return t*H*z + Ai'*d
end

function Hessianf(z; t=1) 
	d = 1 ./ (bi - Ai*z)
	diagdsquare = Diagonal(d.^2)
	return t*H + Ai' * diagdsquare * Ai
end

function fvals(z; t=1)
	f = (t/2)*z'*H*z - sum(log.(bi - Ai*z))
	d = 1 ./ (bi - Ai*z)
	gradf = t*H*z + Ai'*d
	diagdsquare = Diagonal(d.^2)
	Hf = t*H + Ai' * diagdsquare * Ai
	return f, gradf, Hf
end

# ---------------------------
# Use a sparse linear solver
# ---------------------------
function solve_KKT(z, nu; t=1)
	Hf = Hessianf(z; t=t)

	A1 = hcat(Hf, Ae')
	A2 = hcat(Ae, zeros(n_eqs, n_eqs))
	A = vcat(A1, A2)

	b = -vcat(gradf(z) + Ae'*nu, Ae*z - be)
	dvar = A \ b # use a sparse linear solver
	dz = dvar[1:n_vars]
	dnu = dvar[n_vars+1:end]
	return dz, dnu
end
