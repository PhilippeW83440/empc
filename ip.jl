using SparseArrays
using Infiltrator

include("afti16.jl")
include("gpad.jl")

# -----------------------
# Interior Point method
# -----------------------

# Status: it works, but ... slower than ECOS so far ...

function f(z::Vector{Float64}) 
	return (1/2)*z'*t_barH*z-sum(log.(bi - Ai*z))
end

function gradf(z::Vector{Float64}) 
	d = 1 ./ (bi - Ai*z)
	return t_barH*z + Ai'*d
end

function Hessianf(z::Vector{Float64}) 
	d = 1 ./ (bi - Ai*z)
	diagdsquare = Diagonal(d.^2)
	res = t_barH + Ai' * diagdsquare * Ai
	return res
end

#function fvals(z; t=1)
#	f = (t/2)*z'*H*z - sum(log.(bi - Ai*z))
#	d = 1 ./ (bi - Ai*z)
#	gradf = t*H*z + Ai'*d
#	diagdsquare = Diagonal(d.^2)
#	Hf = t*H + Ai' * diagdsquare * Ai
#	return f, gradf, Hf
#end

function residual(z::Vector{Float64}, nu::Vector{Float64})
	return gradf(z) + Aet*nu, Ae*z - be
end

function norm_residual(z::Vector{Float64}, nu::Vector{Float64})
	g, h = residual(z, nu)
	return norm(vcat(g, h))
end

# ---------------------------
# Use a sparse linear solver
# ---------------------------
function solve_KKT(z::Vector{Float64}, nu::Vector{Float64})
	#Hf = Hessianf(z)
	#A1 = hcat(Hf, Aet)
	#A2 = hcat(Ae, zeros(n_eqs, n_eqs))
	#A = vcat(A1, A2)

	KKT[1:n_vars, 1:n_vars] = Hessianf(z)
	b = -vcat(gradf(z) + Aet*nu, Ae*z - be)

	dvar = KKT \ b # use a sparse linear solver
	dz = dvar[1:n_vars]
	dnu = dvar[n_vars+1:end]
	return dz, dnu
end

# ------------------------------------------------------------------------

alpha = 0.01
beta = 0.9
max_iters = 7
epsilon = 1e-6

#N = 10 # Number of steps (using zero-order hold every 0.05s)
N= 100
x_init = [0.0; 0; 0; 0] # Initial State
x_ref = get_x_ref(N) # Reference trajectory to track

@time H, Ae, be, Ai, bi = getAfti16Model(N, x_init, x_ref)

@time x_opt, u_opt, optval = solveAfti16Model(N, x_init, x_ref)
getPlots(x_opt, u_opt, x_ref, "ref")

@time z, runtime, precomp = solverGPAD(H, Ae, be, Ai, bi; type="Proposed")


n_vars = size(H)[1]
n_eqs = size(Ae)[1]
n_ineqs = size(Ai)[1]


Ai = sparse(Ai)
Aet = Ae'
Ae = sparse(Ae)

KKT = zeros(n_vars + n_eqs, n_vars + n_eqs)
KKT[n_vars+1:end, 1:n_vars] = Ae
KKT[1:n_vars, n_vars+1:end] = Ae'
KKT = sparse(KKT)

z = zeros(n_vars)
nu = zeros(n_eqs)

t_bar = 3
t_barH = H
for k_outer in 1:5
	t1 = time_ns()
	for k in 1:max_iters
		global z
		global nu
		dz, dnu = solve_KKT(z, nu)
		t = 1
		while minimum(bi - Ai*(z+t*dz)) <= 0
			t *= beta
		end
		norm_residual_z_nu = norm_residual(z, nu)
		while norm_residual(z+t*dz, nu+t*dnu) > (1 - alpha*t) * norm_residual_z_nu
			t *= beta
		end
		z = z + t * dz
		nu = nu + t * dnu

		if t < 1e-7
			break
		end
	
		#cost = z'*H*z/N
		#re = norm(Ae*z-be)
		#ri = maximum(Ai*z-bi)
		#println("iter $k: t=$t cost=$cost, re=$re ri=$ri")
	end
	t2 = time_ns()
	local runtime = (t2-t1)/1.0e6
	println("	runtime: $runtime ms")
	local cost = z'*H*z/N
	local re = norm(Ae*z-be)
	local ri = maximum(Ai*z-bi)
	println("Outer iter $k_outer: cost=$cost, re=$re ri=$ri")
	global t_bar *= 50
	global t_barH = t_bar * H
end

cost = z'*H*z/N
re = norm(Ae*z-be)
ri = maximum(Ai*z-bi)
printstyled("cost=$cost, re=$re ri=$ri", color=:green)

