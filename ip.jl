using SparseArrays
using Infiltrator

include("afti16.jl")
include("gpad.jl")

# -----------------------
# Interior Point method
# -----------------------

# outer=5 iters=7
# status N=100 : as fast as ECOS

function f(z::Vector{Float64}) 
	return (1/2)*z'*t_barH*z-sum(log.(bi - Ai*z))
end

function gradf(z::Vector{Float64}) 
	d = 1 ./ (bi - Ai*z)
	return t_barH*z + Ai'*d
end

# Note: the Hessian is Diagonal
function Hessianf(z::Vector{Float64}) 
	d = 1 ./ (bi - Ai*z)
	diagdsquare = Diagonal(d.^2)
	res = t_barH + Ai' * diagdsquare * Ai
	#@infiltrate
	return Diagonal(diag(res))
	return res
end


function residual(z::Vector{Float64}, nu::Vector{Float64})
	return gradf(z) + Ae'*nu, Ae*z - be
end

function norm_residual(z::Vector{Float64}, nu::Vector{Float64})
	g, h = residual(z, nu)
	return norm(vcat(g, h))
end

# ---------------------------
# Use a sparse linear solver
# ---------------------------
function solve_KKT(z::Vector{Float64}, nu::Vector{Float64})
	Hf = Hessianf(z)
	KKT[1:n_vars, 1:n_vars] = Hf
	b = -vcat(gradf(z) + Ae'*nu, Ae*z - be)

	dvar = KKT \ b # use a sparse linear solver

	dz = dvar[1:n_vars]
	dnu = dvar[n_vars+1:end]
	return dz, dnu
end

# Faster
function solve_KKT_by_block(z::Vector{Float64}, nu::Vector{Float64})
	g = gradf(z) + Ae'*nu
	h = Ae*z - be

	Hf = Hessianf(z)
	Hf_inv = inv(Hf) # fast diag matrix inv

	S = Hf_inv * Ae'
	g_tmp = Hf_inv * g
	S = Ae * S

	dnu = S \ (h - Ae * g_tmp)
	dz = -Hf_inv * (g + Ae'*dnu)
	return dz, dnu
end


function solverIP(H::Matrix{Float64}, Ae::Matrix{Float64}, be::Vector{Float64}, Ai::Matrix{Float64}, bi::Vector{Float64})

	Nsteps = length(be)/4 -1

	t_start = time_ns()

	global Ai = sparse(Ai)
	global Ae = sparse(Ae)

	global n_vars = size(H)[1]
	n_eqs = size(Ae)[1]
	n_ineqs = size(Ai)[1]
	
	global KKT = zeros(n_vars + n_eqs, n_vars + n_eqs)
	KKT[n_vars+1:end, 1:n_vars] = Ae
	KKT[1:n_vars, n_vars+1:end] = Ae'
	global KKT = sparse(KKT)
	
	z = zeros(n_vars)
	nu = zeros(n_eqs)
	
	alpha = 0.01
	beta = 0.9
	epsilon = 1e-6

	# #iters = max_outer * max_iters
	max_outer = 5
	max_iters = 7

	t_bar = 3
	mu = 50

	global t_barH = H
	for k_outer in 1:max_outer

		t1 = time_ns()
		for k in 1:max_iters
			#dz, dnu = solve_KKT(z, nu)
			dz, dnu = solve_KKT_by_block(z, nu)
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

		t_bar *= mu
		global t_barH = t_bar * H
	end

	t_stop = time_ns()
	local runtime = (t_stop-t_start)/1.0e6

	re = norm(Ae*z-be)
	ri = maximum(Ai*z-bi)
	printstyled("Solver IP: optval = $(z'*H*z/Nsteps), re=$re, ri=$ri\n"; color = :green)

	return z, runtime, 0
end

#N = 10 # Number of steps (using zero-order hold every 0.05s)
N= 100
x_init = [0.0; 0; 0; 0] # Initial State
x_ref = get_x_ref(N) # Reference trajectory to track

@time H, Ae, be, Ai, bi = getAfti16Model(N, x_init, x_ref)

@time x_opt, u_opt, optval = solveAfti16Model(N, x_init, x_ref)
getPlots(x_opt, u_opt, x_ref, "ref")

@time z_opt, runtime, precomp = solverGPAD(H, Ae, be, Ai, bi; type="Proposed")

@time z_hat, _, _ = solverIP(H, Ae, be, Ai, bi);
