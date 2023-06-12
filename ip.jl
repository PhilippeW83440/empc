using SparseArrays

include("afti16.jl")

# -----------------------
# Interior Point method
# -----------------------

# Status: it works, but ... slower than ECOS so far ...

function f(z::Vector{Float64}) 
	return (t_bar/2)*z'*H*z-sum(log.(bi - Ai*z))
end

function gradf(z::Vector{Float64}) 
	d = 1 ./ (bi - Ai*z)
	return t_bar*H*z + Ai'*d
end

function Hessianf(z::Vector{Float64}) 
	d = 1 ./ (bi - Ai*z)
	diagdsquare = Diagonal(d.^2)
	return t_bar*H + Ai' * diagdsquare * Ai
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

	A1 = hcat(Hf, Ae')
	A2 = hcat(Ae, zeros(n_eqs, n_eqs))
	A = vcat(A1, A2)

	b = -vcat(gradf(z) + Ae'*nu, Ae*z - be)
	dvar = A \ b # use a sparse linear solver
	dz = dvar[1:n_vars]
	dnu = dvar[n_vars+1:end]
	return dz, dnu
end

# ------------------------------------------------------------------------

alpha = 0.01
beta = 0.3
max_iters = 20
epsilon = 1e-6

#N = 10 # Number of steps (using zero-order hold every 0.05s)
N= 100
x_init = [0.0; 0; 0; 0] # Initial State
x_ref = get_x_ref(N) # Reference trajectory to track

@time H, Ae, be, Ai, bi = getAfti16Model(N, x_init, x_ref)

@time x_opt, u_opt, optval = solveAfti16Model(N, x_init, x_ref)
getPlots(x_opt, u_opt, x_ref, "ref")

Ai = sparse(Ai)
Ae = sparse(Ae)

n_vars = size(H)[1]
n_eqs = size(Ae)[1]
n_ineqs = size(Ai)[1]

z = zeros(n_vars)
nu = zeros(n_eqs)

t_bar = 1
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
		while norm_residual(z+t*dz, nu+t*dnu) > (1 - alpha*t) * norm_residual(z, nu)
			t *= beta
		end
		z = z + t * dz
		nu = nu + t * dnu

		if t < 1e-7
			break
		end
	
		cost = z'*H*z/N
		re = norm(Ae*z-be)
		ri = maximum(Ai*z-bi)
		println("iter $k: t=$t cost=$cost, re=$re ri=$ri")
	end
	t2 = time_ns()
	runtime = (t2-t1)/1.0e6
	println("	runtime: $runtime ms")
	global t_bar *= 50
end
