using SparseArrays

# Related to our stoppping criteria
epsilon_re = 1e-5
epsilon_ri = 1e-5


# -------------------------------------
# Gradient Projected Dual
# -------------------------------------

function solverGPD(H::Matrix{Float64}, Ae::Matrix{Float64}, be::Vector{Float64}, Ai::Matrix{Float64}, bi::Vector{Float64}; max_iters::Int64=10000)

	Nsteps = length(be)/4 -1

	t1 = time_ns()
	Ae = sparse(Ae)
	Ai = sparse(Ai)

	# H is a diagonal matrix: use Diagonal not diagm:
	# https://web.eecs.umich.edu/~fessler/course/551/julia/tutor/03-diag.html
	invH = Diagonal(1 ./ diag(H))  # N=100: 17 microsecs
	Ha = Ae * invH * Ae';
	invHa = inv(Matrix(Ha)) # N=100: 2 ms

	Z0 = invH * Ae' * invHa * be
	Z1 = invH * Ae' * invHa * Ae * invH * Ai' - invH * Ai'

	# --- Very important: makes a huge difference ---
	Ld = Ai * invH * Ai'
	Ld = 0.5 * Diagonal(Ld)
	invLd = inv(Ld)

	#alpha = 1/norm(Ai * invH * Ai')
	lambda = zero(bi)

	t2 = time_ns()
	precomp = (t2-t1)/1.0e6

	z, iters = nothing, max_iters
	for k in 1:max_iters
		z = Z0 + Z1 * lambda
		#lambda = max.(lambda + alpha*(Ai * z - bi), 0)
		lambda = max.(lambda + invLd*(Ai * z - bi), 0)

		re = norm(Ae*z-be)
		ri = maximum(Ai*z-bi)
		if (re < epsilon_re && ri < epsilon_ri)
			iters = k
			break
		end

		#cost = z'*H*z
		#re = norm(Ae*z-be)
		#ri = maximum(Ai*z-bi)
		#println("iter $k: cost=$cost, re=$re ri=$ri")
	end

	t3 = time_ns()
	runtime = (t3-t1)/1.0e6

	re = norm(Ae*z-be)
	ri = maximum(Ai*z-bi)
	println("Solver GPD : precomp=$precomp ms, iters=$iters optval = $(z'*H*z/Nsteps), re=$re, ri=$ri")
	return z, runtime, precomp
end


# -------------------------------------
# Gradient Projected Accelerated Dual
# -------------------------------------

function solverGPAD(H::Matrix{Float64}, Ae::Matrix{Float64}, be::Vector{Float64}, Ai::Matrix{Float64}, bi::Vector{Float64}; max_iters::Int64=10000, type::String="Proposed")

	Nsteps = length(be)/4 -1

	t1 = time_ns()

	Ae = sparse(Ae)
	Ai = sparse(Ai)

	# H is a diagonal matrix: use Diagonal not diagm:
	# https://web.eecs.umich.edu/~fessler/course/551/julia/tutor/03-diag.html
	invH = Diagonal(1 ./ diag(H))  # N=100: 17 microsecs
	Ha = Ae * invH * Ae';
	invHa = inv(Matrix(Ha)) # N=100: 2 ms

	Z0 = invH * Ae' * invHa * be
	Z1 = invH * Ae' * invHa * Ae * invH * Ai' - invH * Ai'

	if (type == "Bemporad")
		Linv = 1/norm(Ai * invH * Ai') # Too bad actually
		Linv = 0.01
	elseif (type == "Giselsson")
		nL = size(Ai)[1]
		L_var = Variable(nL)
		cost = tr(Diagonal(L_var))
		constraint = [ Diagonal(L_var) >= Ai * invH * Ai' ]
		problem = minimize(cost, constraint)
		solve!(problem, ECOS.Optimizer; silent_solver = true)
		Ld = Diagonal(vec(L_var.value))
		Linv = inv(Ld)
	else # Proposed method
		# Very important: makes a huge difference
		Ld = Ai * invH * Ai'
		Ld = Diagonal(Ld)
		Linv = inv(Ld)
	end

	theta = thetap = 1
	lambda = lambdap = zero(bi)
	z = nothing

	t2 = time_ns()
	precomp = (t2-t1)/1.0e6

	iters = max_iters
	for k in 1:max_iters
		beta = theta * (1 - thetap) / thetap
		omega = lambda + beta * (lambda - lambdap)

		z = Z0 + Z1 * omega

		lambdap = lambda
		lambda = max.(omega + Linv*(Ai * z - bi), 0) # 50 microsecs

		thetap = theta
		theta = 0.5*(sqrt(theta^4 + 4*theta^2) - theta^2)

		re = norm(Ae*z-be)
		ri = maximum(Ai*z-bi)
		if (re < epsilon_re && ri < epsilon_ri)
			iters = k
			break
		end

		#cost = z'*H*z
		#re = norm(Ae*z-be)
		#ri = maximum(Ai*z-bi)
		#println("iter $k: cost=$cost, re=$re ri=$ri")
	end

	t3 = time_ns()
	runtime = (t3-t1)/1.0e6

	re = norm(Ae*z-be)
	ri = maximum(Ai*z-bi)
	println("Solver GPAD $type: precomp=$precomp ms, iters=$iters, optval = $(z'*H*z/Nsteps), re=$re, ri=$ri")
	return z, runtime, precomp
end

# ----------------------
# Dual Projected Newton
# ----------------------

function solverDPN(H::Matrix{Float64}, Ae::Matrix{Float64}, be::Vector{Float64}, Ai::Matrix{Float64}, bi::Vector{Float64}; max_iters::Int64=2000)

	N = length(be)/4 -1

	Ae = sparse(Ae)
	Ai = sparse(Ai)

	# H is a diagonal matrix: use Diagonal not diagm:
	# https://web.eecs.umich.edu/~fessler/course/551/julia/tutor/03-diag.html
	invH = Diagonal(1 ./ diag(H))  # N=100: 17 microsecs
	Ha = Ae * invH * Ae';
	invHa = inv(Matrix(Ha)) # N=100: 2 ms

	Z0 = invH * Ae' * invHa * be
	Z1 = invH * Ae' * invHa * Ae * invH * Ai' - invH * Ai'

	#@infiltrate

	#H = Diagonal(H)

	#Hg = Z1' * H * Z1 + 2 * Ai * Z1
	#Hg = (Hg + Hg')/2
	#F = eigen(Hg)
	#F.values[ abs.(F.values) .< 1e-6 ] .= 0
	#pipo = F.vectors * Diagonal(F.values) * F.vectors'
	#@infiltrate

	# The above matrix has a terrible condition number
	Hg = Ai * invH * Ai'
	Hg = 0.5*Diagonal(Hg)
	invHg = inv(Hg)

	nh, _ = size(Hg)
	Hg_var = Variable(nh)
	cost = tr(Diagonal(Hg_var))
	constraint = [ Diagonal(Hg_var) >= Ai * invH * Ai' ]
	problem = minimize(cost, constraint)
	#solve!(problem, ECOS.Optimizer; silent_solver = true)
	solve!(problem, ECOS.Optimizer)
	Hg_new = Diagonal(vec(Hg_var.value))
	invHg = inv(Hg_new)

	lambda = zero(bi)
	z = nothing
	iters = max_iters
	for k in 1:max_iters
		z = Z0 + Z1 * lambda
		#lambda = max.(lambda + invLd*(Ai * z - bi), 0) # 50 microsecs
		lambda = max.(lambda + invHg*(Ai * z - bi), 0) # 50 microsecs

		re = norm(Ae*z-be)
		ri = maximum(Ai*z-bi)
		if (re < epsilon_re && ri < epsilon_ri)
			iters = k
			break
		end

		#cost = z'*H*z
		#re = norm(Ae*z-be)
		#ri = maximum(Ai*z-bi)
		#println("iter $k: cost=$cost, re=$re ri=$ri")
	end

	re = norm(Ae*z-be)
	ri = maximum(Ai*z-bi)
	println("Solver DPN    : optval = $(z'*H*z), re=$re, ri=$ri")
	return z
end
