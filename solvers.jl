using Convex, SCS
using ECOS
using OSQP


function solverStandard(H::Matrix{Float64}, Ae::Matrix{Float64}, be::Vector{Float64}, Ai::Matrix{Float64}, bi::Vector{Float64}; type::String="ECOS")

	Nsteps = length(be)/4 -1

	t1 = time_ns()
	nz, nz = size(H)
	z = Variable(nz)

	cost = quadform(z, H)
	constraints = [ Ae*z == be ]
	constraints += [ Ai*z <= bi ]

	problem = minimize(cost, constraints)
	#solve!(problem, SCS.Optimizer; silent_solver = true)
	if (type == "ECOS")
		solve!(problem, ECOS.Optimizer; silent_solver = true)
	else
		solve!(problem, SCS.Optimizer; silent_solver = true)
	end
	#solve!(problem, ECOS.Op#timizer; silent_solver = false)

	z = vec(z.value)
	re = norm(Ae*z-be)
	ri = maximum(Ai*z-bi)
	println("Solver $type : status = $(problem.status), optval = $(problem.optval/Nsteps), re=$re, ri=$ri")

	t3 = time_ns()
	runtime = (t3-t1)/1.0e6

	return z, runtime
end
