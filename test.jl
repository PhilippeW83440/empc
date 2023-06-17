include("afti16.jl")
include("solvers.jl")
include("gpad.jl")

#N = 10 # Number of steps (using zero-order hold every 0.05s)
N = 100
x_init = [0.0; 0; 0; 0] # Initial State
x_ref = get_x_ref(N) # Reference trajectory to track

@time H, Ae, be, Ai, bi = getAfti16Model(N, x_init, x_ref)
	
@time x_opt, u_opt, optval = solveAfti16Model(N, x_init, x_ref)
getPlots(x_opt, u_opt, x_ref, "ref")
	
println()
	
@time z_opt, runtime = solverStandard(H, Ae, be, Ai, bi, type="ECOS")
x_hat, u_hat = get_x_u_from_z(z_opt, x_ref)
getPlots(x_hat, u_hat, x_ref, "ecos")

@time z_opt, runtime = solverStandard(H, Ae, be, Ai, bi, type="SCS")
x_hat, u_hat = get_x_u_from_z(z_opt, x_ref)
getPlots(x_hat, u_hat, x_ref, "scs")

@time z, runtime, precomp = solverGPD(H, Ae, be, Ai, bi)
x, u = get_x_u_from_z(z, x_ref)
getPlots(x, u, x_ref, "gpd")

@time z, runtime, precomp = solverGPAD(H, Ae, be, Ai, bi; type="Bemporad")
x, u = get_x_u_from_z(z, x_ref)
getPlots(x, u, x_ref, "gpadBemporad")

@time z, runtime, precomp = solverGPAD(H, Ae, be, Ai, bi; type="Giselsson")
x, u = get_x_u_from_z(z, x_ref)
getPlots(x, u, x_ref, "gpadGiselsson")

@time z, runtime, precomp = solverGPAD(H, Ae, be, Ai, bi; type="Proposed")
x, u = get_x_u_from_z(z, x_ref)
getPlots(x, u, x_ref, "gpadProposed")

#@time z_dpn = solverDPN(H, Ae, be, Ai, bi)
#@time x_dpn, u_dpn = get_x_u_from_z(z_dpn, x_ref)
#getPlots(x_dpn, u_dpn, x_ref, "dpn")
