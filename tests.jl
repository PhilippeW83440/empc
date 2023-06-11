include("afti16.jl")
include("solvers.jl")
include("gpad.jl")

#N = 10 # Number of steps (using zero-order hold every 0.05s)

Ns = collect(10:10:120)

ntests = length(Ns)

hz_ecos = zeros(ntests)
hz_scs = zeros(ntests)
hz_gpd = zeros(ntests)
hz_gpadB = zeros(ntests)
hz_gpadG = zeros(ntests)
hz_gpadP = zeros(ntests)

optvals = zeros(ntests)
optval_gpd = zeros(ntests)
optval_gpadB = zeros(ntests)
optval_gpadG = zeros(ntests)
optval_gpadP = zeros(ntests)

precomp_gpd = zeros(ntests)
precomp_gpadB = zeros(ntests)
precomp_gpadG = zeros(ntests)
precomp_gpadP = zeros(ntests)

for test in 1:ntests
	N = Ns[test]
	println("===========> TEST $test with N=$N")
	x_init = [0.0; 0; 0; 0] # Initial State
	x_ref = get_x_ref(N) # Reference trajectory to track

	@time H, Ae, be, Ai, bi = getAfti16Model(N, x_init, x_ref)
	
	@time x_opt, u_opt, optval = solveAfti16Model(N, x_init, x_ref)
	optvals[test] = optval
	getPlots(x_opt, u_opt, x_ref, "ref")
	
	println()
	
	best_runtime_ecos = Inf
	best_runtime_scs = Inf
	best_runtime_gpd = Inf
	best_runtime_gpadB = Inf
	best_runtime_gpadG = Inf
	best_runtime_gpadP = Inf
	# we iterate 5 times to get the best result (independant of PC alear/reschedulings ...)
	for i in 1:5
		@time z_opt, runtime = solverStandard(H, Ae, be, Ai, bi, type="ECOS")
		if runtime < best_runtime_ecos
			best_runtime_ecos = runtime
			hz_ecos[test] = 1000/runtime
			x_hat, u_hat = get_x_u_from_z(z_opt, x_ref)
			getPlots(x_hat, u_hat, x_ref, "ecos")
		end
		
		@time z_opt, runtime = solverStandard(H, Ae, be, Ai, bi, type="SCS")
		if runtime < best_runtime_scs
			best_runtime_scs = runtime
			hz_scs[test] = 1000/runtime
			x_hat, u_hat = get_x_u_from_z(z_opt, x_ref)
			getPlots(x_hat, u_hat, x_ref, "scs")
		end
		
		@time z, runtime, precomp = solverGPD(H, Ae, be, Ai, bi)
		if runtime < best_runtime_gpd
			best_runtime_gpd = runtime
			hz_gpd[test] = 1000/runtime
			optval_gpd[test] = z'*H*z
			precomp_gpd[test] = precomp
			x, u = get_x_u_from_z(z, x_ref)
			getPlots(x, u, x_ref, "gpd")
		end
		
		@time z, runtime, precomp = solverGPAD(H, Ae, be, Ai, bi; type="Bemporad")
		if runtime < best_runtime_gpadB
			best_runtime_gpadB = runtime
			hz_gpadB[test] = 1000/runtime
			optval_gpadB[test] = z'*H*z
			precomp_gpadB[test] = precomp
			x, u = get_x_u_from_z(z, x_ref)
			getPlots(x, u, x_ref, "gpadBemporad")
		end
		
		@time z, runtime, precomp = solverGPAD(H, Ae, be, Ai, bi; type="Giselsson")
		if runtime < best_runtime_gpadG
			best_runtime_gpadG = runtime
			hz_gpadG[test] = 1000/runtime
			optval_gpadG[test] = z'*H*z
			precomp_gpadG[test] = precomp
			x, u = get_x_u_from_z(z, x_ref)
			getPlots(x, u, x_ref, "gpadGiselsson")
		end
		
		@time z, runtime, precomp = solverGPAD(H, Ae, be, Ai, bi; type="Proposed")
		if runtime < best_runtime_gpadP
			best_runtime_gpadP = runtime
			hz_gpadP[test] = 1000/runtime
			optval_gpadP[test] = z'*H*z
			precomp_gpadP[test] = precomp
			x, u = get_x_u_from_z(z, x_ref)
			getPlots(x, u, x_ref, "gpadProposed")
		end
	end
end

#plot(yaxis=:log, title="runtime", legend=:topleft)
#plot(title="runtime", legend=:topleft)
plot(yaxis=:log, title="Speed", background_color_legend=nothing, legend=:topright)
plot!(Ns, hz_ecos, label="ECOS", linestyle=:dot)
plot!(Ns, hz_scs, label="SCS", linestyle=:dot)
plot!(Ns, hz_gpd,   label="GPD  Ld")
plot!(Ns, hz_gpadB, label="GPAD Bemporad")
plot!(Ns, hz_gpadG, label="GPAD Giselsson")
plot!(Ns, hz_gpadP, label="GPAD Proposed")
xlabel!(L"Horizon $N$")
ylabel!("Hz")
savefig("plots/runtime.png")

#@infiltrate
plot(title="Optimal values", background_color_legend=nothing, legend=:topleft)
plot!(Ns, optvals,   label="ECOS")
plot!(Ns, optval_gpd,   label="GPD  Ld")
plot!(Ns, optval_gpadB, label="GPAD Bemporad")
plot!(Ns, optval_gpadG, label="GPAD Giselsson", linestyle=:dot)
plot!(Ns, optval_gpadP, label="GPAD Proposed")
xlabel!(L"Horizon $N$")
ylabel!(L"f^{*}")
savefig("plots/accuracy.png")

plot(yaxis=:log, title="Precomputation times", background_color_legend=nothing, legend=:topleft)
plot!(Ns, precomp_gpd,   label="GDP  Ld")
plot!(Ns, precomp_gpadB, label="GPAD Bemporad")
plot!(Ns, precomp_gpadG, label="GPAD Giselsson", linestyle=:dot)
plot!(Ns, precomp_gpadP, label="GPAD Proposed")
xlabel!(L"Horizon $N$")
ylabel!(L"Time $ms$")
savefig("plots/precomp.png")

#@time z_dpn = solverDPN(H, Ae, be, Ai, bi)
#@time x_dpn, u_dpn = get_x_u_from_z(z_dpn, x_ref)
#getPlots(x_dpn, u_dpn, x_ref, "dpn")
