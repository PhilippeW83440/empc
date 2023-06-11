using LinearAlgebra

# Do not use BlockDiagonal: too slow
#using BlockDiagonals

using Convex, SCS
using ECOS

using Plots
using LaTeXStrings

using Infiltrator

deg_to_rad(deg) = deg/180 * pi
rad_to_deg(rad) = rad/pi  * 180

# Challenges
# 1) Open-loop unstable poles
# 2) Saturated actuators

# ------------------------
# AFTI-16 Model
# ------------------------

# x: state variables
n = 4
# State
# x1: u(t) forward velocity (m/sec)
# x2: alpha(t) angle of attack (deg)
# x3: q(t) pitch rate (deg/sec)
# x4: theta(t) pitch angle (deg)

# u: control variables
m = 2
# Controls
# u1: delta_e(t) elevator angle (deg) s.t. |u1| <= 25°
# u2: delta_f(t) flaperon angle (deg) s.T. |u2| <= 25°

# dx = A*x + B*u
A = [ 0.999 -3.008 -0.113 -1.608;
      0.0    0.986  0.048  0.0;
	  0.0    2.083  1.009  0.0;
	  0.0    0.053  0.05   1.0]

B = [ -0.080 -0.635;
      -0.029 -0.014;
	  -0.868 -0.092;
	  -0.022 -0.002]

# LTI model: approx of the aircraft longitudinal dynamics
#            at 3000 ft altitude and .6 Mach velocity
# Not sampled at 0.05 sec: not usable
#A = [ -0.0151 -60.5651    0    -32.174;
#      -0.0001 -1.3411   0.9929    0;
#	  0.00018 43.2541  -0.86939   0;
#	     0      0         1      0]      
#
#B = [ -2.516  -13.136;
#      -0.1689 -0.2514;
#	 -17.251  -1.5766;
#	    0       0]

# Sampled one at 0.05 sec: usable
#A = [ 0.0093    -3.0083  -0.1131 -1.6081;
#      4.7030e-6  0.9862   0.0478  3.8501e-6;
#	  3.7028e-6  2.0833   1.0089 -4.3616e-6;
#	  1.3556e-7  0.0526   0.0498  1]
#
#B = [ -0.0804 -0.6347;
#      -0.0291 -0.0143;
#	  -0.8679 -0.0917;
#	  -0.0216 -0.0022]

# Output
# y1: alpha(t) angle of attack (deg)
# y2: theta(t) pitch angle (deg)
# y = C*x
C = [0.0 1 0 0;
       0 0 0 1]

# cost: sum(x'*Q*x + u'*R*u) + xN'*QN*XN   
Q = diagm([1e-4; 1e2; 1e-3; 1e2])
R = diagm([1e-2; 1e-2])
#QN = diagm([0; 1e2; 0; 1e2])
QN = diagm([1e-4; 1e2; 1e-3; 1e2]) # avoid 0 diagonal elements

# |ui| <= 25° for elevator and flaperon angles
#u_min = -[25.0; 25.0]
#u_max = [ 25.0; 25.0]
u_min = -[deg_to_rad(25); deg_to_rad(25)]
u_max =  [deg_to_rad(25); deg_to_rad(25)]
	
# -0.5° <= y1 (=x2) angle of attack <= 0.5°
# -100° <= y2 (=x4) pitch angle <= 100°
#y_min = -[0.5; 100]
#y_max =  [0.5; 100]
y_min = -[deg_to_rad(0.5); deg_to_rad(100)]
y_max =  [deg_to_rad(0.5); deg_to_rad(100)]


# ------------------------------------
# Control goal: 
# ------------------------------------
# Drive the pitching angle y2 from 0° to 10° and then back to 0°
# while the angle of attack statisfies  -0.5° <= y1 <= 0.5°
# The constraints on the angle of attack limits the rate on how fast
# the pitch angle can be changed.

# Construct the reference trajectory
function get_x_ref(N::Int64)
	x_ref = [[0; 0; 0; deg_to_rad(10)]]
	for i in 1:round(N/2)
		# request x2=0 (angle of attack) and x4=10 (pitch angle)
		#push!(x_ref, [0; 0; 0; 10])
		push!(x_ref, [0; 0; 0; deg_to_rad(10)])
	end
	for i in round(N/2)+1:N
		# request x2=0 (angle of attack) and x4=0 (pitch angle)
		push!(x_ref, [0; 0; 0; 0])
	end
	return x_ref
end

# input  : Horizon N
# outputs: H, Ae, be, Ai, bi
function getAfti16Model(N::Int64, x_init::Vector{Float64}, x_ref::Vector{Vector{Float64}})
	Im = Matrix(1.0*I, m,m)
	In = Matrix(1.0*I, n,n)
	
	# We define z=[x0 - x_ref0, u0, x1 - x_ref1, u1, ..., xN-1 - x_refN-1, uN-1, xN - x_refN]

	# -------------------------------------
	# Construct H s.t. cost = 0.5 z'*H*z
	# -------------------------------------
	#W = [Q zeros(n, m); zeros(m, n) R] 
	#blocks = [W]
	#for k in 1:N-1
	#	push!(blocks, W)
	#end
	#push!(blocks, QN)
	#H::Matrix{Float64} = BlockDiagonal(blocks)

	t1 = time_ns()
	W = [diag(Q) ; diag(R)]
	H = W
	for k in 1:N-1
		H = [H; W]
	end
		H = [H; diag(QN)]
	H = diagm(H)
	t2 = time_ns()
	println("Time to construct H: $((t2-t1)/1.0e6) ms")

	nz, nz = size(H)
	
	# -------------------------------------
	# Construct Ai and bi s.t. Ai*z <= bi
	# -------------------------------------
	t1 = time_ns()
	Ai = zeros((2*N+1)*m , nz)
	dx, dy = 0, 0 
	for k in 1:N
		Ai[begin+dx:begin+dx+m-1, begin+dy:begin+dy+n-1] = C
		dx += m
		dy += n
		Ai[begin+dx:begin+dx+m-1, begin+dy:begin+dy+m-1] = Im
		dx += m
		dy += m
	end
	Ai[begin+dx:begin+dx+m-1, begin+dy:begin+dy+n-1] = C
	Ai = [Ai; -Ai]
	t2 = time_ns()
	println("Time to construct Ai: $((t2-t1)/1.0e6) ms")

	#t1 = time_ns()
	#blocks = [C,Im]
	#for k in 1:N-1
	#	push!(blocks, C)
	#	push!(blocks, Im)
	#end
	#push!(blocks, C)
	#Ai = BlockDiagonal(blocks)
	#Ai = [Ai; -Ai]
	#t2 = time_ns()
	#println("Time to construct Ai: $((t2-t1)/1.0e6) ms")

	t1 = time_ns()
	bi = [y_max - C*x_ref[begin]; u_max]
	for k in 1:N-1
		bi = [bi; y_max - C*x_ref[begin + k]; u_max]
	end
	bi = [bi; y_max - C*x_ref[begin + N] ]
	
	for k in 0:N-1
		bi = [bi; -y_min + C*x_ref[begin + k]; -u_min]
	end
	bi = [bi; -y_min + C*x_ref[begin + N] ]
	t2 = time_ns()
	println("Time to construct bi: $((t2-t1)/1.0e6) ms")

	# -------------------------------------
	# Construct Ae and be s.t. Ae*z = be
	# -------------------------------------
	t1 = time_ns()
	rows = []
	
	row_0 = []
	push!(row_0, In)
	for i in 2:2*N+1
		Zero = (i%2 == 1) ? zeros(n,n) : zeros(n,m)
		push!(row_0, Zero)
	end
	push!(rows, hcat(row_0...))
	
	for k in 1:N
		row_k = []
		for i in 1:2*(k-1)
			Zero = (i%2 == 1) ? zeros(n,n) : zeros(n,m)
			push!(row_k, Zero)
		end
		push!(row_k, -A)
		push!(row_k, -B)
		push!(row_k, In)
		for i in 2*(k+1):2*N+1
			Zero = (i%2 == 1) ? zeros(n,n) : zeros(n,m)
			push!(row_k, Zero)
		end
		push!(rows, hcat(row_k...))
	end
	Ae = vcat(rows...)
	
	be = x_init - x_ref[begin + 0]
	for k in 1:N
		be = [be; -x_ref[begin + k] + A*x_ref[begin + k - 1]]
	end
	t2 = time_ns()
	println("Time to construct Ae and be: $((t2-t1)/1.0e6) ms")
	
	# ---------------------------------------
	# Some basic checks
	# ---------------------------------------
	println("N=$N nz=$nz")
	z = rand(nz)
	println("norm(Ae*z-be) = ", norm(Ae*z-be))
	println("norm(Ai*z-bi) = ", norm(Ai*z-bi))
	println("z'*H*z =  ", z'*H*z)
	println("size(Ae) = $(size(Ae)), size(be) = $(size(be))")
	println("size(Ai) = $(size(Ai)), size(bi) = $(size(bi))")
	println("size(H) = $(size(H))")

	n_vars = nz
	m_constraints = size(Ae)[1] + size(Ai)[1]
	println("Problem complexity: (n vars, m constraints) = ($n_vars, $m_constraints)")

	return H, Ae, be, Ai, bi
end


function getPlots(x, u, x_ref, name)
	m, N = size(u)
	n, Nplus1 = size(x)
	@assert N + 1 == Nplus1 

	u1s = rad_to_deg.(u[1, :]) # elevator angles
	u2s = rad_to_deg.(u[2, :]) # flaperon angles
	y1s = rad_to_deg.((C*x)[1, :]) # angles of attack
	y2s = rad_to_deg.((C*x)[2, :]) # pitchs angle

	y1s_ref, y2s_ref = [], []
	for k in 0:N
		yk_ref = rad_to_deg.(C*x_ref[begin + k])
		push!(y1s_ref, yk_ref[1])
		push!(y2s_ref, yk_ref[2])
	end

	plot(title=L"output vector: $y$")
	plot!(y1s, label=L"$y_1$ angle of attack", color=:blue)
	plot!(y2s, label=L"$y_2$ pitch angle", color=:red)
	plot!(y1s_ref, label=L"$y_{1}^{ref}$ angle of attack", linestyle=:dot, color=:blue)
	plot!(y2s_ref, label=L"$y_{2}^{ref}$ pitch angle", linestyle=:dot, color=:red)
	xlabel!(L"steps")
	ylabel!(L"angle")
	hline!([rad_to_deg(y_min[1]), rad_to_deg(y_max[1])], linestyle=:dashdot, color=:black, label=L"$y_1$ constraint")
	savefig("plots/output_" * name * ".png")

	plot(title=L"input vector: $u$")
	plot!(u1s, label=L"$u_1$ elevator angle", color=:blue)
	plot!(u2s, label=L"$u_2$ flaperon angle", color=:red)
	xlabel!(L"steps")
	ylabel!(L"angle")
	hline!([rad_to_deg(u_min[1]), rad_to_deg(u_max[1])], linestyle=:dashdot, color=:black, label=L"$u_1$ constraint")
	hline!([rad_to_deg(u_min[2]), rad_to_deg(u_max[2])], linestyle=:dashdot, color=:black, label=L"$u_2$ constraint")
	savefig("plots/input_" * name * ".png")

end

function get_x_u_from_z(z::Vector{Float64}, x_ref::Vector{Vector{Float64}})
	nz = length(z)
	N = Int64((nz-n) / (n+m))
	#@infiltrate
	x = zeros(n, N+1)
	u = zeros(m, N)

	idx = 1
	for k in 0:N-1
		x[:, begin + k] = z[idx:idx+n-1] + x_ref[begin + k]
		u[:, begin + k] = z[idx+n:idx+n+m-1]
		idx += (n+m)
	end
	x[:, begin + N] = z[end - n + 1: end] + x_ref[begin + N]
	return x, u
end

function solveAfti16Model(N::Int64, x_init::Vector{Float64}, x_ref::Vector{Vector{Float64}})
	x = Variable(n, N+1)
	u = Variable(m, N)

	cost = 0
	constraints = [ x[:, begin] == x_init ]
	for k in 0:N-1
		xk = x[:, begin + k]
		xk_ref = x_ref[begin + k]
		uk = u[:, begin + k]
		cost += quadform(xk - xk_ref, Q) + quadform(uk, R)

		constraints += [ y_min <= C*xk, C*xk <= y_max ]
		constraints += [ u_min <= uk, uk <= u_max ]
		constraints += [ x[:, begin + k+1] == A*xk + B*uk ]
	end
	xN = x[:, begin + N]
	xN_ref = x_ref[begin + N]
	cost += quadform(xN - xN_ref, QN)

	problem = minimize(cost, constraints)
	#solve!(problem, SCS.Optimizer; silent_solver = true)
	solve!(problem, ECOS.Optimizer; silent_solver = true)

	println("Solver Afti16Model: status = $(problem.status), optval = $(problem.optval/N)")
	return x.value, u.value, problem.optval
end
