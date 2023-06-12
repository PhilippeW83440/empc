import numpy as np
import matplotlib.pyplot as plt
import time
#import scipy
from scipy.linalg import cho_factor, cho_solve

import cvxpy as cp

np.random.seed(10)

def f(x):
	assert (x > 0).all()
	return c.T @ x - sum(np.log(x))

def Gradf(x):
	assert (x > 0).all()
	return c - 1/x

def Hf(x):
	assert (x > 0).all()
	return np.diag(1/(x**2))

def Hf_inv(x):
	assert (x > 0).all()
	return np.diag(x**2)

# Problem Setup
# n: num vars, m: num constraints
def random_problem(n, m):
	A = np.random.rand(m, n)
	p = np.random.rand(n)
	b = A @ p
	rankA = np.linalg.matrix_rank(A)
	print("Rank(A)={}, (m,n)=({},{})".format(rankA, m, n))
	assert rankA == min(m, n)
	c = np.random.rand(n)
	#x0 = np.ones(n)
	x0 = np.random.rand(n)
	return A, b, c, x0


class SolverPrimalDual(object):
	# Constructor
	def __init__(self, f, Gradf, Hf, Hf_inv, A, b, x0, alpha=0.01, beta=0.3, max_iters=50, epsilon=1e-6):
		self.setup(f, Gradf, Hf, Hf_inv, A, b, x0, alpha, beta, max_iters, epsilon)

	def setup(self, f, Gradf, Hf, Hf_inv, A, b, x0, alpha=0.01, beta=0.3, max_iters=50, epsilon=1e-6):
		self.f = f
		self.Gradf = Gradf
		self.Hf = Hf
		self.Hf_inv = Hf_inv

		m, n = A.shape
		self.m = m
		self.n = n
		assert len(x0) == n
		assert len(b) == m
		assert isinstance(f(x0), float)
		assert len(Gradf(x0)) == n
		assert Hf(x0).shape == (n, n)
		assert Hf_inv(x0).shape == (n, n)

		self.A = A
		self.b = b
		self.x0 = x0
		self.nu0 = np.zeros(m) # start will all zeros nu

		self.alpha = alpha
		self.beta = beta
		self.max_iters = max_iters
		self.epsilon = epsilon

	def residual(self, x, nu):
		return self.Gradf(x) + self.A.T @ nu, self.A @ x - self.b

	def norm_residual(self, x, nu):
		g, h = self.residual(x, nu)
		return np.linalg.norm(np.hstack((g, h)), ord=2)

	def solve_KKT(self, x, nu):
		g, h = self.residual(x, nu)
		Hf = self.Hf(x)
		M = np.block([[Hf, self.A.T],
		              [self.A, np.zeros((self.m, self.m))]])
		dvar = np.linalg.solve(M, -np.hstack((g, h)))
		dx = dvar[:self.n]
		dnu = dvar[self.n:]
		return dx, dnu

	# Compute primal and dual Newton steps by Block Elimination
	def solve_KKT_by_block_elimination(self, x, nu):
		g, h = self.residual(x, nu)
		Hf = self.Hf(x)
		Hf_inv = self.Hf_inv(x)

		S = Hf_inv @ self.A.T
		g_temp = Hf_inv @ g
		S = self.A @ S

		#S = nearestPD(S)
		#L = cho_factor(S)
		#dnu = cho_solve(L, h - self.A @ g_temp)

		dnu = np.linalg.solve(S, h - self.A @ g_temp)
		dx = -Hf_inv @ (g + self.A.T @ dnu)

		return dx, dnu

	def solve(self):
		x = self.x0
		nu = self.nu0

		residuals = []
		residuals.append(self.norm_residual(x, nu))
		for it in range(1, self.max_iters):
			dx, dnu = self.solve_KKT_by_block_elimination(x, nu)
			#dx, dnu = self.solve_KKT(x, nu)
			t = 1
			while np.min(x + t * dx) <= 0:
				t *= self.beta
			while self.norm_residual(x+t*dx, nu+t*dnu) > (1 - self.alpha*t) * self.norm_residual(x, nu):
				t *= self.beta
			x = x + t*dx
			nu = nu + t*dnu
			norm_residual = self.norm_residual(x, nu)
			residuals.append(norm_residual)
			norm_constraints = np.linalg.norm(self.A @ x - self.b, ord=2)
			print("it {}: norm_residual = {}, norm_constraints = {}".format(it, norm_residual, norm_constraints))
			if norm_residual <= self.epsilon and norm_constraints <= self.epsilon:
				break

		return x, residuals


plt.figure()
plt.grid('on')
plt.xlabel('iterations');
plt.ylabel('residual')
plt.title('Infeasible start Newton method')

n=100
m=500

for i in range(5):
	A, b, c, x0 = random_problem(n, m)

	# 0 < alpha < 0.5 but typically in [0.01, 0.3]
	# 0 < beta  < 1   but typically in [0.1 , 0.8]
	solver = SolverPrimalDual(f, Gradf, Hf, Hf_inv, A, b, x0, alpha=0.01, beta=0.9, max_iters=500, epsilon=1e-6)

	start = time.time()
	x_opt, residuals = solver.solve()
	end = time.time()

	print("runtime: {}\n".format(end -start))
	#print("x_opt = {}".format(x_opt))
	print("f_opt = {}\n\n".format(f(x_opt)))

	plt.plot(residuals)

	# --- Check results against CVXPY ---
	start = time.time()
	x = cp.Variable(n)
	objective = c.T @ x - cp.sum(cp.log(x))
	constraints = [A @ x == b]
	prob = cp.Problem(cp.Minimize(objective), constraints)
	prob.solve()
	end = time.time()
	print("runtime cvxpy: {}n".format(end -start))
	print("status {}".format(prob.status))
	print("value {}\n".format(prob.value))

#plt.legend()
plt.savefig('residuals.png')
#plt.axis('equal')
plt.show()
