# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow
from libc.stdlib cimport malloc, free
from cython.parallel import prange

cdef float float32_eps = np.finfo(np.float32).eps

@cython.boundscheck(False)
cdef void __mls_detail(float[:, :] p,
							  float[:, :] q,
							  float vx, float vy,
							  float *fx, float *fy,
							  float alpha,
							  float *w) nogil:
	cdef size_t num_p = p.shape[0]
	cdef size_t j
	cdef float w_sum = 0.0
	cdef int near_p = 0
	cdef float dist

	# Compute weights for each reference point
	for j in range(num_p):
		dist = sqrt((vx - p[j, 0]) * (vx - p[j, 0]) +
					(vy - p[j, 1]) * (vy - p[j, 1]))
		if dist <= float32_eps:
			# If the test point is very close to a reference point,
			# directly use the corresponding q coordinates.
			fx[0] = q[j, 0]
			fy[0] = q[j, 1]
			near_p = 1
			break
		w[j] = pow(dist, -alpha)
		w_sum += w[j]
	if near_p:
		return

	# Compute weighted centroid for p (p_star)
	cdef float[2] p_star
	p_star[0] = 0.0
	p_star[1] = 0.0
	for j in range(num_p):
		p_star[0] += w[j] * p[j, 0]
		p_star[1] += w[j] * p[j, 1]
	p_star[0] /= w_sum
	p_star[1] /= w_sum

	# Compute weighted centroid for q (q_star)
	cdef float[2] q_star
	q_star[0] = 0.0
	q_star[1] = 0.0
	for j in range(num_p):
		q_star[0] += w[j] * q[j, 0]
		q_star[1] += w[j] * q[j, 1]
	q_star[0] /= w_sum
	q_star[1] /= w_sum

	# Determine the transformation using the centered points
	cdef float mus = 0.0
	cdef float[2] f_tmp
	f_tmp[0] = 0.0
	f_tmp[1] = 0.0
	cdef float[2] p_hat, q_hat, v_hat
	for j in range(num_p):
		p_hat[0] = p[j, 0] - p_star[0]
		p_hat[1] = p[j, 1] - p_star[1]
		q_hat[0] = q[j, 0] - q_star[0]
		q_hat[1] = q[j, 1] - q_star[1]

		f_tmp[0] += (q_hat[0] * p_hat[0] + q_hat[1] * p_hat[1]) * w[j]
		f_tmp[1] += (q_hat[0] * p_hat[1] - q_hat[1] * p_hat[0]) * w[j]
		mus += w[j] * (p_hat[0] * p_hat[0] + p_hat[1] * p_hat[1])
	v_hat[0] = vx - p_star[0]
	v_hat[1] = vy - p_star[1]

	fx[0] = (f_tmp[0] * v_hat[0] + f_tmp[1] * v_hat[1]) / mus + q_star[0]
	fy[0] = (f_tmp[0] * v_hat[1] - f_tmp[1] * v_hat[0]) / mus + q_star[1]

@cython.boundscheck(False)
cdef void __mls_parallel_loop(size_t i,
									   float[:, :] p,
									   float[:, :] q,
									   float[:, :] v,
									   float[:, :] f,
									   float alpha) nogil:
	cdef size_t num_p = p.shape[0]
	# Allocate a temporary weight buffer for each iteration to ensure thread safety
	cdef float *w = <float *>malloc(num_p * sizeof(float))
	__mls_detail(p, q, v[i, 0], v[i, 1],
						&f[i, 0], &f[i, 1],
						alpha, w)
	free(w)

@cython.boundscheck(False)
cdef void __mls(float[:, :] p,
					   float[:, :] q,
					   float[:, :] v,
					   float[:, :] f,
					   float alpha=1.) nogil:
	cdef size_t num_v = v.shape[0]
	cdef size_t i
	for i in prange(num_v, schedule='static'):
		__mls_parallel_loop(i, p, q, v, f, alpha)

cpdef mls(float[:, :] p,
				   float[:, :] q,
				   float[:, :] v,
				   float alpha=1.):
	"""
	Perform weighted mls interpolation on the points in 'v' based on the
	reference points 'p' and their corresponding target points 'q'.

	Parameters:
		p (ndarray): Reference points with shape (n, 2).
		q (ndarray): Target points corresponding to p, with shape (n, 2).
		v (ndarray): Points where the transformation is computed, with shape (m, 2).
		alpha (float): Exponent used in the weight computation (default: 1.0).

	Returns:
		ndarray: Transformed points with shape (m, 2).
	
	See https://www.nealen.net/projects/mls/asapmls.pdf
	"""
	assert p.shape[1] == 2
	assert q.shape[1] == 2
	assert p.shape[0] == q.shape[0]
	assert v.shape[1] == 2
	cdef np.ndarray[float, ndim=2] f = np.empty((v.shape[0], 2), dtype=np.float32)
	__mls(p, q, v, f, alpha)
	return f
