import pdb
import numpy as np
import nose
import cudamat as cm
import learn as cl

def setup():
    cm.cublas_init()

def teardown():
    cm.cublas_shutdown()

def test_mult_by_sigmoid_deriv():
    m = 256
    n = 128
    c_targets = np.array(np.random.randn(m, n)*10, dtype=np.float32, order='F')
    c_acts = np.array(np.random.rand(m, n), dtype=np.float32, order='F')

    g_targets = cm.CUDAMatrix(c_targets)
    g_acts = cm.CUDAMatrix(c_acts)

    c_targets = c_targets * c_acts * (1. - c_acts)
    cl.mult_by_sigmoid_deriv(g_targets, g_acts)

    assert np.max(np.abs(c_acts - g_acts.asarray())) < 10**-2, "Error in cudamat.learn.mult_by_sigmoid_deriv exceeded threshold"

def test_calculate_l1_penalty():
    m = 256
    n = 128
    p = 1
    c_data = np.array(np.random.randn(m, n)*2, dtype=np.float32, order='F')
    c_results = np.array(np.random.rand(m, n), dtype=np.float32, order='F')

    g_data = cm.CUDAMatrix(c_data)
    g_results = cm.CUDAMatrix(c_results)

    d = p * np.sign(c_data)
    c_results = np.where(np.abs(d) > np.abs(c_data), c_data, d)
    cl.calculate_l1_penalty(g_data, p, g_results)
    
    assert np.max(np.abs(c_results - g_results.asarray())) < 10**-2, "Error in cudamat.learn.calculate_l1_penalty exceeded threshold"


if __name__ == '__main__':
    nose.run()
