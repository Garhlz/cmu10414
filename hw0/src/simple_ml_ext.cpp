#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
namespace py = pybind11;


void matrix_product(const float* a, const float* b, float *c, size_t m, size_t n, size_t k, bool transpose_A = false) {
    /*
    X: batch * n 
    theta: n * k
    Z: batch * k
    */
    if(!transpose_A) {
        for(size_t row = 0; row < m; row++) {
            for(size_t col = 0;col < k;col++) {
                float sum = 0.0;
                for(size_t i = 0;i < n; i++){
                    sum += a[row * n + i] * b[i * k + col];
                }
                c[row * k + col] = sum;
            }
        }
    } else { // 将矩阵a视为转置
        for(size_t row = 0; row < n; row++) {
            for(size_t col = 0;col < k;col++) {
                float sum = 0.0;
                for(size_t i = 0;i < m; i++){
                    sum += a[i * n + row] * b[i * k + col];
                }
                c[row * k + col] = sum;
            }
        }
    }

}

void softmax(float* Z, size_t batch_size, size_t k) {
    for(size_t row = 0;row < batch_size; row++) {
        float max = -std::numeric_limits<float>::infinity();
        for(size_t col = 0;col < k;col++) {
            max = std::max(max, Z[row * k + col]);
        }

        float sum = 0;
        for(size_t col = 0;col < k;col++) {
            Z[row * k + col] -= max;
            Z[row * k + col] = std::exp(Z[row * k + col]);
            sum += Z[row * k + col];
        }
        for(size_t col = 0;col < k;col++)  {
            Z[row * k + col] /= sum;
        } 
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float *Z = new float[batch * k]();
    float *delta = new float[n * k]();
    for(size_t i = 0;i < m;i += batch) {
        size_t batch_size = std::min(m - i, batch);
        const float *X_b = X + i * n;   
        const unsigned char *y_b = y + i;
        
        matrix_product(X_b, theta, Z, batch_size, n, k);

        softmax(Z, batch_size, k);

        // dl / dZ = P - Y_oneshot
        for(size_t row = 0;row < batch_size;row++) {
            Z[row * k + y_b[row]] -= 1;
        }

        
        // X_b: n * batch
        // Z: batch * k
        // delta: n * k
        matrix_product(X_b, Z, delta, batch_size, n, k, true);

        for(size_t row = 0; row < n; row++) {
            for(size_t col = 0;col < k;col++) {
                theta[row * k + col] -= delta[row * k + col] / batch_size * lr;
            }
        }
        
    }   
    delete[] delta;
    delete[] Z;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
