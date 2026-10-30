// thrust_sel.cu — Selección de los mejores individuos con Thrust (CUDA 13.x).
//
// Compilar (genera libthrust_sel.so, que ga_cuda.py carga con ctypes):
//   nvcc -O3 -shared -Xcompiler -fPIC -o libthrust_sel.so thrust_sel.cu
//
// Todas las funciones reciben punteros a memoria de DISPOSITIVO (ya en GPU),
// así que no hay copias extra: el vector de fitness que escribió el kernel
// de Numba se reduce directamente en la GPU.
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

extern "C" {

// Índice del individuo con menor fitness (reducción: thrust::min_element,
// implementada internamente como un thrust::reduce sobre pares (valor, índice)).
int thrust_argmin(const double* d_fit, int n) {
    thrust::device_ptr<const double> p(d_fit);
    auto it = thrust::min_element(thrust::device, p, p + n);
    return static_cast<int>(it - p);
}

// Suma de fitness (ejemplo directo de thrust::reduce; útil para el promedio).
double thrust_sum(const double* d_fit, int n) {
    thrust::device_ptr<const double> p(d_fit);
    return thrust::reduce(thrust::device, p, p + n, 0.0);
}

// Top-k: ordena (fitness, índice) por fitness y copia los k primeros índices.
// d_keys se modifica (pasar una copia). d_idx: buffer de n enteros.
void thrust_topk(double* d_keys, int n, int k, int* d_idx, int* d_out) {
    thrust::device_ptr<double> keys(d_keys);
    thrust::device_ptr<int> idx(d_idx);
    thrust::device_ptr<int> out(d_out);
    thrust::sequence(thrust::device, idx, idx + n);
    thrust::sort_by_key(thrust::device, keys, keys + n, idx);
    thrust::copy(thrust::device, idx, idx + k, out);
}

}  // extern "C"
