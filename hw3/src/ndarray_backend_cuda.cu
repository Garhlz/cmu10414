#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * 计算一维 CUDA kernel 调用所需的线程块和网格维度。
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// 填充操作
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact 和 setitem 操作
////////////////////////////////////////////////////////////////////////////////

// 根据步长，将连续布局中的索引 i 转换为实际内存位置的辅助函数



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * 执行 compact 操作的 CUDA kernel。
   * 将非紧凑输入 a 中的单个元素映射到紧凑输出 out 中的对应位置 gid。
   *
   * 参数：
   *   a：指向输入数组的 CUDA 指针。
   *   out：指向输出数组的 CUDA 指针。
   *   size：out 的元素数量。
   *   shape：a 和 out 的形状，以 CudaVec 类型传入 CUDA kernel。
   *   strides：a 的步长。
   *   offset：a 的偏移量。
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * 将数组整理为内存中的紧凑布局。
   * 与 C++ 版本不同，这里主要负责调用相应的 CUDA kernel。
   * 为演示调用方式，已提供本函数的完整实现和 CompactKernel() 的声明。
   * 后续函数则需要自行定义 kernel 来完成实际计算。
   *
   * 参数：
   *   a：输入的非紧凑数组。
   *   out：用于写入结果的紧凑数组。
   *   shape：a 和 out 各维度的大小。
   *   strides：a 的步长；out 使用紧凑步长。
   *   offset：a 的偏移量；紧凑数组 out 的偏移量为 0。
   */

  // 这里无需添加代码
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * 使用 CUDA 设置非紧凑数组中的元素。
   * 通常需要仿照前面的示例实现 EwiseSetitemKernel()，以执行实际计算。
   *
   * 参数：
   *   a：紧凑的输入数组，其元素将写入 out。
   *   out：待写入的非紧凑数组。
   *   shape：a 和 out 各维度的大小。
   *   strides：out 的步长；a 使用紧凑步长。
   *   offset：out 的偏移量；紧凑数组 a 的偏移量为 0。
   */
  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}



void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * 将标量写入非紧凑数组中的元素。
   *
   * 参数：
   *   size：要写入 out 的元素数量。out 是非紧凑子数组，因此该值不一定
   *         等于 out.size，而是等于 shape 各项的乘积；单独传入此值便于使用。
   *   val：要写入的标量值。
   *   out：待写入的非紧凑数组。
   *   shape：out 各维度的大小。
   *   strides：out 的步长。
   *   offset：out 的偏移量。
   */
  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// 逐元素和标量操作
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // 计算当前线程的全局索引。
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * 将两个 CUDA 数组相加。
   * 参数：
   *   a：待相加的输入数组 a。
   *   b：待相加的输入数组 b。
   *   out：保存 a + b 结果的输出数组。
   */
  CudaDims dim = CudaOneDim(out->size);

  // kernel 在 dim.grid 个线程块上执行，每个线程块包含 dim.block 个线程。
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // 计算当前线程的全局索引。
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * 将标量值加到 CUDA 数组的每个元素上。
   * 参数：
   *   a：输入数组 a。
   *   val：待相加的标量值。
   *   out：保存 a + val 结果的输出数组。
   */
  CudaDims dim = CudaOneDim(out->size);

  // 启动 ScalarAddKernel，将标量 val 加到数组 a 的每个元素上，
  // 并将结果保存到数组 out。
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * 参考上面的实现，为以下函数编写对应的逐元素和标量操作。
 * 可参考 NumPy 后端了解各操作的预期行为。
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * 分别朴素实现会产生大量重复代码。可以使用宏或模板精简实现，
 * 但这不是必需的；无论采用何种方式，函数签名都必须符合上述要求。
 */


////////////////////////////////////////////////////////////////////////////////
// 逐元素和标量操作
////////////////////////////////////////////////////////////////////////////////


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * 将两个紧凑矩阵相乘，结果写入同样紧凑的输出矩阵。
   * 实现方法可参考 GPU 线性代数的课堂讲解和讲义。
   * mugrade 只检查正确性，因此可以按输出矩阵中的 (i, j) 位置进行简单并行。
   * 不过，为充分练习本题，建议使用课堂介绍的协同取数、共享内存、寄存器分块等方法。
   * 与 CPU 后端的分块矩阵乘法不同，这里只实现一个函数，必须支持任意大小的矩阵，
   * 无论其维度是否为分块大小的倍数。
   * 与前面的 CUDA 实现一样，本函数主要负责设置 kernel 调用，
   * 实际计算逻辑应写在单独的 MatmulKernel() 中。
   *
   * 参数：
   *   a：大小为 M × N 的紧凑二维数组。
   *   b：大小为 N × P 的紧凑二维数组。
   *   out：大小为 M × P 的紧凑二维输出数组。
   *   M：a 和 out 的行数。
   *   N：a 的列数，也是 b 的行数。
   *   P：b 和 out 的列数。
   */

  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// 最大值和求和归约
////////////////////////////////////////////////////////////////////////////////


void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * 对每组 reduce_size 个连续元素取最大值。
   * 为简化实现，可由单个 CUDA 线程完成一次归约，尽管这并不高效。
   *
   * 参数：
   *   a：待归约的紧凑数组，a.size = out.size * reduce_size。
   *   out：用于写入结果的紧凑数组。
   *   reduce_size：待归约维度的大小。
   */
  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * 对每组 reduce_size 个连续元素求和。
   * 同样，为简化实现，可由单个 CUDA 线程完成一次归约。
   *
   * 参数：
   *   a：待归约的紧凑数组，a.size = out.size * reduce_size。
   *   out：用于写入结果的紧凑数组。
   *   reduce_size：待归约维度的大小。
   */
  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}

}  // 命名空间 cuda
}  // 命名空间 needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // 将 GPU 数据复制到 CPU，并返回 NumPy 数组
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // 将内存中的数据复制到主机
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // 返回 NumPy 数组
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // 将 NumPy 数组复制到 GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  // m.def("ewise_mul", EwiseMul);
  // m.def("scalar_mul", ScalarMul);
  // m.def("ewise_div", EwiseDiv);
  // m.def("scalar_div", ScalarDiv);
  // m.def("scalar_power", ScalarPower);

  // m.def("ewise_maximum", EwiseMaximum);
  // m.def("scalar_maximum", ScalarMaximum);
  // m.def("ewise_eq", EwiseEq);
  // m.def("scalar_eq", ScalarEq);
  // m.def("ewise_ge", EwiseGe);
  // m.def("scalar_ge", ScalarGe);

  // m.def("ewise_log", EwiseLog);
  // m.def("ewise_exp", EwiseExp);
  // m.def("ewise_tanh", EwiseTanh);

  // m.def("matmul", Matmul);

  // m.def("reduce_max", ReduceMax);
  // m.def("reduce_sum", ReduceSum);
}
