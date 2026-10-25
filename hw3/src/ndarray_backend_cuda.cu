#include <cuda_runtime.h>
#include <math_constants.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
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
        // cudaMalloc 设计成了把指针地址传进去改写
        // 接收二级指针 &ptr，然后在显卡显存里切出一块空间，把显存的起始物理地址填进ptr里
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
     * 计算一维 CUDA kernel 调用所需的线程块和网格维度。作为kernel function的启动配置
        输入任务数量
     */
    CudaDims dim;
    // BASE_THREAD_NUM表示一个block中thread的数量，向上取整
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

// 转换的是shape、strides数组
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

// 全局编号gid = 之前完整block的数量 * 每个block中的线程数 + 当前线程在自己block中的编号
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

__device__ auto GetIndex(size_t index, const CudaVec& shape, const CudaVec& strides, size_t offset)
    -> size_t {
    size_t res = offset;
    for (int i = static_cast<int>(shape.size) - 1; i >= 0; i--) {  // 从后往前遍历
        auto cur = index % shape.data[i];
        res += cur * strides.data[i];
        index /= shape.data[i];
    }
    return res;
}

__global__ void CompactKernel(
    const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
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
    if (gid >= size) {
        return;
    }
    /// BEGIN SOLUTION
    // gid直接可以作为紧凑数组out的index
    // 问题就是如何仅通过gid求出a的index
    auto index = GetIndex(gid, shape, strides, offset);
    out[gid] = a[index];
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
    CompactKernel<<<dim.grid, dim.block>>>(
        a.ptr, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(
    const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) {
        return;
    }
    auto index = GetIndex(gid, shape, strides, offset);
    out[index] = a[gid];
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
    CudaDims dim = CudaOneDim(a.size);
    EwiseSetitemKernel<<<dim.grid, dim.block>>>(
        a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
    /// END SOLUTION
}

__global__ void ScalarSetitemKernel(
    scalar_t* out, scalar_t val, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) {
        return;
    }
    auto index = GetIndex(gid, shape, strides, offset);
    out[index] = val;
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
    CudaDims dim = CudaOneDim(size);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(
        out->ptr, val, size, VecToCuda(shape), VecToCuda(strides), offset);
    /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// 逐元素和标量操作
////////////////////////////////////////////////////////////////////////////////

// 每个线程处理一个元素：先检查边界，再读取输入。
#define EWISE_OP(name, kernel_name, expr)                                        \
    __global__ void kernel_name(                                                 \
        const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {      \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                      \
        if (gid >= size) return;                                                 \
        const scalar_t x = a[gid];                                               \
        const scalar_t y = b[gid];                                               \
        out[gid] = (expr);                                                       \
    }                                                                            \
    void name(const CudaArray& a, const CudaArray& b, CudaArray* out) {          \
        if (out->size == 0) return;                                              \
        CudaDims dim = CudaOneDim(out->size);                                    \
        kernel_name<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
    }

EWISE_OP(EwiseAdd, EwiseAddKernel, x + y);
EWISE_OP(EwiseMul, EwiseMulKernel, x* y);
EWISE_OP(EwiseDiv, EwiseDivKernel, x / y);
EWISE_OP(EwiseMaximum, EwiseMaximumKernel, fmaxf(x, y));
EWISE_OP(EwiseEq, EwiseEqKernel, x == y);
EWISE_OP(EwiseGe, EwiseGeKernel, x >= y);
#undef EWISE_OP

#define SCALAR_OP(name, kernel_name, expr)                                                     \
    __global__ void kernel_name(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                    \
        if (gid >= size) return;                                                               \
        const scalar_t x = a[gid];                                                             \
        out[gid] = (expr);                                                                     \
    }                                                                                          \
    void name(const CudaArray& a, scalar_t val, CudaArray* out) {                              \
        if (out->size == 0) return;                                                            \
        CudaDims dim = CudaOneDim(out->size);                                                  \
        kernel_name<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);                 \
    }
SCALAR_OP(ScalarAdd, ScalarAddKernel, x + val);
SCALAR_OP(ScalarMul, ScalarMulKernel, x* val);
SCALAR_OP(ScalarDiv, ScalarDivKernel, x / val);
SCALAR_OP(ScalarPower, ScalarPowerKernel, powf(x, val));
SCALAR_OP(ScalarMaximum, ScalarMaximumKernel, fmaxf(x, val));
SCALAR_OP(ScalarEq, ScalarEqKernel, x == val);
SCALAR_OP(ScalarGe, ScalarGeKernel, x >= val);
#undef SCALAR_OP

#define UNARY_OP(name, kernel_name, expr)                                        \
    __global__ void kernel_name(const scalar_t* a, scalar_t* out, size_t size) { \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                      \
        if (gid >= size) return;                                                 \
        const scalar_t x = a[gid];                                               \
        out[gid] = (expr);                                                       \
    }                                                                            \
    void name(const CudaArray& a, CudaArray* out) {                              \
        if (out->size == 0) return;                                              \
        CudaDims dim = CudaOneDim(out->size);                                    \
        kernel_name<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);        \
    }
UNARY_OP(EwiseLog, EwiseLogKernel, logf(x));
UNARY_OP(EwiseExp, EwiseExpKernel, expf(x));
UNARY_OP(EwiseTanh, EwiseTanhKernel, tanhf(x));
#undef UNARY_OP

////////////////////////////////////////////////////////////////////////////////

// 与模块的 __tile_size__（CPU 分块布局接口）分开命名：这是 CUDA block 的输出块边长。
constexpr unsigned MATMUL_BLOCK_TILE = 16;

__global__ void MatmulKernel(
    const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P) {
    unsigned tx = threadIdx.x;
    unsigned ty = threadIdx.y;
    // 一个 block 负责 C 的 16×16 区域，每个线程负责其中一个输出元素。
    size_t row = static_cast<size_t>(blockIdx.y) * MATMUL_BLOCK_TILE + ty;
    size_t col = static_cast<size_t>(blockIdx.x) * MATMUL_BLOCK_TILE + tx;

    // 静态共享内存：大小在编译时确定，无需传入第三个 kernel 启动参数。
    // 同一 block 的线程共同使用这两个 tile；不同 block 各有独立的副本。
    __shared__ scalar_t tile_a[MATMUL_BLOCK_TILE][MATMUL_BLOCK_TILE];
    __shared__ scalar_t tile_b[MATMUL_BLOCK_TILE][MATMUL_BLOCK_TILE];
    // 每个线程独有的累加器，跨所有输入 tile 累积同一个 C[row, col]。
    scalar_t acc = 0;

    // 沿公共维度 N 每次推进 16 项，分段计算 A 的一行与 B 的一列的点积。
    for (size_t base = 0; base < N; base += MATMUL_BLOCK_TILE) {
        // 每个线程各搬一个 A 元素和一个 B 元素，相邻 tx 读取相邻内存。
        // 最后一个 tile 可能不完整：越界部分填 0，使其不贡献乘积。
        tile_a[ty][tx] = (row < M && base + tx < N) ? a[row * N + base + tx] : 0;
        tile_b[ty][tx] = (base + ty < N && col < P) ? b[(base + ty) * P + col] : 0;

        // 第一道屏障：等整个 block 都完成加载，才能读其他线程搬来的值。
        __syncthreads();
        for (unsigned k = 0; k < MATMUL_BLOCK_TILE; ++k) {
            // ty、tx 固定，k 变化：当前 A tile 的第 ty 行点乘 B tile 的第 tx 列。
            // 一个 A 值可被同行的 16 个线程复用，B 值可被同列的 16 个线程复用。
            acc += tile_a[ty][k] * tile_b[k][tx];
        }
        // 第二道屏障：等所有线程读完，才允许下一轮覆盖共享内存。
        // 即使线程对应的输出越界，也不能在屏障之前提前 return。
        __syncthreads();
    }
    // 所有 N 项都累积完才写出。只过滤写入，不影响前面的协作加载与同步。
    if (row < M && col < P) out[row * P + col] = acc;
}

void Matmul(
    const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N, uint32_t P) {
    // A: M×N，B: N×P，C: M×P，均为行优先紧凑布局。
    if (M == 0 || P == 0) return;
    // 一个 Block 负责算出矩阵 C 中某一个16*16子矩阵的全部数值
    dim3 block(MATMUL_BLOCK_TILE, MATMUL_BLOCK_TILE);
    // x 方向覆盖输出列 P，y 方向覆盖输出行 M；向上取整以覆盖尾部。
    dim3 grid((static_cast<size_t>(P) + MATMUL_BLOCK_TILE - 1) / MATMUL_BLOCK_TILE,
        (static_cast<size_t>(M) + MATMUL_BLOCK_TILE - 1) / MATMUL_BLOCK_TILE);
    MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// 最大值和求和归约
////////////////////////////////////////////////////////////////////////////////

// 每线程先处理约 4 个元素，线程数向上取为 2 的幂，最多 256 个。
// 长组由线程循环取数；短组不必分配完整的 256 个线程。
unsigned ReduceBlockSize(size_t reduce_size) {
    size_t target = reduce_size / 4 + (reduce_size % 4 != 0);
    unsigned threads = 1;
    while (threads < target && threads < BASE_THREAD_NUM) threads *= 2;
    return threads;
}

__global__ void ReduceMaxKernel(const scalar_t* in, scalar_t* out, size_t reduce_size) {
    // __shared__ 声明 block 内线程共同访问的共享内存，各 block 的内存相互独立。
    // extern + [] 表示动态大小，由 kernel 启动参数的第三项指定字节数。
    // 每个线程使用一个槽位，所以需要 blockDim.x * sizeof(scalar_t) 字节。
    extern __shared__ scalar_t shared[];
    unsigned tid = threadIdx.x;
    // 一个 block 处理一组，blockIdx.x 同时也是输出元素编号。
    size_t start = static_cast<size_t>(blockIdx.x) * reduce_size;

    // 第一阶段：线程在自己的局部变量中累积部分结果。
    // 例如组长 16、线程数 4：线程 0 读 0、4、8、12，线程 1 读 1、5、9、13。
    // 相邻线程读取相邻元素，有利于合并显存访问。组长无需是线程数的整数倍。
    // 没有读到元素的线程保留单位元：求和是 0，最大值是负无穷。
    scalar_t local = -CUDART_INF_F;
    for (size_t i = tid; i < reduce_size; i += blockDim.x) {
        local = fmaxf(local, in[start + i]);
    }
    shared[tid] = local;
    // 等所有线程写完，再读取其他线程的槽位。局部变量本身无法供其他线程读取。
    __syncthreads();

    // 第二阶段：树形合并。例如 4 个局部结果 s0、s1、s2、s3：
    // stride=2：得到 s0 与 s2 的合并、s1 与 s3 的合并；
    // stride=1：把上述两个结果合并到 shared[0]。
    // 线程数必须是 2 的幂，否则这个减半算法可能遗漏元素。
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        // 必须在 if 外同步！所有线程都要到达这里，且不能提前 return。
        // 这保证下一轮读取的是本轮已经更新的结果。
        __syncthreads();
    }
    // 每个 block 仅线程 0 写出结果，无需原子操作。
    if (tid == 0) out[blockIdx.x] = shared[0];
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
    if (out->size == 0) return;
    if (reduce_size == 0) throw std::invalid_argument("Cannot reduce max over an empty group");
    unsigned threads = ReduceBlockSize(reduce_size);
    size_t shared_bytes = threads * sizeof(scalar_t);
    // <<<block 数, 每个 block 的线程数, 每个 block 的共享内存字节数>>>
    // 一个block分配一个reduce区域内的任务，一个线程处理4个相邻元素
    // 这里一个 block 产生一个输出，不能再使用 CudaOneDim 的每线程一个输出布局。
    ReduceMaxKernel<<<out->size, threads, shared_bytes>>>(a.ptr, out->ptr, reduce_size);
}

__global__ void ReduceSumKernel(const scalar_t* in, scalar_t* out, size_t reduce_size) {
    // __shared__ 声明 block 内线程共同访问的共享内存，各 block 的内存相互独立。
    // extern + [] 表示动态大小，由 kernel 启动参数的第三项指定字节数。
    // 每个线程使用一个槽位，所以需要 blockDim.x * sizeof(scalar_t) 字节。
    extern __shared__ scalar_t shared[];
    unsigned tid = threadIdx.x;
    // 一个 block 处理一组，blockIdx.x 同时也是输出元素编号。
    size_t start = static_cast<size_t>(blockIdx.x) * reduce_size;

    // 第一阶段：线程在自己的局部变量中累积部分结果。
    // 例如组长 16、线程数 4：线程 0 读 0、4、8、12，线程 1 读 1、5、9、13。
    // 相邻线程读取相邻元素，有利于合并显存访问。组长无需是线程数的整数倍。
    // 没有读到元素的线程保留单位元：求和是 0，最大值是负无穷。
    scalar_t local = 0;
    for (size_t i = tid; i < reduce_size; i += blockDim.x) {
        local += in[start + i];
    }
    shared[tid] = local;
    // 等所有线程写完，再读取其他线程的槽位。局部变量本身无法供其他线程读取。
    __syncthreads();

    // 第二阶段：树形合并。例如 4 个局部结果 s0、s1、s2、s3：
    // stride=2：得到 s0 与 s2 的合并、s1 与 s3 的合并；
    // stride=1：把上述两个结果合并到 shared[0]。
    // 线程数必须是 2 的幂，否则这个减半算法可能遗漏元素。
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        // 必须在 if 外同步！所有线程都要到达这里，且不能提前 return。
        // 这保证下一轮读取的是本轮已经更新的结果。
        __syncthreads();
    }
    // 每个 block 仅线程 0 写出结果，无需原子操作。
    if (tid == 0) out[blockIdx.x] = shared[0];
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
    if (out->size == 0) return;
    unsigned threads = ReduceBlockSize(reduce_size);
    size_t shared_bytes = threads * sizeof(scalar_t);
    // <<<block 数, 每个 block 的线程数, 每个 block 的共享内存字节数>>>
    // 这里一个 block 产生一个输出，不能再使用 CudaOneDim 的每线程一个输出布局。
    ReduceSumKernel<<<out->size, threads, shared_bytes>>>(a.ptr, out->ptr, reduce_size);
}

}  // namespace cuda
}  // namespace needle

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
    m.def("to_numpy",
        [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
            size_t offset) {
            std::vector<size_t> numpy_strides = strides;
            std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                [](size_t& c) { return c * ELEM_SIZE; });

            // 将内存中的数据复制到主机
            scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
            if (host_ptr == 0) throw std::bad_alloc();
            cudaError_t err =
                cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

            // 返回 NumPy 数组
            py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
            return py::array_t<scalar_t>(
                shape, numpy_strides, host_ptr + offset, deallocate_buffer);
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

    m.def("ewise_mul", EwiseMul);
    m.def("scalar_mul", ScalarMul);
    m.def("ewise_div", EwiseDiv);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);

    m.def("ewise_maximum", EwiseMaximum);
    m.def("scalar_maximum", ScalarMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("scalar_eq", ScalarEq);
    m.def("ewise_ge", EwiseGe);
    m.def("scalar_ge", ScalarGe);

    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);

    m.def("matmul", Matmul);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}
