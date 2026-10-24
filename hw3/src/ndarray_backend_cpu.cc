#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

/**
 * 用于管理按 ALIGNMENT 字节边界对齐的数组的辅助结构体。
 * 对齐大小至少应为 TILE * ELEM_SIZE；此处默认采用更大的值。
 */
struct AlignedArray {
    AlignedArray(const size_t size) {
        int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
        if (ret != 0) throw std::bad_alloc();
        this->size = size;
    }
    ~AlignedArray() { free(ptr); }
    size_t ptr_as_int() { return reinterpret_cast<size_t>(ptr); }
    scalar_t* ptr;
    size_t size;
};

void Fill(AlignedArray* out, scalar_t val) {
    /**
     * 将对齐数组中的所有元素填充为 val。
     */
    for (int i = 0; i < out->size; i++) {
        out->ptr[i] = val;
    }
}
namespace {
void Increase(std::vector<int32_t>& indexes, const std::vector<int32_t>& shape) {
    auto length = indexes.size();
    indexes[length - 1]++;
    for (size_t i = length - 1; i > 0; i--) {
        indexes[i - 1] += indexes[i] / shape[i];
        indexes[i] %= shape[i];
    }
}

auto GetIndex(std::vector<int32_t>& indexes, const std::vector<int32_t>& strides, size_t offset)
    -> size_t {
    auto length = indexes.size();
    auto sum = offset;
    for (size_t i = 0; i < length; i++) {
        sum += indexes[i] * strides[i];
    }
    return sum;
}
}  // namespace

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
    /**
     * 将数组整理为内存中的紧凑布局。
     *
     * 参数：
     *   a：输入的非紧凑数组。
     *   out：用于写入结果的紧凑数组。
     *   shape：a 和 out 各维度的大小。
     *   strides：a 的步长；out 使用紧凑步长。
     *   offset：a 的偏移量；紧凑数组 out 的偏移量为 0。
     *
     * 返回：
     *   void：直接修改 out，不返回结果。本文件中其余待实现函数也遵循此约定。
     */
    /// BEGIN SOLUTION
    size_t length = 1;  // 总长度
    for (size_t i = 0; i < shape.size(); i++) {
        length *= shape[i];
    }

    std::vector<int32_t> a_indexes(shape.size(), 0);
    for (size_t i = 0; i < length; i++) {
        auto index = GetIndex(a_indexes, strides, offset);
        out->ptr[i] = a.ptr[index];
        Increase(a_indexes, shape);
    }
    /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
    /**
     * 设置非紧凑数组中的元素。
     *
     * 参数：
     *   a：紧凑的输入数组，其元素将写入 out。
     *   out：待写入的非紧凑数组。
     *   shape：a 和 out 各维度的大小。
     *   strides：out 的步长；a 使用紧凑步长。
     *   offset：out 的偏移量；紧凑数组 a 的偏移量为 0。
     */
    /// BEGIN SOLUTION
    size_t length = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        length *= shape[i];
    }
    std::vector<int32_t> out_indexes(shape.size(), 0);
    for (size_t i = 0; i < length; i++) {
        auto index = GetIndex(out_indexes, strides, offset);
        out->ptr[index] = a.ptr[i];
        Increase(out_indexes, shape);
    }
    /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
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
    std::vector<int32_t> out_indexes(shape.size(), 0);
    for (size_t i = 0; i < size; i++) {
        auto index = GetIndex(out_indexes, strides, offset);
        out->ptr[index] = val;
        Increase(out_indexes, shape);
    }
    /// END SOLUTION
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

#define EWISE_OP(name, expr)                                                     \
    void name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
        for (size_t i = 0; i < out->size; i++) {                                 \
            const auto x = a.ptr[i];                                             \
            const auto y = b.ptr[i];                                             \
            out->ptr[i] = (expr);                                                \
        }                                                                        \
    }

EWISE_OP(EwiseAdd, x + y);
EWISE_OP(EwiseMul, x* y);
EWISE_OP(EwiseDiv, x / y);
EWISE_OP(EwiseMaximum, std::fmax(x, y));
EWISE_OP(EwiseEq, x == y);
EWISE_OP(EwiseGe, x >= y);

#undef EWISE_OP

#define SCALAR_OP(name, expr)                                           \
    void name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
        for (size_t i = 0; i < a.size; i++) {                           \
            const auto x = a.ptr[i];                                    \
            out->ptr[i] = (expr);                                       \
        }                                                               \
    }

SCALAR_OP(ScalarAdd, x + val);
SCALAR_OP(ScalarMul, x* val);
SCALAR_OP(ScalarDiv, x / val);
SCALAR_OP(ScalarPower, std::pow(x, val));
SCALAR_OP(ScalarMaximum, std::fmax(x, val));
SCALAR_OP(ScalarEq, x == val);
SCALAR_OP(ScalarGe, x >= val);

#undef SCALAR_OP

// 单输入操作只读取 a，与双输入逐元素操作使用相同的连续遍历方式。
#define UNARY_OP(name, expr)                              \
    void name(const AlignedArray& a, AlignedArray* out) { \
        for (size_t i = 0; i < out->size; i++) {          \
            const auto x = a.ptr[i];                      \
            out->ptr[i] = (expr);                         \
        }                                                 \
    }

UNARY_OP(EwiseLog, std::log(x));
UNARY_OP(EwiseExp, std::exp(x));
UNARY_OP(EwiseTanh, std::tanh(x));

#undef UNARY_OP

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
    /**
     * 将两个紧凑矩阵相乘，结果写入同样紧凑的输出矩阵。
     * 本函数可使用朴素的三重循环算法。
     *
     * 参数：
     *   a：大小为 m × n 的紧凑二维数组。
     *   b：大小为 n × p 的紧凑二维数组。
     *   out：大小为 m × p 的紧凑二维输出数组。
     *   m：a 和 out 的行数。
     *   n：a 的列数，也是 b 的行数。
     *   p：b 和 out 的列数。
     */

    /// BEGIN SOLUTION
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            out->ptr[i * p + j] = 0;
        }
    }
    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < n; k++) {
            auto a_ik = a.ptr[i * n + k];
            for (size_t j = 0; j < p; j++) {
                out->ptr[i * p + j] += a_ik * b.ptr[k * p + j];
            }
        }
    }
    /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ out) {
    /**
     * 将两个 TILE × TILE 矩阵相乘，并将结果累加到 out。
     * 必须累加到已有的 out 上，不能在此函数中预先将 out 清零。
     * 这里提供了帮助编译器使用向量运算的标记。
     * __restrict__ 告诉编译器 a、b 和 out 的内存互不重叠；
     * 这是向量化运算与非向量化运算结果一致的必要条件，
     * 可以思考当三者内存重叠时会发生什么。
     * __builtin_assume_aligned 则告诉编译器输入数组满足指定的内存对齐要求，
     * 同样有助于向量化。
     *
     * 参数：
     *   a：大小为 TILE × TILE 的紧凑二维数组。
     *   b：大小为 TILE × TILE 的紧凑二维数组。
     *   out：用于累加结果的 TILE × TILE 紧凑二维数组。
     */

    a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
    b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
    out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

    /// BEGIN SOLUTION
    for (size_t i = 0; i < TILE; i++) {
        for (size_t k = 0; k < TILE; k++) {
            float a_ik = a[i * TILE + k];

            for (size_t j = 0; j < TILE; j++) {
                out[i * TILE + j] += a_ik * b[k * TILE + j];
            }
        }
    }
    /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
    /**
     * 对分块表示的数组执行矩阵乘法。
     * a、b 和 out 都是适当大小的紧凑四维数组，例如 a 的布局为：
     *   a[m/TILE][n/TILE][TILE][TILE]
     * 应逐块进行乘法以提高性能，即调用上面实现的 AlignedDot()。
     *
     * 仅当 m、n、p 都是 TILE 的倍数时才会调用本函数，因此整除不会有余数。
     *
     * 参数：
     *   a：大小为 m/TILE × n/TILE × TILE × TILE 的紧凑四维数组。
     *   b：大小为 n/TILE × p/TILE × TILE × TILE 的紧凑四维数组。
     *   out：大小为 m/TILE × p/TILE × TILE × TILE 的紧凑四维输出数组。
     *   m：a 和 out 的行数。
     *   n：a 的列数，也是 b 的行数。
     *   p：b 和 out 的列数。
     */
    /// BEGIN SOLUTION
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            out->ptr[i * p + j] = 0;
        }
    }
    for (size_t i = 0; i < m / TILE; i++) {
        for (size_t k = 0; k < n / TILE; k++) {
            const float* a_ik = a.ptr + (i * n / TILE + k) * TILE * TILE;
            for (size_t j = 0; j < p / TILE; j++) {
                float* out_ij = out->ptr + (i * p / TILE + j) * TILE * TILE;
                const float* b_kj = b.ptr + (k * p / TILE + j) * TILE * TILE;
                AlignedDot(a_ik, b_kj, out_ij);
            }
        }
    }
    /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
    /**
     * 对每组 reduce_size 个连续元素取最大值。
     *
     * 参数：
     *   a：待归约的紧凑数组，a.size = out.size * reduce_size。
     *   out：用于写入结果的紧凑数组。
     *   reduce_size：待归约维度的大小。
     */

    /// BEGIN SOLUTION
    size_t cnt = 0;
    for (size_t i = 0; i < a.size; i += reduce_size) {
        float max = a.ptr[i];
        for (size_t j = i + 1; j < i + reduce_size; j++) {
            max = a.ptr[j] > max ? a.ptr[j] : max;
        }
        out->ptr[cnt++] = max;
    }
    /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
    /**
     * 对每组 reduce_size 个连续元素求和。
     *
     * 参数：
     *   a：待归约的紧凑数组，a.size = out.size * reduce_size。
     *   out：用于写入结果的紧凑数组。
     *   reduce_size：待归约维度的大小。
     */

    /// BEGIN SOLUTION
    size_t cnt = 0;
    for (size_t i = 0; i < a.size; i += reduce_size) {
        float sum = 0.0;
        for (size_t j = i; j < i + reduce_size; j++) {
            sum += a.ptr[j];
        }
        out->ptr[cnt++] = sum;
    }
    /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
    namespace py = pybind11;
    using namespace needle;
    using namespace cpu;

    m.attr("__device_name__") = "cpu";
    m.attr("__tile_size__") = TILE;

    py::class_<AlignedArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def("ptr", &AlignedArray::ptr_as_int)
        .def_readonly("size", &AlignedArray::size);

    // 返回 NumPy 数组；为简化内存生命周期管理而采用复制方式，
    // 否则垃圾回收处理会比较麻烦。
    m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                         std::vector<size_t> strides, size_t offset) {
        std::vector<size_t> numpy_strides = strides;
        std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                       [](size_t& c) { return c * ELEM_SIZE; });
        return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
    });

    // 从 NumPy 数组复制数据
    m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
        std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
    m.def("matmul_tiled", MatmulTiled);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}
