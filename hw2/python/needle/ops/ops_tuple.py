"""TensorTuple 算子：组合多个输出、选择输出，以及融合标量加法。"""

from ..autograd import NDArray, Op, Tensor, TensorTuple, Value, TensorOp, TensorTupleOp
import needle.init as init


class MakeTensorTuple(TensorTupleOp):
    """将多个 Tensor 包装为一个具有计算图关系的 TensorTuple。"""

    def compute(self, *args: NDArray) -> tuple[NDArray, ...]:
        return tuple(args)

    def gradient(self, out_grad: TensorTuple, node: TensorTuple) -> tuple[Tensor, ...]:
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args: Tensor) -> TensorTuple:
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    """取出元组中的一个 Tensor，反向时其余分量接收零梯度。"""

    def __init__(self, index: int) -> None:
        self.index = index

    def __call__(self, a: TensorTuple, fold_const: bool = True) -> Tensor:
        assert isinstance(a, TensorTuple)
        # 常量折叠：若元组刚由 MakeTensorTuple 创建，直接取原输入，省去冗余节点。
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a: tuple[NDArray, ...]) -> NDArray:
        return a[self.index]

    def gradient(self, out_grad: Tensor, node: Tensor) -> TensorTuple:
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value: TensorTuple, index: int) -> Tensor:
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    """由一个输入同时产生 a+c0 和 a+c1，反向时合并两份梯度。"""

    def __init__(self, c0: float, c1: float) -> None:
        self.c0 = c0
        self.c1 = c1

    def compute(self, a: NDArray) -> tuple[NDArray, NDArray]:
        return a + self.c0, a + self.c1

    def gradient(self, out_grad: TensorTuple, node: TensorTuple) -> Tensor:
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x: Tensor, c0: float, c1: float) -> TensorTuple:
    return FusedAddScalars(c0, c1)(x)
