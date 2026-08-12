from collections.abc import Sequence
from types import TracebackType
from typing import Literal

class Tensor:
    def item(self) -> float: ...

class _DType: ...

class _NoGrad:
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]: ...

float32: _DType

def tensor(data: Sequence[float], *, dtype: _DType) -> Tensor: ...
def no_grad() -> _NoGrad: ...
