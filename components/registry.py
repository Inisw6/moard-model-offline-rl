import importlib
import os
import pkgutil
from typing import Any, Callable, Dict, Type

# 컴포넌트 키와 클래스 매핑을 저장하는 레지스트리
_REGISTRY: Dict[str, Type[Any]] = {}


def register(key: str) -> Callable[[Type[Any]], Type[Any]]:
    """컴포넌트 클래스를 레지스트리에 등록하는 데코레이터를 반환합니다.

    Args:
        key (str): 레지스트리에 등록할 키 이름.

    Returns:
        Callable[[Type[Any]], Type[Any]]: 클래스를 등록한 뒤 그대로 반환하는 데코레이터.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        """클래스를 레지스트리에 등록하고 반환합니다.

        Args:
            cls (Type[Any]): 등록할 클래스.

        Returns:
            Type[Any]: 등록 후 그대로 반환된 클래스.
        """
        _REGISTRY[key] = cls
        return cls

    return decorator


def make(key: str, **kwargs: Any) -> Any:
    """레지스트리에서 키에 해당하는 컴포넌트 인스턴스를 생성합니다.

    Args:
        key (str): 생성할 컴포넌트의 키.
        **kwargs (Any): 컴포넌트 생성자에 전달할 인자.

    Returns:
        Any: 요청한 키에 매핑된 클래스의 인스턴스.

    Raises:
        KeyError: 레지스트리에 해당 키가 없을 경우.
    """
    if key not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(
            f"No component registered under '{key}'. Available keys: {available}"
        )
    cls = _REGISTRY[key]
    return cls(**kwargs)


# 자동 모듈 로딩: 패키지 디렉토리 내 모든 모듈과 서브패키지 로드
_package_dir = os.path.dirname(__file__)
_package_name = __package__ or ""

for finder, module_name, is_pkg in pkgutil.iter_modules([_package_dir]):
    importlib.import_module(f"{_package_name}.{module_name}")
    if is_pkg:
        subpkg_dir = os.path.join(_package_dir, module_name)
        for _, subname, _ in pkgutil.iter_modules([subpkg_dir]):
            importlib.import_module(f"{_package_name}.{module_name}.{subname}")
