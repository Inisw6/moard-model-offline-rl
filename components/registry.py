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
        Callable[[Type[Any]], Type[Any]]: 클래스를 등록하고 그대로 반환하는 데코레이터.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        """클래스를 레지스트리에 등록하고 반환합니다.

        Args:
            cls (Type[Any]): 등록할 클래스.

        Returns:
            Type[Any]: 등록된 클래스.
        """
        _REGISTRY[key] = cls
        return cls

    return decorator


def make(key: str, **kwargs: Any) -> Any:
    """레지스트리에서 키에 해당하는 컴포넌트를 생성합니다.

    Args:
        key (str): 생성할 컴포넌트의 키.
        **kwargs (Any): 컴포넌트 클래스 생성자에 전달할 인자들.

    Returns:
        Any: 요청한 키에 매핑된 클래스의 인스턴스.

    Raises:
        KeyError: 레지스트리에 해당 키가 없을 경우.
    """
    if key not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(
            "No component registered under '%s'. Available keys: %s" % (key, available)
        )
    cls = _REGISTRY[key]
    return cls(**kwargs)


# 현재 패키지 디렉토리 내 모든 모듈을 자동으로 import하여
# @register 데코레이터 호출을 보장합니다.
_package_dir = os.path.dirname(__file__)
for _finder, module_name, _ispkg in pkgutil.iter_modules([_package_dir]):
    # 1) 컴포넌트 패키지/모듈 import
    importlib.import_module(f"{__package__}.{module_name}")

    # 2) 만약 agents 같은 서브패키지라면, 그 안의 모듈들도 import
    if _ispkg:
        subpkg_dir = os.path.join(_package_dir, module_name)
        for _f2, subname, _isp2 in pkgutil.iter_modules([subpkg_dir]):
            importlib.import_module(f"{__package__}.{module_name}.{subname}")
