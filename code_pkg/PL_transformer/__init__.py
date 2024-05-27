from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {"configuration_plf": ["PLFConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_plf"] = [
        "PLFForPreTraining",
        "PLFLayer",
        "PLFModel",
        "PLFPreTrainedModel",
        "PLFForImageClassification",
        "PLFForJointClassificationRegression",
    ]


if TYPE_CHECKING:
    from .configuration_plf import PLFConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_plf import (
            PLFForPreTraining,
            PLFLayer,
            PLFModel,
            PLFPreTrainedModel,
            PLFForImageClassification,
            PLFForJointClassificationRegression,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
