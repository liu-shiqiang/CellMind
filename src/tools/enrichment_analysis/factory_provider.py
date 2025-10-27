from .gsea import GSEAFactory
from .ssgsea import SSGSEAFactory
from .gsva import GSVAFactory
from .ora import ORAFactory
# from .scPAFA import SCPAFAFactory

# 其他方法工厂也可以逐步加进来

FACTORY_MAP = {
    "gsea": GSEAFactory,
    "ssgsea": SSGSEAFactory,
    "gsva": GSVAFactory,
    "ora": ORAFactory,
    # "scPAFA": SCPAFAFactory,
}

def get_factory(method: str):
    key = method.lower()
    if key not in FACTORY_MAP:
        raise ValueError(f"Unsupported enrichment method: {method}")
    return FACTORY_MAP[key]()