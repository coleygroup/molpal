import warnings
from molpal.pools.base import MoleculePool, EagerMoleculePool
from molpal.pools.lazypool import LazyMoleculePool


def pool(pool: str, *args, **kwargs):
    try:
        return {"eager": MoleculePool, "lazy": LazyMoleculePool}[pool](*args, **kwargs)
    except KeyError:
        warnings.warn(f'Unrecognized pool type: "{pool}". Defaulting to EagerMoleculePool.')
        return MoleculePool(*args, **kwargs)
