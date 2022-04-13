import warnings
from molpal.pools.base import MoleculePool, EagerMoleculePool
from molpal.pools.lazypool import LazyMoleculePool


def pool(pool: str, *args, **kwargs):
    try:
<<<<<<< HEAD
        return {
            "eager": MoleculePool,
            "lazy": LazyMoleculePool
        }[pool](*args, **kwargs)
    except KeyError:
        warnings.warn(
            f'Unrecognized pool type: "{pool}". Defaulting to EagerMoleculePool.'
        )
=======
        return {"eager": MoleculePool, "lazy": LazyMoleculePool}[pool](*args, **kwargs)
    except KeyError:
        print(f'WARNING: Unrecognized pool type: "{pool}". Defaulting to EagerMoleculePool.')
>>>>>>> 05a5aa29ef43e65f654271b43730f8d9411aee92
        return MoleculePool(*args, **kwargs)
