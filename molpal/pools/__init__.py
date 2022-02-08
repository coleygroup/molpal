from molpal.pools.base import MoleculePool, EagerMoleculePool
from molpal.pools.lazypool import LazyMoleculePool

def pool(pool_type: str, *args, **kwargs):
    try:
        return {
            'eager': MoleculePool,
            'lazy': LazyMoleculePool
        }[pool_type](*args, **kwargs)
    except KeyError:
        print(f'WARNING: Unrecognized pool type: "{pool_type}". Defaulting to EagerMoleculePool.')
        return MoleculePool(*args, **kwargs)