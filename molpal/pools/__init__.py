from .pools import MoleculePool, EagerMoleculePool, LazyMoleculePool

def pool(pool: str, **kwargs):
    try:
        return {
            'eager': MoleculePool,
            'lazy': LazyMoleculePool
        }[pool](**kwargs)
    except KeyError:
        print(f'WARNING: Unrecognized pool type: "{pool}".',
               'Defaulting to EagerMoleculePool.')
        return MoleculePool(**kwargs)