import random
from typing import Iterable, List

import numpy as np
import pytest
from rdkit import Chem

from molpal.featurizer import Featurizer

def interleave(a: Iterable, b: Iterable) -> List:
    return list(map(next, random.sample([iter(a)]*len(a) + [iter(b)]*len(b), len(a)+len(b))))

@pytest.fixture(params=[
    ['c1ccccc1', 'CCCC', 'CC(=O)C', 'CN=C=O',
     'CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C'],
    ['n1c[nH]cc1', 'O=Cc1ccc(O)c(OC)c1', 'CC(=O)NCCC1=CNc2c1cc(OC)cc2'],
    ['FC(Br)(Cl)F', 'c1cc(ccc1)C', 'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5']
])
def smis(request):
    return request.param

@pytest.fixture(params=[
    ['hello', 'Ccc1cCC3C', 'bob', 'not_a_molecule'],
    ['foo', 'baz', 'bar'],
])
def bad_smis(request):
    return request.param

@pytest.fixture
def some_bad_smis(smis, bad_smis):
    return interleave(smis, bad_smis)

@pytest.fixture(params=['pair', 'morgan', 'rdkit'])
def fingerprint(request):
    return request.param

@pytest.fixture(params=[1,2,3])
def radius(request):
    return request.param

@pytest.fixture(params=[512, 1024, 2048])
def length(request):
    return request.param

@pytest.fixture
def featurizer(fingerprint, radius, length):
    return Featurizer(fingerprint, radius, length)

def test_maccs_legth(radius, length):
    featurizer = Featurizer("maccs", radius, length)

    assert len(featurizer) == 167

def test_filter(featurizer, bad_smis):
    fps = [featurizer(smi) for smi in bad_smis]

    assert all(fp is None for fp in fps)

def test_dimension(featurizer, smis):
    fps = np.array([featurizer(smi) for smi in smis])

    assert fps.shape == (len(smis), len(featurizer))

def test_dimension_some_bad(featurizer, some_bad_smis):
    fps = [featurizer(smi) for smi in some_bad_smis]
    fps = np.array([fp for fp in fps if fp is not None])

    mols = [Chem.MolFromSmiles(smi) for smi in some_bad_smis]
    num_valid = sum(1 for m in mols if m is not None)

    assert fps.shape == (num_valid, len(featurizer))