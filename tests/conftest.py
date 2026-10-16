import pytest

from provsurvive.sign.c2pa_io import C2paSession
from provsurvive.sign.test_ca import create_test_ca


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory):
    """Tests never read or write the user's master / generative caches."""
    mp = pytest.MonkeyPatch()
    mp.setenv("PROVSURVIVE_CACHE", str(tmp_path_factory.mktemp("cache")))
    yield
    mp.undo()


@pytest.fixture(scope="session")
def session() -> C2paSession:
    return C2paSession(create_test_ca(), thumbnail=False)


@pytest.fixture(scope="session")
def trustmark():
    from provsurvive.bindings.trustmark import TrustMarkBinding

    return TrustMarkBinding(device="cpu")
