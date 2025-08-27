from OcchioOnniveggente.src.storage import cache

class DummyCache:
    def __init__(self):
        self.store = {}
    def set(self, key, value, ex=None):
        self.store[key] = (value, ex)


def test_cache_set_writes_when_cache_available(monkeypatch):
    dummy = DummyCache()
    monkeypatch.setattr(cache, "_cache", dummy)
    cache.cache_set("foo", "bar", ex=123)
    assert dummy.store["foo"] == ("bar", 123)
