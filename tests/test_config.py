"""Settings helpers: CORS origin parsing + the production flag."""

from app.config import Settings


def test_cors_origins_splits_and_trims():
    s = Settings()
    s.CORS_ALLOW_ORIGINS = "https://a.com, https://b.com ,"
    assert s.cors_origins == ["https://a.com", "https://b.com"]


def test_cors_origins_wildcard():
    s = Settings()
    s.CORS_ALLOW_ORIGINS = "*"
    assert s.cors_origins == ["*"]


def test_is_production_case_insensitive():
    s = Settings()
    s.APP_ENV = "production"
    assert s.is_production is True
    s.APP_ENV = "Development"
    assert s.is_production is False
