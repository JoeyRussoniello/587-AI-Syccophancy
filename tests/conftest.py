from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Base


@pytest.fixture
def sqlite_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def session_factory(sqlite_engine):
    return sessionmaker(bind=sqlite_engine)


@pytest.fixture
def session(session_factory) -> Generator[Session, None, None]:
    db_session = session_factory()
    try:
        yield db_session
    finally:
        db_session.close()


@pytest.fixture
def datasets_dir(tmp_path: Path) -> Path:
    return tmp_path / "datasets"
