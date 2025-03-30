from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .schema import Base


def initialize_database_connection(database_url: str) -> sessionmaker:
    engine = create_engine(database_url, echo=False)
    db_session_maker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
    return db_session_maker
