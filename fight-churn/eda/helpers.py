import datetime as dt
import os
import pandas as pd
import pathlib
from sqlalchemy import create_engine
from typing import Dict


def connect_db():
    DB_NAME = os.environ["CHURN_DB"],
    USERNAME = os.environ["CHURN_DB_USER"]
    PASSWORD = os.environ["CHURN_DB_PASS"]

    db_str = f"postgresql://{USERNAME}:{PASSWORD}@localhost/{DB_NAME}"
    return create_engine(db_str)


def get_queries(path: str = ".") -> Dict[str, str]:
    d = {}
    p = pathlib.Path(path)
    d = {
        f.stem: f.read_text() for f in p.iterdir() if f.suffix == ".sql" and f.is_file()
    }

    for x in p.iterdir():
        if x.is_dir():
            p = x
            d.update(get_queries(p))
    
    return d


def run_query(query: str) -> pd.DataFrame:
    tic = dt.datetime.now()
    df = pd.read_sql_query(query_dict[query], connection)
    print(
        f"""time taken: {str(dt.datetime.now() - tic)}\n
    shape: {df.shape}
    """
    )
    return df

if __name__ == "__main__":
    connection = connect_db()
    query_dict = get_queries("../eda")
    