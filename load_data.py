import flask
from sqlalchemy import create_engine
import pandas as pd

db_path = 'sqlite:///filialdata.db'
data_to_load = "data/Filialdaten.xlsx"

df = pd.read_excel(data_to_load)

engine = create_engine(db_path)

df.to_sql("filialdata", engine, if_exists='replace', index=False)