# Colby Tomita 12/1/2023

# Importing the libraries
import pandas as pd
import os
from dotenv import load_dotenv
import psycopg2 as DB
import requests
import json
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

time_frame = {
    "start": "2010-01-01",
    "end": "2023-10-31"
}

res_filter = ('07199450', '07205500')

res_meta = {
    'Maloya': {
        'stn_id': '07199450',
        'loc': '-104.3738889, 36.9838333',
    },
    'Eagle Nest': {
        'stn_id': '07205500',
        'loc': '-105.22945, 36.5314222'
    }
}

load_dotenv()
res_url = os.getenv("RES_URL")
grid_url = os.getenv("GRID_URL")

create_table = """
    CREATE TABLE IF NOT EXISTS res_data (
        stn_id VARCHAR(10),
        datetime TIMESTAMP,
        pcpn FLOAT,
        avgt FLOAT,
        res_elevation FLOAT,
        res_storage FLOAT
    )
"""

def get_data(time_frame):
    res_elev_data = pd.DataFrame()
    res_storage_data = pd.DataFrame()
    merged_df = pd.DataFrame()
    options = {
            'res_elevation': '62614',
            'res_storage': '00054',
        }
    for option in options:
        response = requests.get(res_url.format(time_frame['start'], time_frame['end'], options[option]))
        lines = response.text.strip().split('\n')
        data = [line.split('\t') for line in lines if not line.startswith('#') and not line.startswith('5s') and not line.startswith('agency_cd')]
        df = pd.DataFrame(columns=['agency_cd', 'stn_id', 'datetime', 'storage_value', 'cd'])
        for line in data:
            if line[1] in res_filter:
                df.loc[-1] = (pd.Series(line, index=df.columns))
                df.reset_index(drop=True, inplace=True)
        df = df.drop(df.columns[0], axis=1)
        df = df.drop(df.columns[-1], axis=1)
        df.rename(columns={df.columns[-1]: option}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])

        if option == 'res_elevation':
            res_elev_data = df
        elif option == 'res_storage':
            res_storage_data = df
    
    merged_df = pd.merge(res_elev_data, res_storage_data, on=['datetime', 'stn_id'])
    
    res_climate_data = pd.DataFrame(columns=['stn_id', 'pcpn', 'avgt'])
    for res in res_meta:
        params = {"loc":"{}".format(res_meta[res]['loc']),
                  "sdate":"{}".format(time_frame['start']),
                  "edate":"{}".format(time_frame['end']),
                  "elems":[
                      {"name":"pcpn","interval":"dly","duration":"dly"},
                      {"name":"avgt","interval":"dly","duration":"dly"}
                  ],
                  "grid":"21"}
        k = requests.post(grid_url, data=json.dumps(params), headers = {'content-type': 'application/json'}, timeout = 60)
        df = pd.DataFrame(k.json()['data'])
        df.columns = ['datetime', 'pcpn', 'avgt']
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['stn_id'] = res_meta[res]['stn_id']
        res_climate_data = pd.concat([res_climate_data, df])

    merged_df = pd.merge(merged_df, res_climate_data, on=['datetime', 'stn_id'])
    return merged_df

def push_data(df, conn):
    cur = conn.cursor()
    cur.execute(create_table)
    for index, row in df.iterrows():
        cur.execute("INSERT INTO res_data VALUES (%s, %s, %s, %s, %s, %s)", (row['stn_id'], row['datetime'], row['pcpn'], row['avgt'], row['res_elevation'], row['res_storage']))
    conn.commit()
    cur.close()

def create_db():
    conn = DB.connect(f'host={os.getenv("HOST")} port={os.getenv("PORT")} dbname={os.getenv("DEFAULT_DBNAME")} user={os.getenv("DBUSER")}')
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    dbname = os.getenv("DBNAME")
    cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{dbname}'")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f"CREATE DATABASE {dbname}")
    conn.commit()
    cur.close()
    conn.close()


def main():
    create_db()
    conn = DB.connect(f'host={os.getenv("HOST")} port={os.getenv("PORT")} dbname={os.getenv("DBNAME")} user={os.getenv("DBUSER")}')
    df = get_data(time_frame)
    push_data(df, conn)
    
if __name__ == "__main__":
    main()