10344300# Colby Tomita 12/1/2023

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

res_filter = (
    '09379900', '09422500', '07199450', '07205500', '08294200', '08341400',
    '10254005', '10338400', '10340300', '10342900', '10344300', '10344490',
    '11020600', '11022100', '11042510', '11109700', '11122000', '11128300',
    '11149300', '11150100', '11162618', '11173490', '11451290'
)


# res_meta = {
#     'Maloya': {
#         'stn_id': '07199450',
#         'loc': '-104.3738889, 36.9838333',
#     },
#     'Eagle Nest': {
#         'stn_id': '07205500',
#         'loc': '-105.22945, 36.5314222'
#     }
# }

res_meta = {
    "LAKE POWELL AT GLEN CANYON DAM, AZ": {
        "stn_id": "09379900",
        "loc": "-111.4980000, 36.9985000",
        "state": "az"
    },
    "LAKE MOHAVE AT DAVIS DAM, AZ-NV": {
        "stn_id": "09422500",
        "loc": "-114.5852000, 35.1545000",
        "state": "az"
    },
    "LAKE MALOYA NR RATON, NM": {
        "stn_id": "07199450",
        "loc": "-104.4016000, 36.8984000",
        "state": "nm"
    },
    "EAGLE NEST LAKE NR EAGLE NEST, NM": {
        "stn_id": "07205500",
        "loc": "-105.2300000, 36.4531000",
        "state": "nm"
    },
    "NAMBE FALLS RESERVOIR NEAR NAMBE, NM": {
        "stn_id": "08294200",
        "loc": "-105.9061000, 35.8456000",
        "state": "nm"
    },
    "BLUEWATER LAKE NEAR BLUEWATER, NM": {
        "stn_id": "08341400",
        "loc": "-108.1115000, 35.2926000",
        "state": "nm"
    },
    "SALTON SEA NR WESTMORLAND, CA": {
        "stn_id": "10254005",
        "loc": "-115.7170000, 33.1435000",
        "state": "ca"
    },
    "DONNER LK NR TRUCKEE, CA": {
        "stn_id": "10338400",
        "loc": "-120.2475000, 39.3266000",
        "state": "ca"
    },
    "PROSSER C RES NR TRUCKEE, CA": {
        "stn_id": "10340300",
        "loc": "-120.1980000, 39.3585000",
        "state": "ca"
    },
    "INDEPENDENCE LK NR TRUCKEE, CA": {
        "stn_id": "10342900",
        "loc": "-120.2620000, 39.3182000",
        "state": "ca"
    },
    "STAMPEDE RES NR BOCA, CA": {
        "stn_id": "10344300",
        "loc": "-120.3175000, 39.3406000",
        "state": "ca"
    },
    "BOCA RES NR TRUCKEE, CA": {
        "stn_id": "10344490",
        "loc": "-120.2642000, 39.3268000",
        "state": "ca"
    },
    "EL CAPITAN RES NR LAKESIDE, CA": {
        "stn_id": "11020600",
        "loc": "-116.9165000, 32.8545000",
        "state": "ca"
    },
    "SAN VICENTE RES NR LAKESIDE, CA": {
        "stn_id": "11022100",
        "loc": "-116.9904000, 32.9454000",
        "state": "ca"
    },
    "VAIL LK NR TEMECULA, CA": {
        "stn_id": "11042510",
        "loc": "-116.8714000, 33.5202000",
        "state": "ca"
    },
    "LK PIRU NR PIRU, CA": {
        "stn_id": "11109700",
        "loc": "-118.7331000, 34.4830000",
        "state": "ca"
    },
    "SANTA YNEZ R AB GIBRALTAR DAM NR SANTA BARBARA, CA": {
        "stn_id": "11122000",
        "loc": "-119.6950000, 34.5518000",
        "state": "ca"
    },
    "ALISAL RES NR SOLVANG, CA": {
        "stn_id": "11128300",
        "loc": "-120.1261000, 34.5927000",
        "state": "ca"
    },
    "NACIMIENTO RES NR BRADLEY, CA": {
        "stn_id": "11149300",
        "loc": "-121.1826000, 36.6800000",
        "state": "ca"
    },
    "SAN ANTONIO RES NR BRADLEY, CA": {
        "stn_id": "11150100",
        "loc": "-121.2951000, 36.6012000",
        "state": "ca"
    },
    "PILARCITOS LK NR HILLSBOROUGH, CA": {
        "stn_id": "11162618",
        "loc": "-122.4667000, 37.5890000",
        "state": "ca"
    },
    "CALAVERAS RESERVOIR NR SUNOL, CA": {
        "stn_id": "11173490",
        "loc": "-121.8447000, 37.5781000",
        "state": "ca"
    },
    "INDIAN VALLEY RES A CLEARLAKE OAKS, CA": {
        "stn_id": "11451290",
        "loc": "-122.5349812, 39.0804486",
        "state": "ca"
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
    states = ['az','nm','ca']
    for option in options:
        for state in states:
            response = requests.get(res_url.format(state, time_frame['start'], time_frame['end'], options[option]))
            lines = response.text.strip().split('\n')
            data = [line.split('\t') for line in lines if not line.startswith('#') and not line.startswith('5s') and not line.startswith('agency_cd')]
            df = pd.DataFrame(columns=['agency_cd', 'stn_id', 'datetime', 'storage_value', 'cd'])
            # for line in data:
            #     if line[1] in res_filter:
            #         df.loc[-1] = (pd.Series(line, index=df.columns))
            #         df.reset_index(drop=True, inplace=True)
            for line in data:
                # Check if the length of the line matches the number of columns in df
                if len(line) != len(df.columns):
                    continue  # Skip this station if lengths don't match

                if line[1] in res_filter:
                    df.loc[-1] = (pd.Series(line, index=df.columns))
                    df.reset_index(drop=True, inplace=True)
            df = df.drop(df.columns[0], axis=1)
            df = df.drop(df.columns[-1], axis=1)
            df.rename(columns={df.columns[-1]: option}, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'])

            if option == 'res_elevation':
                res_elev_data = pd.concat([res_elev_data, df], ignore_index=True)
            elif option == 'res_storage':
                res_storage_data = pd.concat([res_storage_data, df], ignore_index=True)

    
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