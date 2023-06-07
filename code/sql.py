import pymysql

def connect():
    # Open database connection
    mysql_conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', db='windcalculate')
    return mysql_conn

def close_connect(mysql_conn):
    # disconnect from server
    mysql_conn.close()


def execute_sql(mysql_conn, cursor, sql):
    cursor.execute(sql)
    mysql_conn.commit()
    r = cursor.fetchall()
    return r

def insert_gridinfo(mysql_conn, cursor, grid):
    sql = f'''INSERT INTO GRIDINFO
    (GRIDID, X, Y, Z, MSG)
    VALUES({grid});
    '''
    r = execute_sql(mysql_conn, cursor, sql)

def insert_datecalculate(mysql_conn, cursor, date):
    sql = f'''INSERT INTO datecalculate
    (calculatetime, 
    1dwind, 1dtemperature,
    2dwind, 2dtemperature,
    3dwind, 3dtemperature,
    4dwind, 4dtemperature,
    5dwind, 5dtemperature,
    6dwind, 6dtemperature,
    7dwind, 7dtemperature
    )
    VALUES({date});
    '''
    r = execute_sql(mysql_conn, cursor, sql)


def insert_hourcalculate(mysql_conn, cursor, hour):
    sql = f'''INSERT INTO hourcalculate
    (gridid, calculatetime, 
    1hwind,  1htemperature,  1hresult,
    2hwind,  2htemperature,  2hresult,
    3hwind,  3htemperature,  3hresult,
    4hwind,  4htemperature,  4hresult,
    5hwind,  5htemperature,  5hresult,
    6hwind,  6htemperature,  6hresult,
    7hwind,  7htemperature,  7hresult,
    8hwind,  8htemperature,  8hresult,
    9hwind,  9htemperature,  9hresult,
    10hwind, 10htemperature, 10hresult,
    11hwind, 11htemperature, 11hresult,
    12hwind, 12htemperature, 12hresult,
    13hwind, 13htemperature, 13hresult,
    14hwind, 14htemperature, 14hresult,
    15hwind, 15htemperature, 15hresult,
    16hwind, 16htemperature, 16hresult,
    17hwind, 17htemperature, 17hresult,
    18hwind, 18htemperature, 18hresult
    )
    VALUES({hour});
    '''
    r = execute_sql(mysql_conn, cursor, sql)