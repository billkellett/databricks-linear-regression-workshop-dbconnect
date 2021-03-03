def setup(p_user_name, p_spark, p_api_token):
    # import to enable removal on non-alphanumeric characters (re stands for regular expressions)
    import re
    from pyspark.dbutils import DBUtils
    import wget
    import requests
    import base64
    import os
    import sys

    # This sys configuration parameter must be set in order for wget to work on the Databricks driver
    sys.stdout.fileno = lambda: False

    print("Starting setup process")

    # Spark session
    spark = p_spark

    # Get the email address entered by the user on the calling notebook
    user_name = p_user_name
    print(f"Data entered in user_name field: {user_name}")

    # Strip out special characters and make it lower case
    clean_user_name = re.sub('[^A-Za-z0-9]+', '', user_name).lower();
    print(f"User Name with special characters removed: {clean_user_name}");

    # Construct the unique path to be used to store files on the local file system
    local_data_path = f"ml-with-sql-{clean_user_name}/"
    print(f"Path to be used for Local Files: {local_data_path}")

    # Construct the unique path to be used to store files on the DBFS file system
    dbfs_data_path = f"ml-with-sql-{clean_user_name}/"
    print(f"Path to be used for DBFS Files: {dbfs_data_path}")

    # Construct the unique database name
    database_name = f"ml_with_sql_{clean_user_name}"
    print(f"Database Name: {database_name}")

    # Create a database using the user's name
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    print(f"Created database {database_name}")

    # Set the new database to be the default
    spark.sql(f"USE {database_name}")

    # The notebook version does this with Python subprocess.Popen()
    # That won't work with dbconnect, because it's a non-spark call,
    # and will run on the local developer machine.
    # Fortunately, dbutils (which resolves to the Databricks cluster) can work on local files using "file:/"
    dbutils = DBUtils(spark)
    dbutils.fs.rm(f"file:/databricks/driver/{local_data_path}", True)  # True means recurse
    print(f"Removed local path {local_data_path}")

    # Similarly, we replace subprocess.Popen() with dbutils in order to CREATE a local path on the driver
    dbutils.fs.mkdirs(f"file:/databricks/driver/{local_data_path}")
    print(f"Recreated local path {local_data_path}")

    # print("listing files on driver")
    # print(dbutils.fs.ls("file:/databricks/driver/"))
    # print("done listing")

    # Now download the data we'll use as input
    # If we're using dbconnect, this will download to the local dev machine into the Pycharm's project directory
    # If we're running in production, this will download to the driver.
    # Since we don't know the download target, we'll just download to the current working directory.
    # We'll give the download a unique name based on the user name parameter, to avoid name collisions.
    # Normally I'd want to avoid name collisions by creating a unique path, but I can't do that since
    # I don't know which machine I'm downloading to.

    # Regardless of the machine, I should be able to delete any previous version of the download
    if os.path.exists(f"{clean_user_name}_download.csv"):
        os.remove(f"{clean_user_name}_download.csv")

    # Now download the file
    url = 'https://raw.githubusercontent.com/billkellett/databricks-ml-with-sql-workshop/main/data/kc_house_data.csv'
    wget.download(url, f"{clean_user_name}_download.csv")
    print(f"Downloaded {clean_user_name}_download.csv")

    # Put downloaded file to DBFS
    # Note: we have to do this with the REST API.  We can't use dbutils because we don't know
    # whether the file is on the local dev machine, or on the cluster driver
    print("Beginning upload to DBFS")
    DOMAIN = 'adb-8245268741408838.18.azuredatabricks.net'
    TOKEN = p_api_token
    BASE_URL = f'https://{DOMAIN}/api/2.0/dbfs/'

    def dbfs_rpc(action, body):
        """ A helper function to make the DBFS API request, request/response is encoded/decoded as JSON """
        response = requests.post(
            BASE_URL + action,
            headers={'Authorization': f'Bearer {TOKEN}'},
            json=body
        )
        return response.json()

    # Create a handle that will be used to add blocks
    handle = \
    dbfs_rpc("create", {"path": f"/FileStore/mlwithsql/{dbfs_data_path}/kc_house_data.csv", "overwrite": "true"})[
        'handle']
    with open(f'{clean_user_name}_download.csv') as f:
        while True:
            # A block can be at most 1MB
            block = f.read(1 << 20)
            if not block:
                break
            # Kellett mods (doc error) the commented line is from doc
            # The following 2 lines are Kellett's replacement
            # data = base64.standard_b64encode(block)
            block_ascii = block.encode('ascii')
            data = base64.b64encode(block_ascii).decode()
            # End of Kellett replacement
            dbfs_rpc("add-block", {"handle": handle, "data": data})
    # close the handle to finish uploading
    dbfs_rpc("close", {"handle": handle})

    print("File uploaded to DBFS")

    # Now create Delta tables from the uploaded data
    dataPath = f"dbfs:/FileStore/mlwithsql/{dbfs_data_path}kc_house_data.csv"

    df = spark.read \
        .option("header", "true") \
        .option("delimiter", ",") \
        .option("inferSchema", "true") \
        .csv(dataPath)

    df.createOrReplaceTempView("house_data_vw")

    spark.sql("""
        DROP TABLE IF EXISTS house_data""")
    spark.sql("""
        CREATE TABLE house_data
        USING DELTA
        AS (
            SELECT * FROM house_data_vw
           )
    """)

    # This table has the price column removed to create an unlabeled version of the house data
    spark.sql("""
       DROP TABLE IF EXISTS house_data_unlabeled""")

    spark.sql("""
    CREATE TABLE house_data_unlabeled 
    USING DELTA
    AS (
      SELECT 
        id,
        date,
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition,
        grade,
        sqft_above,
        sqft_basement,
        yr_built,
        yr_renovated,
        zipcode,
        lat,
        long,
        sqft_living15,
        sqft_lot15    
      FROM house_data_vw
    ) 
    """)

    # If I were coding this from scratch, I wouldn't build my return data this way...
    # However, I'm trying to match my notebook version as closely as possible,
    # and a notebook can only return a string.
    return f"{local_data_path} {dbfs_data_path} {database_name}"

# This allows you to run setup standalone.
# Enter your user name as the first argument in the command line
# E.g., python setup.py Bill
if __name__ == "__main__":
    import argparse
    from pyspark.sql import SparkSession

    # process command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_name', help='Any unique text string that uniquely identifies the user')
    parser.add_argument('--api_token', help='Databricks Personal Access Token')
    parser.add_argument('--run_cleanup_at_eoj', help='y - run cleanup, any other value - do not run cleanup')
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    setup_responses = setup(args.user_name, spark, args.api_token).split()
    local_data_path, dbfs_data_path, database_name = setup_responses

    print(f"Path to be used for Local Files: {local_data_path}")
    print(f"Path to be used for DBFS Files: {dbfs_data_path}")
    print(f"Database Name: {database_name}")