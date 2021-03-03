def cleanup(p_user_name, p_spark):
    # import to enable removal on non-alphanumeric characters (re stands for regular expressions)
    import os
    import re
    from pyspark.dbutils import DBUtils

    print("Starting cleanup process")

    # Get the email address entered by the user on the calling notebook
    print(f"Data entered in user_name field: {p_user_name}")

    # Strip out special characters and make it lower case
    clean_user_name = re.sub('[^A-Za-z0-9]+', '', p_user_name).lower();
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

    # Remove the user's database if it is present
    p_spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")

    # Regardless of the machine, I should be able to delete any previous version of the download
    if os.path.exists(f"{clean_user_name}_download.csv"):
        os.remove(f"{clean_user_name}_download.csv")

    # The notebook version does this with Python subprocess.Popen()
    # That won't work with dbconnect, because it's a non-spark call,
    # and will run on the local developer machine.
    # Fortunately, dbutils (which resolves to the Databricks cluster) can work on local files using "file:/"
    dbutils = DBUtils(p_spark)
    dbutils.fs.rm(f"file:/databricks/driver/{local_data_path}", True)  # True means recurse

    # Delete DBFS directories that may be present
    dbutils.fs.rm(f"dbfs:/FileStore/mlwithsql/{dbfs_data_path}", True)  # True means recurse

    # Go home
    return 'Workshop cleanup is complete'


# This allows you to run cleanup standalone.
# Enter your user name as the first argument in the command line
# E.g., python cleanup.py Bill
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

    if args.run_cleanup_at_eoj.lower() == 'y':
        res = cleanup(args.user_name, spark)
        print(res)
    else:
        print("Cannot run cleanup unless command-line argument --run_cleanup_at_eoj is set to y")
