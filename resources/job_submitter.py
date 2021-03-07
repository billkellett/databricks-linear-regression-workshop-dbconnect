# Create my linear regression job on an existing cluster

import json
import requests

DOMAIN = '<my-databricks-domain>'
TOKEN = '<my-personal-access-token>'
URL = f'https://{DOMAIN}/api/2.0/jobs/create'

response = requests.post(
    URL,
    headers={'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'},
    json={
        "name": "<my-job-name>",
        "existing_cluster_id": "<my-cluster-id>",
        "spark_python_task": {
            "python_file": "dbfs:/FileStore/python/lrwkshp/main.py",
            "parameters": [
                "--user_name", "<my-user-id-any-unique-string>", "--api_token", "<my-personal-access-token>",
                "--run_cleanup_at_eoj", "y", "--mlflow_experiment_id", "<my-experiment-id>"
            ]
        }
    }
)

# We can treat a json response just like a Python dictionary,
# and pluck out the data using the key that represents the level of the json structure we are interested in
# json.dumps() lets us define pretty printing
# Unlike the List Clusters request earlier, here we do not index into the response.
# We just pretty-print the whole json response.
if response.status_code == 200:
    pretty_response = json.dumps(response.json(), indent=4)
else:
    pretty_response = \
        f"Error when attempting to list contents of DBFS root: {response.json()['error_code']}: " \
        f"{response.json()['message']}"

print(pretty_response)
