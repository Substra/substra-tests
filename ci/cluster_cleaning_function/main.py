import datetime
import base64
import json
import requests

from google.cloud.container_v1 import ClusterManagerClient

zone = "europe-west4-a"
project_id = "connect-314908"
cluster_prefix = "connect-tests"
max_hours = 24

text_from_type = {
    "success": "*:white_check_mark:* CLUSTER REMOVED:\n",
    "error": "*:x:* CLUSTER REMOVAL FAILURE:\n",
    "skipped": "*:warning:* CLUSTER SKIPPED\n",
}


def format_to_slack(data):
    res = {"blocks": []}
    res["blocks"].append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f":star-struck: {data['message']}"},
        }
    )

    for key_type, text in text_from_type.items():
        if data[key_type]:
            elts = "\n\t\t- " + "\n\t\t- ".join(data[key_type])
            res["blocks"].append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": text + elts}],
                }
            )

    return json.dumps(res)


def clean_cluster(event, context):

    # Handle pub/sub message
    if event is not None and "data" in event:
        print(event)
        pubsub_message = base64.b64decode(event["data"]).decode("utf-8")
        print(pubsub_message)

    data = {
        "message": None,
        "success": [],
        "error": [],
        "skipped": [],
    }

    # List clusters
    cluster_manager_client = ClusterManagerClient()
    clusters_list = cluster_manager_client.list_clusters(project_id=project_id, zone=zone)
    test_clusters = list(filter(lambda c: c.name.startswith(cluster_prefix), clusters_list.clusters))
    list_clusters = "\n".join(list(map(lambda c: c.name, test_clusters)))

    list_result = (
        f'Found {len(test_clusters)} clusters with a name starting with "{cluster_prefix}":\n' f"{list_clusters}"
    )
    print(list_result)

    if test_clusters:
        list_clusters = "\n\t\t- " + "\n\t\t- ".join(list(map(lambda c: f"*{c.name}*", test_clusters)))

    slack_list_result = (
        f"Found *{len(test_clusters)}* clusters with a name starting with *{cluster_prefix}*" f"{list_clusters}"
    )
    data["message"] = slack_list_result

    for cluster in test_clusters:
        cluster_creation_time = cluster.create_time
        start = datetime.datetime.fromisoformat(cluster_creation_time).replace(tzinfo=None)
        age_hours = (datetime.datetime.utcnow() - start).total_seconds() // 3600

        print(f'"{cluster.name}" is {int(age_hours)} hours old. ', end="")
        if age_hours >= max_hours:
            print("Destroying.")
            try:
                cluster_manager_client.delete_cluster(project_id=project_id, zone=zone, cluster_id=cluster.name)
            except Exception:
                data["error"].append(f"*{cluster.name}*")
            else:
                data["success"].append(f"*{cluster.name}*")
        else:
            print("Skipping.")
            data["skipped"].append(f"*{cluster.name}* ({int(age_hours)} hours old)")

    # Send notification each monday at 8 or if success/error
    if (
        ((datetime.datetime.now().weekday() == 0) and ((datetime.datetime.now().hour % 24) == 8)) or
        data["success"] or
        data["error"]
    ):

        response = requests.post(
            "https://hooks.slack.com/services/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",
            data=format_to_slack(data),
        )

        print(response)
