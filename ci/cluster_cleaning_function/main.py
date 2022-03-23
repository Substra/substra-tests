import base64
import json
import re
import traceback
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pprint import pprint
from typing import Dict

import requests
from google.cloud.container_v1 import ClusterManagerClient
from googleapiclient import discovery

ZONE = "us-east1-b"
PROJECT_ID = "connect-314908"
CLUSTER_PREFIX = "connect-tests"
MAX_HOURS = 24


def main(event, context):
    # Handle pub/sub message
    if event is not None and "data" in event:
        print(event)
        pubsub_message = base64.b64decode(event["data"]).decode("utf-8")
        print(pubsub_message)

    try:
        delete_old_clusters()
    except Exception:
        print("Failed to delete old clusters")
        traceback.print_exc()

    try:
        delete_orphaned_disks()
    except Exception:
        print("Failed to delete orphaned disks")
        traceback.print_exc()


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


def delete_orphaned_disks():
    """Delete all the Google Compute Disks which are not attached to a VM."""
    now = datetime.now(timezone.utc)
    disks_to_delete = []

    service = discovery.build("compute", "v1", cache_discovery=False)
    req = service.disks().list(project=PROJECT_ID, zone=ZONE)

    while req is not None:
        res = req.execute()

        for disk in res.get("items", []):
            if should_delete_disk(disk, now):
                disks_to_delete.append(disk.get("name"))

        req = service.disks().list_next(previous_request=req, previous_response=res)

    for disk_name in disks_to_delete:
        deletion_request = service.disks().delete(project=PROJECT_ID, zone=ZONE, disk=disk_name)
        deletion_response = deletion_request.execute()

        if deletion_response.get("error"):
            pprint(deletion_response["error"])
        else:
            print(f"Successfully deleted {disk_name}")


def should_delete_disk(disk: Dict, now: datetime) -> bool:
    # The disk must correspond to a PVC
    if not re.match("^gke-connect-tests-.*-pvc.*", disk["name"]):
        return False

    # The disk must not be attached to a VM
    if "users" in disk and disk["users"]:
        return False

    # The disk must have been created more than 30 min ago.
    # Note: this filter ensures we don't accidentally delete disks which haven't been attached to a VM yet. Disks are
    # typically attached within a few seconds, so 30 minutes is a safe time to wait.
    if now - datetime.fromisoformat(disk["creationTimestamp"]) < timedelta(minutes=30):
        return False

    return True


def delete_old_clusters():
    data = {
        "message": None,
        "success": [],
        "error": [],
        "skipped": [],
    }

    # List clusters
    cluster_manager_client = ClusterManagerClient()
    clusters_list = cluster_manager_client.list_clusters(project_id=PROJECT_ID, zone=ZONE)
    test_clusters = list(filter(lambda c: c.name.startswith(CLUSTER_PREFIX), clusters_list.clusters))
    list_clusters = "\n".join(list(map(lambda c: c.name, test_clusters)))

    list_result = (
        f'Found {len(test_clusters)} clusters with a name starting with "{CLUSTER_PREFIX}":\n' f"{list_clusters}"
    )
    print(list_result)

    if test_clusters:
        list_clusters = "\n\t\t- " + "\n\t\t- ".join(list(map(lambda c: f"*{c.name}*", test_clusters)))

    slack_list_result = (
        f"Found *{len(test_clusters)}* clusters with a name starting with *{CLUSTER_PREFIX}*" f"{list_clusters}"
    )
    data["message"] = slack_list_result

    for cluster in test_clusters:
        cluster_creation_time = cluster.create_time
        start = datetime.fromisoformat(cluster_creation_time).replace(tzinfo=None)
        age_hours = (datetime.utcnow() - start).total_seconds() // 3600

        print(f'"{cluster.name}" is {int(age_hours)} hours old. ', end="")
        if age_hours >= MAX_HOURS:
            print("Destroying.")
            try:
                cluster_manager_client.delete_cluster(project_id=PROJECT_ID, zone=ZONE, cluster_id=cluster.name)
            except Exception:
                data["error"].append(f"*{cluster.name}*")
            else:
                data["success"].append(f"*{cluster.name}*")
        else:
            print("Skipping.")
            data["skipped"].append(f"*{cluster.name}* ({int(age_hours)} hours old)")

    # Send notification each monday at 8 or if success/error
    if ((datetime.now().weekday() == 0) and ((datetime.now().hour % 24) == 8)) or data["success"] or data["error"]:

        response = requests.post(
            "https://hooks.slack.com/services/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",
            data=format_to_slack(data),
        )

        print(response)
