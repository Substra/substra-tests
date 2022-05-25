import json
from typing import Optional

import requests


class SlackSendError(Exception):
    pass


def _format_markdown(title: str, markdown_text: str, job_status: str, job_link: str, duration: Optional[int]) -> str:
    title = f"*{title}: <{job_link}|{job_status}>*"
    if duration is not None:
        title += f" in {_get_friendly_duration(duration)}"
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": title}},
        {"type": "section", "text": {"type": "mrkdwn", "text": markdown_text}},
    ]
    return json.dumps({"blocks": blocks})


def send_message(
    hook_url: str, title: str, markdown_text: str, job_status: str, job_link: str, duration: Optional[int]
) -> None:
    response = requests.post(
        hook_url,
        data=_format_markdown(title, markdown_text, job_status, job_link, duration),
        headers={"Content-Type": "application/json"},
    )
    if response.status_code // 100 != 2:
        raise SlackSendError(f"{response.status_code} {response.text}")


def _get_friendly_duration(duration: int) -> str:
    minutes, seconds = divmod(duration, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
