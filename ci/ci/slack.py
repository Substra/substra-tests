import requests
import json
from typing import Optional


class SlackSendError(Exception):
    pass


def _format_markdown(title: str, markdown_text: str, markdown_context_text: Optional[str]) -> str:
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title,
            },
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": markdown_text}},
    ]
    if markdown_context_text:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": markdown_context_text}]})
    return json.dumps({"blocks": blocks})


def send_message(hook_url: str, title: str, markdown_text: str, markdown_context_text: Optional[str] = None) -> None:
    response = requests.post(
        hook_url,
        data=_format_markdown(title, markdown_text, markdown_context_text),
        headers={"Content-Type": "application/json"},
    )
    if response.status_code // 100 != 2:
        raise SlackSendError(f"{response.status_code} {response.text}")
