"""
CLI helper for the code review agent.

Two modes:
  1. Default: call the GET /review endpoint exposed by code_review_agent.
  2. --use-webhook: send the legacy GitHub-style webhook payload to /webhook.

Example:
    python client.py --repo-url https://github.com/huggingface/accelerate --pr-number 3321
"""

from __future__ import annotations

import argparse
from typing import Tuple
from urllib.parse import urlparse

import requests


def _render_value(value, indent: int) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        for key, val in value.items():
            print(f"{pad}{key}:")
            _render_value(val, indent + 2)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            print(f"{pad}- [{index}]")
            _render_value(item, indent + 2)
    elif isinstance(value, str):
        if "\n" in value:
            for line in value.splitlines():
                print(f"{pad}{line}")
        else:
            print(f"{pad}{value}")
    else:
        print(f"{pad}{value}")


def parse_repo(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Repository URL must start with http:// or https://")
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError("Repository URL must include owner and repo name")
    return path_parts[0], path_parts[1]


def build_payload(repo_url: str, pr_number: int, comment_url: str, comment_body: str) -> dict:
    owner, repo = parse_repo(repo_url)
    pull_api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    return {
        "repository": {"full_name": f"{owner}/{repo}"},
        "comment": {
            "body": comment_body,
            "pull_request_url": pull_api_url,
            "url": comment_url,
        },
    }


def _print_json(response: requests.Response) -> None:
    try:
        data = response.json()
    except ValueError:
        print("Response:", response.text)
    else:
        print("Response:")
        _render_value(data, 2)


def trigger_review_get(agent_endpoint: str, repo_url: str, pr_number: int) -> None:
    params = {"repo_url": repo_url, "pr_number": pr_number}
    response = requests.get(agent_endpoint, params=params, timeout=60*10,verify=False)
    print(f"Status code: {response.status_code}")
    #_print_json(response)


def trigger_review_webhook(
    agent_endpoint: str,
    repo_url: str,
    pr_number: int,
    comment_body: str,
    comment_url: str,
) -> None:
    payload = build_payload(repo_url, pr_number, comment_url, comment_body)
    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": "pull_request_review_comment",
        "User-Agent": "code-review-client/1.0",
    }

    response = requests.post(agent_endpoint, json=payload, headers=headers, timeout=60)
    print(f"Status code: {response.status_code}")
    _print_json(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger a review on the code_review_agent webhook.")
    parser.add_argument(
        "--agent-endpoint",
        default="http://127.0.0.1:8860/review",
        help="Review endpoint exposed by code_review_agent.",
    )
    parser.add_argument(
        "--repo-url",
        required=True,
        help="Full GitHub repository URL (e.g. https://github.com/org/repo).",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        required=True,
        help="Pull request number to review.",
    )
    parser.add_argument(
        "--comment-body",
        default="@code_review Please review this pull request.",
        help="Body content for the synthetic review comment (webhook mode only).",
    )
    parser.add_argument(
        "--comment-url",
        default="http://127.0.0.1:7860/comment-callback",
        help="Callback URL where the agent will attempt to post its review (webhook mode only).",
    )
    parser.add_argument(
        "--use-webhook",
        action="store_true",
        help="Send the legacy webhook payload to POST /webhook instead of calling GET /review.",
    )
    args = parser.parse_args()

    if args.use_webhook:
        trigger_review_webhook(
            agent_endpoint=args.agent_endpoint,
            repo_url=args.repo_url,
            pr_number=args.pr_number,
            comment_body=args.comment_body,
            comment_url=args.comment_url,
        )
    else:
        trigger_review_get(
            agent_endpoint=args.agent_endpoint,
            repo_url=args.repo_url,
            pr_number=args.pr_number,
        )


if __name__ == "__main__":
    main()
