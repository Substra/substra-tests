import os
import shutil
import subprocess

from ci.call import call
from ci.call import call_output
from ci.config import Config
from ci.config import GitConfig
from ci.config import Repository


def clone_repos(cfg: Config, source_dir: str) -> Config:
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)

    os.makedirs(source_dir)

    print(f"\n# Clone repos in {source_dir}")

    for repo in cfg.get_repos():
        repo = _clone_repository(cfg.git, repo, source_dir)

    print("\nCommit hashes:\n")
    for repo in cfg.get_repos():
        print(f"- {repo.repo_name:<30} {repo.commit}")
    print("\n")

    return cfg


def _clone_repository(cfg: GitConfig, repo: Repository, source_dir: str) -> Repository:
    extra_args = {}
    if cfg.use_token:
        extra_args["print_cmd"] = False

    url = _build_repo_url(cfg, repo)
    repo_dir = os.path.join(source_dir, repo.name)

    call(
        f'git clone -q --depth 1 {url} --branch "{repo.ref}" {repo_dir}',
        **extra_args,
    )

    if not repo.commit:
        try:
            commit = call_output(f"git --git-dir={repo_dir}/.git rev-parse origin/{repo.ref}")
        except subprocess.CalledProcessError:
            # It didn't work with a branch name. Try with a tag name.
            commit = call_output(f"git --git-dir={repo_dir}/.git rev-list {repo.ref}")
        repo.commit = commit

    return repo


def _build_repo_url(cfg: GitConfig, repo: Repository) -> str:
    url = ""
    if cfg.clone_method == "https":
        url = "https://{}github.com/{}"
        if cfg.use_token:
            url = url.format(f"{cfg.git_token}:x-oauth-basic@", repo.repo_name)
        else:
            url = url.format("", repo.repo_name)
    elif cfg.clone_method == "ssh":
        url = f"git@github.com:{repo.repo_name}"
    return url
