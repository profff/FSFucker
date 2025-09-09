import subprocess
from git import Repo, InvalidGitRepositoryError
import os

#! python
def get_git_info():
    try:
        current_path = os.getcwd()
        while current_path:
            try:
                repo = Repo(current_path)
                break
            except InvalidGitRepositoryError:
                parent_path = os.path.dirname(current_path)
                if parent_path == current_path:  # Reached the root directory
                    raise InvalidGitRepositoryError("No valid git repository found")
                current_path = parent_path
        if repo.bare:
            return {"error": "Repository is bare"}
    except InvalidGitRepositoryError:
        return {"error": "Not a valid git repository"}
    except Exception as e:
        return {"error": str(e)}
    git_info = {}
    git_info['branch'] = repo.active_branch.name
    git_info['hash'] = repo.head.commit.hexsha
    git_info['tag'] = next((tag.name for tag in repo.tags if tag.commit == repo.head.commit), None)
    git_info['name'] = repo.remotes.origin.url if repo.remotes else None
    return git_info

def print_banner(title):
    git_info = get_git_info()
    print("=" * 50)
    print(f"{title}")
    print("=" * 50)
    if "error" in git_info:
        print(f"Git Info: {git_info['error']}")
    else:
        print(f"Branch: {git_info['branch']}")
        print(f"Commit Hash: {git_info['hash']}")
        print(f"Tag: {git_info['tag']}")
        print(f"Remote URL: {git_info['name']}")
        if git_info['tag']:
            try:
                tag_parts = git_info['tag'].split('_v')
                if len(tag_parts) == 2:
                    version_parts = tag_parts[1].split('.')
                    major_rev = version_parts[0]
                    minor_rev = version_parts[1] if len(version_parts) > 1 else "0"
                    print(f"Rev: {major_rev}.{minor_rev}")
            except Exception as e:
                print(f"Error decoding tag: {e}")
    print("=" * 50)
# -*- coding: utf-8 -*-
