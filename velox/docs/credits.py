import os
import subprocess
import argparse
import json
from pathlib import Path


def is_git_root(path):
    """Check if a directory is a Git root."""
    git_file = Path(os.path.join(path, ".git"))
    return git_file.exists()


def list_contributors(velox_root, since, until):
    # Change directory to Velox root.
    os.chdir(velox_root)
    print("Velox Path: " + str(velox_root) + ", range:[" + since + " - " + until + "]")

    # Unpack mailmap
    try:
        subprocess.run(
            ["base64", "-D", "-i", "velox/docs/mailmap_base64", "-o", "./.mailmap"]
        )
    except subprocess.CalledProcessError as e:
        print("Error:", e)

    # Get a list of contributors using git log
    try:
        contributors = subprocess.check_output(
            ["git", "shortlog", "-se", "--since", since, "--until", until], text=True
        )
    except subprocess.CalledProcessError as e:
        print("Error:", e)

    # Load affiliations map
    affiliateMap = json.load(open("velox/docs/affiliations_map.txt"))

    # Output sample: " 1  John <john@abc.com>"
    # Format contributor affiliation from the output
    unknownAffiliations = []
    for line in contributors.splitlines():
        start = line.find("<") + 1
        end = line.find(">")
        affiliation = line[start:end]
        # Get affiliation from email if present
        if "@" in affiliation:
            domain = affiliation.split("@")[1]
            if domain in affiliateMap:
                affiliation = affiliateMap.get(domain)
            else:
                unknownAffiliations.append(affiliation)
                affiliation = ""
        if affiliation != "":
            print(line[0 : start - 2], "-", affiliation)
        else:
            print(line[0 : start - 2])

    print("Unknown affiliations found: ", unknownAffiliations)
    # Remove .mailmap
    try:
        subprocess.run(["rm", "./.mailmap"])
    except subprocess.CalledProcessError as e:
        print("Error:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("since")
    parser.add_argument("until")
    parser.add_argument("--path", help="Velox root directory")
    args = parser.parse_args()

    velox_git_root = Path.cwd()
    if args.path:
        velox_git_root = args.path

    if not is_git_root(velox_git_root):
        print("Invalid Velox git root path", velox_git_root)
        exit()

    list_contributors(velox_git_root, args.since, args.until)
