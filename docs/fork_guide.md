## Terminologies:
| Repo name | Description | Location |
| ----------- | ----------- | ----------- |
| Upstream repo | Group project repo | UBC public Github account |
| Your remote repo | Your forked repo from above | Your public Github account |
| Your local repo | Cloned from your remote repo | Your PC |


## Setup:
- **Step 1**: Fork the upstream repo to your remote repo.
![How to fork](https://i.ibb.co/fkc0sBP/howtofork.png)

- **Step 2**: Clone it from your remote repo to our laptop to make a local repo.

## Make changes:
<img src="https://i.ibb.co/W3CR5BR/push-flow.png" alt="push-flow" border="0" width="700px">

- **Step 1**: Create a branch (good practice) in your local repo, and switch to it

`git switch -c <your_branch>`

- **Step 2**: Make changes

- **Step 3**: Push from your local repo to your remote repo

`git push --set-upstream origin <your_branch>`

- **Step 4**: Go to upstream repo, create a Pull Request. You can either click on the "Compare & pull request" or go to "Pull requests" tab.
<img src="https://i.ibb.co/xDkzLLw/create-pr.png">

- **Step 5**: Chose which branch to merge and add a reviewer to your PR. This person will provide feedback and merge your PR.
<img src="https://i.ibb.co/cyYwXSd/pr-details.png" alt="pr-details" border="0">

Highlighted in red should be our group repo, and the branch should be main.

Highlighted in green should be your remote repo, and the branch should be <your_branch>

## Update changes from upstream repo:
<img src="https://i.ibb.co/02BKNBc/pull-flow.png" alt="pull-flow" border="0" width="650px">

- **Step 1**: Add the upstream repo as one of your remote repos for your local repo. Only do it once at the beginning.

`git remote add upstream <link_to_our_group_project>`

- **Step 2**: confirm it is added successfully

`git remote -v`

(It should have 2 links as upstream and 2 as remote)

- **Step 3**: pull updates from upstream repo

`git pull upstream main`

- **Step 4**: push updates to your remote repo

`git push`
