# Working on this project

The user requires every completed project update to be committed and pushed
to the connected GitHub repository, unless the user explicitly requests local
work only. Use meaningful messages describing the actual change.

- Canonical checkout: the folder containing this file.
- Repository: https://github.com/TienMoi79/digits-rf-optimization
- Respect the code-only submission scope. Do not add reports or slides unless
  the user requests them.
- Keep the root layout: src/, notebooks/, results/, figures/, data/.
- Run checks appropriate to the change. For experiment/protocol changes, use
  `python -m unittest discover -s tests -v` and `python -m src.validation`.
- Commit code-generated results and figures when they change. Never invent
  results, backdate commits, or simulate a five-week development history.
- Exclude virtual environments, caches, credentials and unrelated files.
- Do not rerun the full HPO just to update docs or repository structure.
- Push completed changes using ordinary git push. Do not force-push or rewrite
  published history. Report the commit and synchronization result.

The Windows helper `update_github.ps1 -Message "Describe the real change"`
checks the project, creates a commit when needed, and pushes the current branch.
