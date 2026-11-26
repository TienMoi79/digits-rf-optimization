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
- Follow PROGRESS.md for the remaining academic-week 8–11 checkpoints. Record
  the actual date, completed work, evidence and validation in that file on each
  meaningful update. Planned tasks must not be marked complete in advance.
- Existing code/results predate this checkpoint review. Do not describe them
  as newly implemented in a later week. Student understanding, instructor
  approval and LMS submission require the student's confirmation.
- After each completed part, provide a Markdown progress report in report/
  for the student to share with the instructor. Include new work, evidence,
  actual validation, remaining gaps and review questions. Default to .md;
  create Word/PDF reports or slides only when requested. Keep prior checkpoint reports available.
- Report folders: report/checkpoints/ for weekly reports, report/reviews/ for
  feedback and audits, report/updates/ for document/chart updates, and
  report/maintenance/ for repository housekeeping. Keep report/README.md as
  the index. Existing DOCX paths are protected by hashes; do not relocate them.
- README.md is the short entry point. Keep detailed technical guidance in
  docs/huong-dan-ky-thuat.md and folder navigation in docs/README.md.
- When moving documentation, update readers/writers and relative links;
  preserve generated numeric sections and frozen result/model/notebook bytes.
- Web demo source is in web/. Export data with scripts/export_web_data.py;
  verify with --check and node web/tests/check-model.mjs. Never alter frozen
  experiment results to improve the web presentation. Gallery samples are
  development-only; drawn digits are not a new accuracy benchmark. Preserve
  the Site ID in web/.openai/hosting.json and keep publication credentials out
  of Git. Sites publishing uses a separate checkout, not the project remote.
- Submission archives are generated from a clean committed tree by
  scripts/package_submission.py --output <path-outside-repo>. Keep archives,
  checksums and temporary extraction folders outside the checkout. Do not
  delete checkpoint logs, reproducibility runs or prior reports as "duplicates".
- On 2026-11-03 the user explicitly requested one Word report containing train,
  validation, test and baseline comparisons. Maintain that DOCX and its source
  builder/evidence as authorized; this supersedes the earlier .md-only scope
  for that report. Do not infer that slides or additional PDFs were requested.
- When the student provides instructor feedback, record the feedback and the
  corresponding fixes and validation. Never invent instructor feedback or
  mark a checkpoint approved without the student's confirmation.
- Check reported numbers with `python -m src.report_facts --check`. The marked
  numeric section in the week-8 report is generated from frozen logs/model;
  refresh it with --write instead of entering replacement values by hand.
  Distinguish a best individual trial from a mean over search seeds. Do not
  change experiment results to match a report revision or feedback.

The Windows helper `update_github.ps1 -Message "Describe the real change"`
checks the project, creates a commit when needed, and pushes the current branch.
