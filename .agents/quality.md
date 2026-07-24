# Code quality

The repository uses Ruff for linting and formatting, mypy in strict mode for
static type checking, and pytest for tests.

Before handing off a code change:

1. Run `make format` when Python source or tests changed.
2. Run `make check` and address every failure.
3. Do not use `git commit --no-verify`; fix the pre-commit hook's findings,
   review the changes it made, stage them, and commit again.

Keep changes focused. Preserve public behavior unless the task explicitly asks
to change it, and add or update tests for behavior changes.
