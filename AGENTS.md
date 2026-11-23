# Project Coding Rules (Python Research Projects)

This file defines the coding standards for this project. Follow these rules when
creating, changing, reviewing, or refactoring code.

Based on the iHuman Lab Research Manual, with additional research-software
safeguards marked **Research addition**.

## 1. Readability and naming

- Prioritize clear code over clever or compressed code.
- Use descriptive, precise names for variables, functions, modules, and classes.
- Use `snake_case` for variables/functions, `PascalCase` for classes, and
  `UPPER_SNAKE_CASE` for constants.
- Use 4-space indentation and consistent PEP 8 style. Follow the project's
  Ruff configuration (the manual mentions 79-character lines as a baseline).
- Remove duplicate, unused, and commented-out code when it is safe to do so.

## 2. Small, focused functions

- Each function should have one clear responsibility.
- Give functions explicit inputs and useful return values.
- Reduce deep nesting with guard clauses and early returns when appropriate.
- Do not mix data loading, analysis, plotting, and output writing inside one
  large function. A short orchestration function may call separate steps.
- Avoid unnecessary abstractions. Prefer simple functions over elaborate
  classes when no class is needed.

## 3. Decoupling and modules

- Keep data I/O, preprocessing, feature engineering, computation/modeling,
  plotting, and reporting in separate modules when the project warrants it.
- Pass data, dependencies, and configuration into functions instead of relying
  on hidden global state or fixed file paths.
- Use modules and packages with clear imports; use `__init__.py` where needed.
- Keep `main.py` or the command-line entry point focused on orchestration.
- Apply interfaces/abstract classes only when interchangeable implementations
  make them useful.

## 4. Configuration and reproducibility

- Put tunable paths, parameters, thresholds, and experimental settings in a
  configuration file such as `config/config.yaml`.
- Maintain a shareable environment specification such as `environment.yml`.
- Never modify raw source data in place; store raw data in `data/raw/`,
  intermediate data in `data/interim/`, and outputs in `data/processed/`.
- Document how to reproduce the pipeline from raw inputs to outputs.
- **Research addition:** Record random seeds, data versions, software versions,
  analysis choices, and any exclusions that affect reported results.
- **Research addition:** Validate input column names, types, shapes, units,
  ranges, and missing values before expensive analysis.
- **Research addition:** For ML workflows, split data before fitting any
  preprocessing that learns from data; prevent train/test leakage.

## 5. Documentation

- Include Google-style docstrings for public functions, classes, and modules.
- State purpose, arguments, returns, and important errors or limitations.
- Use short comments for *why* a non-obvious decision was made, not a narration
  of obvious code.
- Keep `README.md` current: purpose, setup, folder layout, inputs, commands,
  outputs, and steps to reproduce results.
- Document known edge cases and limitations.

## 6. Error handling and logging

- Validate arguments and raise informative, specific exceptions.
- Catch only exceptions that the code can usefully handle.
- Never use a bare `except:` or silently `pass` on a failure.
- **Research addition:** Use logging for pipeline progress and warnings;
  reserve `print()` mainly for intentional user-facing command-line output.

## 7. Testing and quality checks

- Write tests under `tests/` with files named `test_*.py`.
- Use `pytest` for normal cases, boundary cases, invalid inputs, and key
  computations. Use fixtures and parameterized cases when helpful.
- Test pure computation independently from file I/O.
- Run formatting, linting, and tests before committing changes:

  ```bash
  ruff format .
  ruff check .
  pytest
  ```

- Use pre-commit hooks where practical. Add CI for shared repositories when
  useful. Tests should be expanded whenever behavior changes or bugs are fixed.

## 8. Git and safe modifications

- Track code and documentation with Git; make small commits with clear messages.
- Use branches and pull requests for collaboration when needed.
- **Research addition:** Do not commit passwords, API keys, private participant
  records, or large generated data files without explicit project policy.
- **Research addition:** When refactoring, preserve existing outputs, equations,
  units, file formats, experiment timings, and public interfaces unless the
  requested change explicitly requires a difference.
- **Research addition:** State what changed and what tests were run. Do not
  claim code is tested if tests were not executed.

## Recommended layout (adapt as needed)

```text
project/
├── AGENTS.md
├── README.md
├── environment.yml
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
└── tests/
```

## Review checklist

Before considering a change complete, verify:

- [ ] Names and module boundaries are clear.
- [ ] Functions are focused; duplicate work is minimized.
- [ ] Parameters are configurable; raw input data is preserved.
- [ ] Errors are handled explicitly; documentation is current.
- [ ] Tests cover the changed logic and important edge cases.
- [ ] Ruff checks and relevant pytest tests pass, or failures are reported.
- [ ] Results are reproducible and no unrelated behavior was changed.

## Source and attribution

Adapted from the iHuman Lab Research Manual (2026), CC BY 4.0:
https://ihuman-lab.github.io/research-manual/

Relevant pages:
- https://ihuman-lab.github.io/research-manual/coding_practices/clean_code.html
- https://ihuman-lab.github.io/research-manual/coding_practices/code_decoupling.html
- https://ihuman-lab.github.io/research-manual/coding_practices/code_testing.html
- https://ihuman-lab.github.io/research-manual/coding_practices/code_documentation.html
- https://ihuman-lab.github.io/research-manual/coding_practices/project_structure.html
- https://ihuman-lab.github.io/research-manual/coding_practices/development_setup.html
- https://ihuman-lab.github.io/research-manual/tutorials/python_tutorials/code_structure.html
