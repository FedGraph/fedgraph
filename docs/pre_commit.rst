Pre-commit Workflow
===================

FedGraph uses `pre-commit <https://pre-commit.com/>`__ to run lightweight
repository checks before a commit is created. These checks are a code-quality
gate; they do not run the test suite in ``tests/``. Run tests separately with
``pytest`` when a change needs behavioral verification.

Setup and Trigger Points
------------------------

Install the development tools and the Git hook once in a checkout::

   python -m pip install -e '.[dev]'
   python -m pre_commit install --install-hooks

After that, ``git commit`` invokes ``.git/hooks/pre-commit``. The hook reads
``.pre-commit-config.yaml`` and runs its checks on staged files. To run the
same checks over the whole checkout, for example after updating hooks, use::

   python -m pre_commit run --all-files

The hook environments are cached independently by pre-commit, so the first run
downloads and creates them; later runs reuse the cache. Continuous integration
can use the same ``run --all-files`` command to apply the same gate to a clean
checkout.

Configuration
-------------

``.pre-commit-config.yaml`` is the entry point. Its
``minimum_pre_commit_version`` requires a runner new enough to interpret this
configuration. Each ``repos`` item pins a hook repository revision, which
makes developers and CI use the same checker versions.

The configured hooks are:

* ``check-merge-conflict``: rejects unresolved Git conflict markers such as
  ``<<<<<<<`` and ``>>>>>>>``.
* ``check-yaml``: parses YAML files and reports invalid syntax.
* ``end-of-file-fixer``: ensures text files end with exactly one newline.
* ``trailing-whitespace``: removes whitespace at the ends of lines.
* ``isort``: orders and groups Python imports. Its layout follows Black's
  formatting profile, configured in ``pyproject.toml``.
* ``mypy``: performs static type checking for ``fedgraph/`` only. It installs
  ``types-requests`` in its isolated hook environment because the package uses
  ``requests`` types.
* ``black``: applies one consistent Python code format.

The final three hooks may rewrite files. Review and stage those edits, then run
the commit again. The first four are repository-hygiene checks.

Related Project Files
---------------------

``pyproject.toml`` supplies tool-specific settings read by the hooks:

* ``[tool.isort] profile = "black"`` prevents isort and Black from disagreeing
  about import formatting.
* ``[tool.mypy] python_version = "3.9"`` checks against FedGraph's supported
  minimum Python version, while ``ignore_missing_imports = true`` avoids
  treating untyped optional third-party packages as project errors.

``setup.py`` defines the optional ``dev`` dependency group. Installing
``.[dev]`` provides the pre-commit runner, a compatible Mypy release, and
pytest for local development. Pre-commit itself creates isolated environments
from the exact revisions in ``.pre-commit-config.yaml``; ``setup.py`` makes the
runner and normal development tools available in the developer's environment.

Typing Changes in Core Source
-----------------------------

Updating Mypy exposed a small set of static-type ambiguities in the core
implementation. The following changes are annotations or narrowly scoped
``# type: ignore`` markers; they do not alter FedGraph's runtime behavior.

* ``fedgraph/server_class.py`` annotates aggregation statistics and handles the
  optional OpenFHE class import.
* ``fedgraph/trainer_class.py`` marks optional constructor metadata, handles
  the optional OpenFHE import, and documents model/context attributes that are
  initialized later in the trainer lifecycle.
* ``fedgraph/federated_methods.py`` marks optional OpenFHE availability and
  Ray's runtime-created ``Trainer.remote`` API, which static analysis cannot
  infer from the decorator.
* ``fedgraph/low_rank/server_lowrank.py`` annotates compression statistics.
* ``fedgraph/low_rank/trainer_lowrank.py`` documents model attributes inherited
  from the base trainer and initialized after construction.
* ``fedgraph/utils_nc.py`` gives batched node-index partitions their nested
  integer-list type.
* ``fedgraph/gnn_models.py`` documents that GIN's optional layers are fully
  configured before its training-time ``forward`` path executes.

Black and isort also reformatted other Python files. Those mechanical changes
are separate from the typing adjustments above.
