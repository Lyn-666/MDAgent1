"""The domain-general part of the agent: task, evidence, reasoning, decisions, records.

Nothing in this package knows about proteins, observables, simulation engines or any test system.
It imports only the Python standard library and other modules of this package
(tests/test_core_architecture.py checks this). Task-specific numbers come in through two
registries (registry.py):

- estimators turn data of one kind into an Assessment (evidence.py) with their own success
  criteria; there is no universal convergence threshold here;
- tools run a simulation method and return data for the estimators.

The decision loop (loop.py) is: observe -> assess -> hypothesize (reasoner.py) -> propose and
select an action under permissions and budget (controller.py) -> record (ledger.py) -> execute.
See docs/architecture.md.
"""
