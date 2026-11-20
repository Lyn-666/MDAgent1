import math

from mdagent.cost import Cost, CostModel, of, of_run, reduction


def test_starting_a_replica_costs_something():
    """Ten short replicas and one long replica have the same aggregate time but not the same cost."""
    model = CostModel(equilibration_ns=0.6)
    many = of([10.0] * 10, model)
    one = of([100.0], model)
    assert many.aggregate_ns == one.aggregate_ns == 100.0
    assert many.compute_ns == 106.0 and one.compute_ns == 100.6
    assert many.compute_ns > one.compute_ns


def test_running_replicas_side_by_side_shortens_the_wait_not_the_cost():
    serial = of([10.0] * 4, CostModel(equilibration_ns=0.0, n_parallel=1))
    parallel = of([10.0] * 4, CostModel(equilibration_ns=0.0, n_parallel=4))
    assert serial.compute_ns == parallel.compute_ns == 40.0
    assert serial.wall_ns == 40.0 and parallel.wall_ns == 10.0


def test_uneven_replicas_are_spread_over_the_gpus():
    c = of([10.0, 6.0, 4.0], CostModel(equilibration_ns=0.0, n_parallel=2))
    assert c.wall_ns == 10.0           # 10 on one GPU, 6 + 4 on the other.


def test_gpu_hours_and_reduction():
    agent = of([20.0] * 3, CostModel(equilibration_ns=0.6))
    baseline = of([40.0] * 5, CostModel(equilibration_ns=0.6))
    assert math.isclose(agent.gpu_hours(600.0), 61.8 / 600 * 24)
    r = reduction(agent, baseline)
    assert 0.69 < r["compute"]["saved"] < 0.70
    assert r["compute"]["speedup"] > 3.2
    assert math.isnan(Cost(1, 1, 0, 1, 1).gpu_hours(0.0))


def test_cost_of_a_real_run_reads_run_json():
    state = {"rounds_ns": [1.0, 5.0, 5.0], "n_replicas": 4, "md": {"equilibration_ps": 600.0},
             "notes": {"rounds": [{"ns": 1.0, "n_replicas": 4, "seconds": 100.0},
                                  {"ns": 5.0, "n_replicas": 4, "seconds": 500.0}]},
             "enhanced": {"settings": {"n_replicas": 2}, "outcome": {"ns_per_replica": 5.0}}}
    c = of_run(state)
    assert c.aggregate_ns == 4 * 11.0 + 2 * 5.0        # Plain MD plus enhanced sampling.
    assert c.overhead_ns == 4 * 0.6                     # Enhanced sampling needs no new equilibration.
    assert c.measured_seconds == 600.0
