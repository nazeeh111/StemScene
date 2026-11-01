"""Analytic signal fixtures test metric correctness, not empirical separation quality."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = spec_from_file_location("evaluate_refiner", Path(__file__).parents[1] / "scripts/evaluate_refiner.py")
assert SPEC is not None and SPEC.loader is not None
EVALUATE = module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def signals():
    time = np.arange(8000) / 8000
    return 0.3 * np.sin(2*np.pi*337*time), 0.2 * np.sin(2*np.pi*719*time)


def test_wrong_gain_is_penalized_by_direct_sdr_but_not_si_sdr():
    reference, _ = signals()
    assert EVALUATE.sdr(reference, reference * 0.5) == pytest.approx(6.020599913, abs=1e-6)
    assert EVALUATE.si_sdr(reference, reference * 0.5) > 100
    assert EVALUATE.si_sdr(reference, np.zeros_like(reference)) == -120
    assert EVALUATE.sdr(np.zeros_like(reference), reference) is None


def test_perfect_source_edit_has_no_error_and_absence_uses_energy():
    a, b = signals()
    references = np.stack([a, np.zeros_like(a)])
    metadata = {"queries": ["a", "absent"], "presence_by_request": [True, False],
                "edit": {"start_seconds": 0.1, "end_seconds": 0.9, "gain": 0.25}}
    perfect = EVALUATE.metrics(a+b, references, references, 8000, metadata, 0.01)
    assert perfect["edit_nmse"] == 0
    assert perfect["outside_edit_max_delta"] == 0
    assert perfect["absence_energy_over_mixture"] == 0
    assert perfect["per_query"][1]["sdr_db"] is None
    contaminated = EVALUATE.metrics(a+b, np.stack([a+b, b]), references, 8000, metadata, 0.01)
    assert contaminated["edit_nmse"] > 0
    assert contaminated["retained_source_projection_nmse"] > 0
    assert contaminated["absence_false_extraction_rate"] == 1
    assert contaminated["excess_pairwise_attribution"] > 0


def test_confirmation_gate_requires_matching_frozen_decisions(tmp_path):
    with pytest.raises(ValueError, match="Confirmation requires"):
        EVALUATE.validate_confirmation_gate("confirmation", None, "checkpoint", "recipes")
    freeze = tmp_path / "decisions.json"
    freeze.write_text(json.dumps({"frozen": True, "selection_finished": True,
                                 "checkpoint_sha256": "checkpoint", "frozen_recipes_sha256": "recipes"}))
    assert EVALUATE.validate_confirmation_gate("confirmation", freeze, "checkpoint", "recipes")
    with pytest.raises(ValueError, match="does not match"):
        EVALUATE.validate_confirmation_gate("confirmation", freeze, "different", "recipes")


def test_record_identity_cannot_substitute_other_sources_or_presence():
    recipe = {"split": "selection", "inference": {"queries": ["rain", "dog"]},
              "evaluator_only": {"source_files": ["a.wav", "b.wav"], "presence_by_request": [True, False], "edit": {"gain": 0.25}}}
    metadata = {"recipe_id": "selection-000", "source_ids": ["b.wav", "a.wav"],
                "queries": ["rain", "dog"], "presence_by_request": [True, False], "edit": {"gain": 0.25}}
    EVALUATE.validate_recipe_identity(metadata, {"selection-000": recipe}, "selection")
    metadata["presence_by_request"] = [True, True]
    with pytest.raises(ValueError, match="differs from frozen recipe"):
        EVALUATE.validate_recipe_identity(metadata, {"selection-000": recipe}, "selection")


def test_wiener_preserves_a_residual_and_is_permutation_equivariant():
    a, b = signals()
    estimates = np.stack([a * 0.6, np.zeros_like(a)])
    result = EVALUATE.residual_wiener(a+b, estimates)
    reversed_result = EVALUATE.residual_wiener(a+b, estimates[::-1].copy())[::-1]
    np.testing.assert_allclose(result, reversed_result, atol=1e-12)
    assert np.mean((a+b-result.sum(axis=0))**2) > 1e-5
    np.testing.assert_array_equal(result[1], np.zeros_like(a))


def test_fuss_oracle_matches_only_present_queries_and_never_claims_absence():
    a, b = signals()
    references = np.stack([a, np.zeros_like(a), b])
    present = np.array([True, False, True])
    outputs = np.stack([b, np.zeros_like(a), a, -b])
    assigned = EVALUATE.oracle_fuss_assignment(outputs, references, present)
    np.testing.assert_array_equal(assigned, references)
    meta = {"queries": ["a", "absent", "b"], "presence_by_request": list(present),
            "edit": {"request_index": 1, "start_seconds": 0.1, "end_seconds": 0.9, "gain": 0.25}}
    result = EVALUATE.metrics(a+b, assigned, references, 8000, meta, 0.01, present_only_oracle=True)
    assert result["target_sdr_db"] == 120
    assert result["waveform_nmse"] == 0
    assert result["absence_energy_over_mixture"] is None
    assert result["absence_false_extraction_rate"] is None
    assert result["edit_nmse"] is None
    assert result["per_query"][1]["extracted_rms"] is None


def test_all_request_metric_catches_damage_hidden_by_first_query():
    a, b = signals()
    refs = np.stack([a, b])
    estimate = np.stack([a, a+b])
    meta = {"queries": ["a", "b"], "presence_by_request": [True, True],
            "edit": {"request_index": 0, "start_seconds": .1, "end_seconds": .9, "gain": .25}}
    result = EVALUATE.metrics(a+b, estimate, refs, 8000, meta, .01)
    assert result["edit_nmse"] == 0
    assert result["all_request_edit_nmse"] > 0
    assert result["all_request_edit_errors"][0] == 0
    assert result["all_request_edit_nmse"] == pytest.approx(result["all_request_edit_errors"][1]/2)
    perfect = EVALUATE.metrics(a+b, refs, refs, 8000, meta, .01)
    assert perfect["all_request_edit_nmse"] == 0
    assert perfect["simultaneous_edit_nmse"] == 0


def test_reference_conditioned_duplicates_and_target_error_split():
    a,b = signals()
    refs = np.stack([a,b])
    meta = {'queries':['a','b'],'presence_by_request':[True,True],
            'edit':{'request_index':0,'start_seconds':.1,'end_seconds':.9,'gain':.25}}
    perfect = EVALUATE.metrics(a+b,refs,refs,8000,meta,.01,all_sources=refs)
    assert perfect['source_duplicate_energy_over_mixture']==pytest.approx(0,abs=1e-14)
    duplicated = EVALUATE.metrics(a+b,np.stack([a,a+b]),refs,8000,meta,.01,all_sources=refs)
    assert duplicated['source_duplicate_energy_over_mixture']>0
    assert duplicated['all_request_edit_nmse']==pytest.approx(duplicated['all_request_target_projection_edit_nmse']+duplicated['all_request_orthogonal_edit_nmse'],abs=1e-12)
    indistinguishable = EVALUATE.metrics(a+b,refs,refs,8000,meta,.01,all_sources=np.stack([a,a]))
    assert indistinguishable['source_duplicate_energy_over_mixture'] is None
