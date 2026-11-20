"""Tests for exact terrain selection, including streaming input."""
import pytest

from sim_split import split_trials


def test_split_preserves_both_partitions_from_one_pass_input():
    train = {'terrain_instance_id': 'train', 'pair_id': 'a', 'seed': 1}
    held_out = {'terrain_instance_id': 'held-out', 'pair_id': 'b', 'seed': 2}
    assert split_trials(iter((train, held_out)), {'held-out'}) == ([train], [held_out])


def test_split_rejects_partial_matches_for_requested_terrain_ids():
    rows = [
        {'terrain_instance_id': 'train', 'pair_id': 'a', 'seed': 1},
        {'terrain_instance_id': 'held-out', 'pair_id': 'b', 'seed': 2},
    ]
    with pytest.raises(ValueError, match='Unknown held-out terrain IDs.*typo'):
        split_trials(rows, {'held-out', 'typo'})
