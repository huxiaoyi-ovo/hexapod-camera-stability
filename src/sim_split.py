"""Leakage-resistant split by terrain-instance and seed, not per video frame."""
from __future__ import annotations
from collections import defaultdict

def split_trials(rows, heldout_terrain_ids):
    ids=set(heldout_terrain_ids)
    if not ids:
        raise ValueError('Explicit heldout terrain IDs required')
    train=[r for r in rows if r['terrain_instance_id'] not in ids]
    test=[r for r in rows if r['terrain_instance_id'] in ids]
    if not train or not test:
        raise ValueError('Train and heldout test must both be nonempty')
    # Off/on pairs may not be split between train and test; impossible given
    # terrain grouped split, but check to catch inconsistent pair labels.
    pairs=defaultdict(set)
    for partition,name in ((train,'train'),(test,'test')):
        for row in partition:
            pairs[(str(row['pair_id']),str(row['seed']))].add(name)
    crossing=[p for p,x in pairs.items() if len(x)>1]
    if crossing:
        raise ValueError(f'Paired trials leaked across partitions: {crossing[:3]}')
    return train,test
