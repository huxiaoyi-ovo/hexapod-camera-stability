"""Synthetic toy sanity check ONLY, not evidence of camera-stabilization gains."""
import numpy as np
from controller import Gait, Camera, TrackedScene, Candidate, Settings, select_gait, image_interaction_matrix


def test_compensation_can_reverse_raw_motion_ranking():
    """Different gait candidates can reverse raw and post-compensation rankings.

    The synthetic signals are constructed by hand. This checks code mechanics,
    not whether real gait-induced vibrations exhibit this property.
    """
    camera = Camera(640., 640., 320., 240., 0.01, 1/30)
    scene = TrackedScene(
        uv=np.array([[240., 180.], [400., 180.], [240., 300.], [400., 300.]]),
        depth_m=np.array([1.5, 2., 1.5, 2.]),
        weights=np.ones(4),
    )
    A = Gait(1.2, 0.64, 0.05, 0.2)
    B = Gait(1.4, 0.62, 0.05, 0.3)
    d_A = np.zeros((6, 6)); d_A[:, 4] = 0.08  # raw disturbance larger
    d_B = np.zeros((6, 6)); d_B[:, 4] = 0.04  # raw disturbance smaller
    lim_A = np.full((6,6), 1e-8); lim_A[:,4] = 0.075
    lim_B = np.full((6,6), 1e-8); lim_B[:,4] = 0.003
    items = [
       Candidate(A, d_A, lim_A, 0.12, 0.05, 0.03, True),
       Candidate(B, d_B, lim_B, 0.12, 0.05, 0.03, True),
    ]
    setting = Settings(0.1, 0.02, 0.01, np.full(6, 1.),
                       regularization=1e-7, smoothing=1e-7, gait_change_penalty=0)
    L = image_interaction_matrix(camera, scene)
    assert np.linalg.norm(L @ d_A[0]) > np.linalg.norm(L @ d_B[0])
    result = select_gait(camera, scene, items, setting, previous_gait=B)
    assert result.gait == A
