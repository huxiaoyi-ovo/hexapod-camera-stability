"""Synthetic plumbing tests; never present as Isaac Gym validation."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from pcr_sim_bridge import read_hexground_state, PcrSimTap, TrialIdentity
from pcr_sim_windows import (build_response_window, validate_pair_balance,
                            optical_twist_from_nominal_body)
from sim_split import split_trials


def mock_env(t=0.0, commanded_speed=.15):
    root=np.array([[0.,commanded_speed*t,.2,0.,0.,0.,1.,0.,commanded_speed,0.,0.,0.,0.]])
    rigid=np.zeros((1,9,13)); rigid[0,:,2]=.01; rigid[0,:,9]=.02
    gym=SimpleNamespace(refresh_rigid_body_state_tensor=lambda sim: None)
    return SimpleNamespace(root_states=root, dof_pos=np.zeros((1,18)),
        dof_vel=np.zeros((1,18)),torques=np.zeros((1,18)),
        torque_limits=np.full(18,27.),dof_pos_limits=np.tile([-3.,3.],(18,1)),
        dof_vel_limits=np.full(18,5.),contact_forces=np.ones((1,9,3)),
        feet_indices=np.arange(6),rb_states=rigid,gym=gym,sim=object())


def good_calibration(path,verified=True):
    payload={'extrinsic_verified':verified,
        'base_to_camera_position_m':[0.,0.,0.],
        'base_to_camera_quat_xyzw':[0.,0.,0.,1.],
        'camera':{'fx':400.,'fy':400.,'cx':320.,'cy':240.,
                  'exposure_s':.01,'frame_interval_s':1/30},
        'scene':{'uv':[[300.,240.],[350.,260.]],
                 'depth_m':[2.,3.], 'weights':[1.,1.]}}
    path.write_text(json.dumps(payload),encoding='utf-8')


def predecision():
    return dict(frequency_hz=1.5, duty_factor=.6,step_length_m=.08,
        swing_height_m=.06,touchdown_softness=.5,desired_speed_mps=.15,
        roughness_est_m=.02,slope_est_rad=.01,planned_stance_count=3,
        predicted_joint_headroom=.5, phase_rad=0.,
        roll_rate_rad_s=0.,pitch_rate_rad_s=0.,previous_frequency_hz=1.5,
        previous_duty_factor=.6,previous_touchdown_softness=.5,
        time_since_gait_change_s=1.0)


def write_trace(path,on=1):
    identity=TrialIdentity('rollout_on' if on else 'rollout_off',
                           'pair0','rough_2',2,bool(on),'gaitA')
    with PcrSimTap(path,identity) as tap:
        for t in np.arange(0,.62,.02):
            env=mock_env(t)
            tap.append(env,0,float(t),[0.,.15,0.],[True,True,True,False,False,False])
    return path


def test_pcr_state_extracts_actual_root_and_contact():
    state=read_hexground_state(mock_env(.4),0)
    assert state['root'].shape==(13,)
    assert state['foot_forces'].shape==(6,3)
    assert state['q'].shape==(18,)
    assert state['toe_positions'].shape==(6,3)
    assert np.allclose(state['toe_velocities'],[0.,0.,.02])
    assert state['min_position_margin']==pytest.approx(3.)
    assert state['max_dq_ratio']==0.


def test_pcr_state_no_silent_missing():
    with pytest.raises(AttributeError):read_hexground_state(SimpleNamespace(),0)


def test_nominal_body_to_camera_includes_lever_arm():
    actual=optical_twist_from_nominal_body([0.,.15,1.],[.2,0.,0.],[0.,0.,0.,1.])
    assert np.allclose(actual, [0.,.35,0.,0.,0.,1.])


def test_pcr_log_to_window_geometric_proxy(tmp_path):
    cal=tmp_path/'cal.json';good_calibration(cal)
    logfile=write_trace(tmp_path/'rollout.csv')
    row=build_response_window(logfile,predecision(),cal)
    assert row['controller_on']==1
    assert row['measured_jitter_px_frame']<1e-7  # nominal motion is smooth
    assert abs(row['measured_speed_mps']-.15)<1e-6
    assert row['n_camera_frames']>5


def test_missing_calibration_rejected(tmp_path):
    cal=tmp_path/'cal.json';good_calibration(cal,verified=False)
    logfile=write_trace(tmp_path/'rollout.csv')
    with pytest.raises(ValueError,match='calibrated'):
        build_response_window(logfile,predecision(),cal)


def test_off_on_pair_coverage(tmp_path):
    cal=tmp_path/'cal.json';good_calibration(cal)
    a=build_response_window(write_trace(tmp_path/'off.csv',0),predecision(),cal)
    b=build_response_window(write_trace(tmp_path/'on.csv',1),predecision(),cal)
    assert validate_pair_balance([a,b])==1
    with pytest.raises(ValueError,match='Missing off/on'):
        validate_pair_balance([a])


def test_group_split_keeps_pair_together():
    rows=[{'pair_id':'p1','seed':1,'terrain_instance_id':'train','controller_on':i} for i in (0,1)]
    rows.extend({'pair_id':'p2','seed':2,'terrain_instance_id':'heldout','controller_on':i} for i in (0,1))
    train,test=split_trials(rows,{'heldout'})
    assert len(train)==2 and len(test)==2
    assert set(r['pair_id'] for r in train)=={'p1'}


def test_hexground_action_conversion_no_silent_clip():
    from pcr_action_interface import targets_to_hexground_actions
    env=mock_env()
    env.default_dof_pos=np.zeros((1,18))
    env.dof_pos_limits=np.array([[-1.,1.]]*18)
    env.dof_vel_limits=np.full(18,5.)
    env.dt=.02
    env.cfg=SimpleNamespace(control=SimpleNamespace(action_scale=.5),
                           normalization=SimpleNamespace(clip_actions=1.))
    valid=np.full((1,18),.02)
    actions=targets_to_hexground_actions(env,valid)
    assert actions.shape==(1,18) and np.allclose(actions,.04)
    with pytest.raises(ValueError, match='velocity'):
        targets_to_hexground_actions(env,np.full((1,18),.2))
    prev=np.full((1,18),.15)
    assert targets_to_hexground_actions(env,np.full((1,18),.2),
                                        previous_reference=prev).shape==(1,18)
    with pytest.raises(ValueError, match='joint position'):
        targets_to_hexground_actions(env,np.full((1,18),1.05),check_velocity=False)
