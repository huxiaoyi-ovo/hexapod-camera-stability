"""Off-simulator checks of the actual PCR URDF chain *assumptions*.

These tests do not run PhysX or certify stable locomotion.
"""
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pcr_urdf_ik import HexUrdfIK, LEG_ORDER, JOINTS
from pcr_trial_controller import PcrSingleTrialController, FixedPoseCompensator
from tripod_gait import TripodConfig, ContinuousPhaseClock, foot_targets, foot_targets_by_phase


def example_urdf(tmp_path, bad_axis=False):
    robot = ET.Element('robot', name='fixture_hex')
    for leg in LEG_ORDER:
        x = -0.1 if leg.startswith('l') else 0.1
        y = {'f': 0.22, 'm': 0.0, 'b': -0.22}[leg[1]]
        origins = [(f'{x} {y} 0', '0 0 3.141592653589793' if leg.startswith('l') else '0 0 0'),
                   ('0.072 0 0','1.5707963267948966 0 0'),
                   ('0.13 0 0','0 0 0'),('0.1855 0 0','0 0 0')]
        for i, joint in enumerate(JOINTS+('toe',)):
            j = ET.SubElement(robot, 'joint', name=f'j_{leg}_{joint}',
                              type='fixed' if joint=='toe' else 'revolute')
            ET.SubElement(j,'origin', xyz=origins[i][0],rpy=origins[i][1])
            ET.SubElement(j,'parent',link='body' if i==0 else f'l_{leg}_{(JOINTS+('toe',))[i-1]}')
            ET.SubElement(j,'child',link=f'l_{leg}_{joint}')
            if joint!='toe':
                ET.SubElement(j,'axis',xyz='1 0 0' if bad_axis else '0 0 1')
                ET.SubElement(j,'limit',lower='-2.8',upper='2.8',effort='27',velocity='5')
    path=tmp_path/'hex_ground.urdf'
    ET.ElementTree(robot).write(path)
    return path


def poses():
    q = np.tile(np.array([0., 0.67, -2.2]), (6,1))
    q[[0,4],0]=0.5
    q[[1,3],0]=-0.5
    return q


def fake_env(ik):
    names=list(ik.canonical_dof_names[::-1])  # ensure no hardcoded action ordering
    q_default=ik.to_sim_order(poses(),names)
    return SimpleNamespace(
        num_envs=1, dof_names=names,
        default_dof_pos=q_default.reshape(1,18),
        dof_pos=q_default.reshape(1,18).copy(),
        dof_pos_limits=np.tile([-3.,3.],(18,1)),
        dof_vel_limits=np.full(18,5.), dt=.02,
        cfg=SimpleNamespace(control=SimpleNamespace(action_scale=.5),
                            normalization=SimpleNamespace(clip_actions=100.)),
        root_states=np.array([[0.,0.,.1,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]]))


def test_urdf_geometry_and_fk_ik_roundtrip(tmp_path):
    kin=HexUrdfIK.from_urdf(example_urdf(tmp_path))
    assert kin.l3==pytest.approx(0.1855)  # not the outdated 0.17 config hint
    q=poses()
    toe=kin.fk(q)
    assert toe.shape==(6,3)
    np.testing.assert_allclose(kin.ik(toe,q_reference=q),q,atol=1e-10)
    assert np.all(toe[:,2]<0)


def test_wrong_axis_rejected(tmp_path):
    with pytest.raises(ValueError,match='axes'):
        HexUrdfIK.from_urdf(example_urdf(tmp_path,bad_axis=True))


def test_ik_rejects_unreachable_target(tmp_path):
    kin=HexUrdfIK.from_urdf(example_urdf(tmp_path))
    toes=kin.fk(poses()); toes[0,0]+=0.9
    with pytest.raises(ValueError,match='reach'):
        kin.ik(toes)


def test_noncanonical_dof_map_permutation(tmp_path):
    kin=HexUrdfIK.from_urdf(example_urdf(tmp_path))
    names=list(kin.canonical_dof_names[::-1])
    q=poses(); ordered=kin.to_sim_order(q,names)
    np.testing.assert_allclose(kin.to_canonical(ordered,names),q)
    with pytest.raises(ValueError,match='DOF'):
        kin.to_sim_order(q,names[:-1]+['j_foo'])


def test_controller_emits_valid_first_step_with_realistic_geometry(tmp_path):
    kin=HexUrdfIK.from_urdf(example_urdf(tmp_path)); env=fake_env(kin)
    ctrl=PcrSingleTrialController(env,kin,TripodConfig(1.1,.58,.025,.5),compensate=False)
    actions,stance=ctrl.action(0.)
    assert actions.shape==(1,18)
    np.testing.assert_allclose(actions,0,atol=1e-7)
    assert np.array_equal(stance,np.array([True,True,True,True,True,True]))
    assert ctrl.clock.cycles==pytest.approx(.022)
    actions,_=ctrl.action(.02)
    assert np.isfinite(actions).all()


def test_bounded_compensation_changes_foot_commands(tmp_path):
    kin=HexUrdfIK.from_urdf(example_urdf(tmp_path)); env=fake_env(kin)
    comp=FixedPoseCompensator()
    feet=kin.fk(poses())
    q=Rotation.from_euler('xyz',[.08,-.05,0]).as_quat()
    adjusted=comp.foot_reference(feet,q,.11,.10)
    assert not np.allclose(adjusted,feet)
    assert np.max(np.linalg.norm(adjusted-feet,axis=1))<.03


def test_continuous_phase_across_frequency_switch(tmp_path):
    kin=HexUrdfIK.from_urdf(example_urdf(tmp_path)); feet=kin.fk(poses())
    clock=ContinuousPhaseClock(.35)
    before=foot_targets_by_phase(clock.cycles,feet,0,.08,0,TripodConfig(1,.58,.025,.5))[0]
    clock.advance(.02,2.0)
    after=foot_targets_by_phase(clock.cycles,feet,0,.08,0,TripodConfig(2,.58,.025,.5))[0]
    assert np.linalg.norm(after-before)<.1
    assert clock.cycles==pytest.approx(.39)
    # Old API stays backward compatible for fixed-frequency gait.
    np.testing.assert_allclose(foot_targets(.35,feet,0,.08,0,TripodConfig(1,.58,.025,.5))[0],before)


def test_canonical_toe_handle_reordering(tmp_path):
    from pcr_sim_bridge import canonical_toe_permutation
    from types import SimpleNamespace
    class MockGym:
        def find_actor_rigid_body_handle(self, env, actor, name):
            return {f'l_{leg}_toe': i*3 + 1 for i, leg in enumerate(LEG_ORDER)}[name]
    handles=[13,1,16,4,10,7]
    env=SimpleNamespace(gym=MockGym(),envs=['env0'],actor_handles=['actor0'],feet_indices=np.array(handles))
    permutation=canonical_toe_permutation(env)
    assert permutation.tolist()==[1,3,5,4,0,2]