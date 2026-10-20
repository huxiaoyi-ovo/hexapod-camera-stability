"""Process actual PCR-Net simulator root-state traces into paired model labels.

The output metric is a camera-geometric, static-scene residual *proxy*.
No rendered RGB or empirical image quality is claimed. All predictor features
must come from a predecision manifest, never future ground-truth contact.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from controller import Camera, TrackedScene
from trace_metrics import image_motion_metrics
from camera_pose_recorder import CameraPoseRecorder
from closed_loop_surrogate import INPUT_NAMES

ROOT_KEYS = tuple('root_'+x for x in ('px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz'))


def optical_twist_from_nominal_body(cmd_body, p_bc, q_bc):
    """Body +x right/+y front nominal velocity -> camera optical 6D twist.

    The body angular part is yaw around body +z; lever arm is respected.
    q_bc maps camera optical coordinates into body coordinates (xyzw).
    """
    v = np.asarray([cmd_body[0],cmd_body[1],0.],float)
    omega = np.asarray([0.,0.,cmd_body[2]],float)
    r_cb = Rotation.from_quat(q_bc).inv()
    return np.r_[r_cb.apply(v+np.cross(omega,p_bc)), r_cb.apply(omega)]


def camera_model_from_calibration(calibration_path):
    obj = json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    if obj.get('extrinsic_verified') is not True:
        raise ValueError('Camera extrinsic must be calibrated and marked verified')
    p = np.asarray(obj['base_to_camera_position_m'], float)
    q = np.asarray(obj['base_to_camera_quat_xyzw'], float)
    if p.shape != (3,) or q.shape != (4,) or abs(np.linalg.norm(q)-1) > .01:
        raise ValueError('Invalid camera extrinsic')
    cam = Camera(**obj['camera'])
    dat = obj['scene']
    scene = TrackedScene(uv=np.asarray(dat['uv'],float),
                         depth_m=np.asarray(dat['depth_m'],float),
                         weights=np.asarray(dat['weights'],float))
    scene.validate()
    return p, q, cam, scene


def read_pcr_rollout(path):
    with open(path,newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        required=set(ROOT_KEYS).union({'trial_id','pair_id','terrain_instance_id','seed',
                                       'controller_on','gait_id','time_s',
                                       'cmd_vx_body_mps','cmd_vy_body_mps','cmd_wz_body_rad_s'})
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing PCR trial CSV columns')
        rows=list(reader)
    if len(rows)<3:
        raise ValueError('Need at least 3 simulator samples')
    for col in ('trial_id','pair_id','terrain_instance_id','seed','controller_on','gait_id'):
        if len({r[col] for r in rows})!=1:
            raise ValueError('Metadata inconsistent within one rollout')
    return rows


def build_response_window(log_path, predecision, calibration_path, window_start_s=None,
                          window_end_s=None):
    """Build ONE outcome window. Predecision dict must be externally recorded
    *before* executing the trial; it may not include future contact/velocity.
    """
    rows=read_pcr_rollout(log_path)
    start=float(rows[0]['time_s']) if window_start_s is None else float(window_start_s)
    stop=float(rows[-1]['time_s']) if window_end_s is None else float(window_end_s)
    rows=[r for r in rows if start<=float(r['time_s'])<=stop]
    if len(rows)<3 or float(rows[-1]['time_s'])-float(rows[0]['time_s'])<.08:
        raise ValueError('Window has insufficient duration/samples')
    t=np.array([float(r['time_s']) for r in rows]);
    if np.any(np.diff(t)<=0):
        raise ValueError('Nonmonotonic timestamps')
    p_bc,q_bc,cam,scene=camera_model_from_calibration(calibration_path)
    recorder=CameraPoseRecorder(p_bc,q_bc)
    root=np.array([[float(r[k]) for k in ROOT_KEYS] for r in rows])
    cmd=np.array([[float(r[k]) for k in ('cmd_vx_body_mps','cmd_vy_body_mps','cmd_wz_body_rad_s')] for r in rows])
    for j in range(len(rows)):
        recorder.append(t[j], root[j,0:3],root[j,3:7],
                        optical_twist_from_nominal_body(cmd[j],p_bc,q_bc))
    # Derivative uses endpoint camera frame consistently with CameraPoseRecorder.
    cam_twist=[]
    for j,(time,position,rot,nominal) in enumerate(recorder.samples):
        a,b=(0,1) if j==0 else (j-1,j)
        t0,p0,r0,_=recorder.samples[a];t1,p1,r1,_=recorder.samples[b]
        dt=t1-t0
        v=rot.inv().apply((p1-p0)/dt)
        w=-(r1.inv()*r0).as_rotvec()/dt
        cam_twist.append(np.r_[v,w])
    residual=np.stack(cam_twist)-np.stack([s[3] for s in recorder.samples])
    metrics=image_motion_metrics(t,residual,cam,scene)
    # Signed progress along initially commanded world-frame direction;
    # not absolute path length, which would reward slipping sideways.
    body_dir=np.array([cmd[0,0],cmd[0,1],0.])
    if np.linalg.norm(body_dir[:2])<1e-8:
        raise ValueError('This first-stage protocol requires forward translation')
    desired_world=Rotation.from_quat(root[0,3:7]).apply(body_dir)[:2]
    desired_world/=np.linalg.norm(desired_world)
    actual_progress=float((root[-1,0:2]-root[0,0:2])@desired_world)
    actual_speed=max(0.,actual_progress/float(t[-1]-t[0]))
    feature_keys=set(INPUT_NAMES)
    missing=feature_keys.difference(predecision)
    if missing:
        raise ValueError(f'Missing predecision feature fields: {sorted(missing)}')
    # Never populate predecision features from the future state trace.
    meta=rows[0]
    out={k:predecision[k] for k in INPUT_NAMES}
    out.update(pair_id=meta['pair_id'],terrain_instance_id=meta['terrain_instance_id'],
               seed=int(meta['seed']),trial_id=meta['trial_id'],gait_id=meta['gait_id'],
               controller_on=int(meta['controller_on']),
               measured_jitter_px_frame=metrics.jitter_px_rms_per_frame,
               measured_speed_mps=actual_speed,
               n_camera_frames=metrics.n_frames,duration_s=metrics.duration_s,
               proxy_exposure_trace_px=metrics.exposure_trace_px_rms)
    return out


def write_response_windows(path,rows):
    rows=list(rows)
    if not rows:
        raise ValueError('No windows to write')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(key for r in rows for key in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader();writer.writerows(rows)
    return path


def validate_pair_balance(rows):
    seen={}
    for r in rows:
        key=(r['terrain_instance_id'],r['seed'],r['pair_id'],r['gait_id'])
        seen.setdefault(key,set()).add(int(r['controller_on']))
    missing=[k for k,v in seen.items() if v!={0,1}]
    if missing:
        raise ValueError(f'Missing off/on trial for {len(missing)} pairs')
    return len(seen)
