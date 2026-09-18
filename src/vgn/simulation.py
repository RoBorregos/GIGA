from pathlib import Path
import time

import numpy as np
import pybullet

from vgn.grasp import Grasp, Label
from vgn.perception import *
from vgn.utils import btsim, workspace_lines
from vgn.utils.transform import Rotation, Transform

# Scene scale, decoupled from the gripper. 0.45m is the practical ceiling for
# a 40^3 TSDF: 11.3mm per voxel.
WORKSPACE_SIZE = 0.45
# Fraction of the workspace the packed scene drops objects into.
PACKED_PATCH_FRAC = (0.22, 0.78)
from vgn.utils.misc import apply_noise, apply_translational_noise


class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, add_noise=False, sideview=False, save_dir=None, save_freq=8):
        assert scene in ["pile", "packed"]

        self.urdf_root = Path("data/urdfs")
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
            
        }.get(object_set, 1.0)
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq)
        self.gripper = Gripper(self.world)
        # Workspace size used to be 6 * finger_depth, which tied the scene
        # scale to a gripper dimension. With FRIDA's 106mm fingers that gave a
        # 0.62m cube, far too coarse for a 40^3 TSDF (15mm voxels, a 41mm apple
        # is under 3 voxels). Size is now its own parameter: 0.45m keeps the
        # voxel at 11.3mm while giving the 198mm-wide gripper body room to
        # approach. Measured: at 0.30m, 89% of pregrasp poses were already in
        # collision (62% palm-vs-table), so 98.7% of attempts were labelled
        # FAILURE before the fingers ever closed.
        self.size = WORKSPACE_SIZE
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height)
        else:
            raise ValueError("Invalid scene argument")

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        # Centre and scale with the workspace. These were hardcoded to 0.15 and
        # 0.6, i.e. to a 0.30m cube, while the cameras and the TSDF recentre on
        # size/2 -- so enlarging the workspace left the table sitting in a
        # corner, outside the camera's target. plane.obj is a 1m square, so
        # scale = size gives a table that exactly covers the workspace.
        pose = Transform(Rotation.identity(), [0.5 * self.size, 0.5 * self.size, height])
        self.world.load_urdf(urdf, pose, scale=2.0 * self.size)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            # Was uniform(0.08, 0.22): a 140mm patch, hardcoded to the old
            # 0.30m cube and narrower than the gripper body is wide (198mm),
            # so the hand could not fit between any two objects. Now a
            # proportional band centred on the workspace.
            lo, hi = PACKED_PATCH_FRAC
            x = self.rng.uniform(lo * self.size, hi * self.size)
            y = self.rng.uniform(lo * self.size, hi * self.size)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

    def acquire_tsdf(self, n, N=None, resolution=40):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.

        If N is None, the n viewpoints are equally distributed on circular trajectory.

        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, resolution)
        high_res_tsdf = TSDFVolume(self.size, 120)

        if self.sideview:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, self.size / 3])
            theta = np.pi / 3.0
        else:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
            theta = np.pi / 6.0
        r = 2.0 * self.size

        N = N if N else n
        if self.sideview:
            assert n == 1
            phi_list = [- np.pi / 2.0]
        else:
            phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

        timing = 0.0
        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)[1]

            # add noise
            depth_img = apply_noise(depth_img, self.add_noise)
            
            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower, self.upper)
        pc = high_res_tsdf.get_cloud()
        pc = pc.crop(bounding_box)

        return tsdf, pc, timing

    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read()
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()

        return result

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


class Gripper(object):
    """Simulated FRIDA gripper (was: simulated Panda hand).

    This gripper's joints run the opposite way round to the Panda hand this
    code was written for, and the two are not interchangeable:

        q = 0.000  ->  fingers 0.1060m apart (fully OPEN)
        q = 0.028  ->  fingers 0.0500m apart
        q = 0.056  ->  fingers -0.0060m apart (fully CLOSED)

    (measured in PyBullet from the finger link AABBs.) So opening width is
    0.106 - 2q, not 2q. The Panda mapping made execute_grasp's
    `self.gripper.move(0.0)` -- meant to close on the object -- fling this
    gripper wide open instead, and check_success then asks for
    `read() > 0.1 * max_opening_width` on a gripper reading zero. Every
    grasp attempt was therefore labelled a failure: a 960-grasp run came
    back 0/960 positive, and clean_balance_data.py, which drops
    len(negatives) - len(positives) rows, then emptied the dataset.

    max_opening_width is likewise the full opening, 0.106m. The 0.056 it
    used to hold is the per-finger travel from gripper.xacro's joint limit,
    which is half the stroke, not the aperture.

    finger_depth and
    T_body_tcp are reasoned starting values, NOT measured/derived from the
    mesh -- both also double as general scene-scale parameters elsewhere
    (place_table() sets table_height = finger_depth; scene cube size is
    6*finger_depth in reset_sim), not literal mechanical dimensions, so
    they were kept at Panda's original order of magnitude rather than
    naively set to this gripper's full 0.1m mesh finger length (which would
    make the scene cube ~0.6m, likely too large for a tabletop scenario).
    Validate empirically (render/inspect a generated scene, or watch a
    pilot run's grasp success rate) before trusting these at scale.
    """

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("data/urdfs/frida/hand.urdf")

        # Full aperture at q=0, and how far one finger travels to close it.
        self.max_opening_width = 0.106
        self.finger_travel = 0.056
        # Was 0.05, inherited from the Panda hand. FRIDA's fingers span
        # z in [0.014, 0.120] and the TCP sits at z = 0.017, so the fingertip
        # is 0.103m ahead of the TCP -- finger_depth is exactly that distance
        # in VGN's convention (sample_grasp_point places the object surface up
        # to finger_depth ahead of the TCP, i.e. right at the fingertip).
        # At 0.05 the sampler only ever reached halfway down the fingers.
        # T_body_tcp was swept before and made things worse at 0.070/0.100;
        # that sweep moved the right variable in the wrong place -- the TCP is
        # correct, finger_depth was the Panda leftover.
        self.finger_depth = 0.103
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.017])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered -- PyBullet doesn't enforce
        # the URDF <mimic> tag on "leftfinger", so this gear constraint is
        # still required, just pointed at this gripper's own link/joint
        # names instead of Panda's.
        self.world.add_constraint(
            self.body,
            self.body.links["left_finger"],
            self.body,
            self.body.links["right_finger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        # Start fully open, which for this gripper is q = 0.
        self.joint1 = self.body.joints["rightfinger"]
        self.joint1.set_position(0.0, kinematics=True)
        self.joint2 = self.body.joints["leftfinger"]
        self.joint2.set_position(0.0, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def _joint_position_for_width(self, width):
        """Joint angle that holds the fingers `width` apart.

        Each finger covers half the difference between the full aperture and
        the requested one, clamped to the joint's own travel so a commanded
        width outside the mechanism's range saturates instead of asking
        PyBullet for a position the joint limits will silently ignore.
        """
        q = 0.5 * (self.max_opening_width - width)
        return float(np.clip(q, 0.0, self.finger_travel))

    def move(self, width):
        q = self._joint_position_for_width(width)
        self.joint1.set_position(q)
        self.joint2.set_position(q)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        """Current opening width in meters (not the joint positions)."""
        closed_by = self.joint1.get_position() + self.joint2.get_position()
        return self.max_opening_width - closed_by


# Execution error the robust labels are meant to tolerate: about half a TSDF
# voxel (11.25mm at the 0.45m workspace), which is how far the detector can
# snap a grasp, plus a few degrees of arm/calibration error.
ROBUST_NOISE_POS = 0.005
ROBUST_NOISE_DEG = 5.0


def grasp_robustness(sim, ori, pos, trials, rng, noise_pos=ROBUST_NOISE_POS, noise_deg=ROBUST_NOISE_DEG):
    """Fraction of `trials` executions that succeed under small pose noise.

    Each trial restores the saved scene, jitters the position (gaussian,
    noise_pos metres per axis) and the orientation (random axis, gaussian
    angle of noise_deg degrees), and executes without removing the object.
    """
    ok = 0
    for _ in range(trials):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        r = ori * Rotation.from_rotvec(axis * np.radians(rng.normal(0.0, noise_deg)))
        p = pos + rng.normal(0.0, noise_pos, 3)
        sim.restore_state()
        outcome, _ = sim.execute_grasp(Grasp(Transform(r, p), sim.gripper.max_opening_width), remove=False)
        ok += outcome == Label.SUCCESS
    return ok / trials
