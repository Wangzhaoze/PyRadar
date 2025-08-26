import math
import numpy as np
import mitsuba as mi
import trimesh
from typing import List

# -------------------- Mitsuba Variant Selection --------------------
def try_set_mitsuba_variant():
    """
    Try to set Mitsuba rendering variant, preferring GPU if available.
    """
    for v in ["cuda_rgb", "llvm_ad_rgb", "scalar_rgb"]:
        try:
            mi.set_variant(v)
            print(f"[Mitsuba] Using variant: {v}")
            return
        except Exception:
            pass
    raise RuntimeError("Failed to set Mitsuba variant. Please check Mitsuba installation.")

# -------------------- Uniform Cone Direction Sampling --------------------
def sample_directions_in_cone(n, cone_angle_deg, forward=np.array([0,0,-1], dtype=float)):
    """
    Uniformly sample n directions within a cone around the given forward vector.
    """
    forward = np.asarray(forward, dtype=float)
    forward /= np.linalg.norm(forward)
    cone = math.radians(cone_angle_deg)
    cos_max = math.cos(cone)
    u = np.random.rand(n)
    v = np.random.rand(n)
    cos_theta = (1 - u) + u * cos_max
    sin_theta = np.sqrt(np.maximum(0.0, 1 - cos_theta**2))
    phi = 2 * np.pi * v
    dirs_local = np.stack([sin_theta * np.cos(phi),
                           sin_theta * np.sin(phi),
                           cos_theta], axis=1)
    # Rotate local z=[0,0,1] to forward direction
    z = forward
    h = np.array([0,0,1.0])
    if np.allclose(z, h):
        R = np.eye(3)
    elif np.allclose(z, -h):
        R = np.diag([1,-1,-1])
    else:
        v_ = np.cross(h, z)
        s = np.linalg.norm(v_)
        c = np.dot(h, z)
        vx = np.array([[0, -v_[2], v_[1]],
                       [v_[2], 0, -v_[0]],
                       [-v_[1], v_[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return (dirs_local @ R.T).astype(np.float32)

# -------------------- Optics: Reflection, Refraction, Fresnel --------------------
def reflect(dir_in, normal):
    """
    Calculate the reflection direction given an incident direction and surface normal.
    Both vectors must be normalized.
    """
    d = dir_in / np.linalg.norm(dir_in)
    n = normal / np.linalg.norm(normal)
    return d - 2.0 * np.dot(d, n) * n

def refract(dir_in, normal, eta_i, eta_t):
    """
    Snell's law refraction: from medium eta_i to eta_t.
    Returns (success, dir_out). If total internal reflection occurs, success=False.
    """
    d = dir_in / np.linalg.norm(dir_in)
    n = normal / np.linalg.norm(normal)

    # Ensure n points against d (positive incident angle cosine)
    cosi = -np.dot(d, n)
    if cosi < 0:
        # From inside to outside, flip normal and swap indices
        n = -n
        cosi = -np.dot(d, n)
        eta_i, eta_t = eta_t, eta_i

    eta = eta_i / eta_t
    k = 1.0 - eta**2 * (1.0 - cosi**2)
    if k < 0.0:
        return False, None  # Total internal reflection
    cost = math.sqrt(k)
    t = eta * d + (eta * cosi - cost) * n
    t /= np.linalg.norm(t)
    return True, t

def fresnel_schlick(cos_theta, eta_i, eta_t):
    """
    Schlick's approximation for Fresnel reflectance.
    """
    r0 = ((eta_t - eta_i) / (eta_t + eta_i))**2
    return r0 + (1 - r0) * (1 - cos_theta)**5

# -------------------- Visualization: Draw Ray as Cylinder --------------------
def cylinder_between(p0, p1, radius, rgba):
    """
    Create a cylinder mesh between two points for ray visualization.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-9:
        return None
    vn = v / L
    # Default cylinder along z-axis, centered at origin
    cyl = trimesh.creation.cylinder(radius=radius, height=L, sections=12)
    # Align to direction vn
    T = trimesh.geometry.align_vectors([0,0,1], vn)
    cyl.apply_transform(T)
    # Translate to midpoint
    mid = (p0 + p1) * 0.5
    cyl.apply_translation(mid)
    cyl.visual.face_colors = rgba
    return cyl

def main_mitsuba(
    objects: List[dict],
    num_rays: int = 800,
    max_bounces: int = 6,
    tx_pos=(0.0, 1.5, 5.0),
    tx_dir=(0.0, 0.0, -0.25),
    cone_deg=60.0,
    rx_center=(0.0, 0.25, 0.0),
    rx_radius=0.5,
    air_eta=1.0,
    ray_radius=0.015
):
    """
    Main function for ray tracing simulation and visualization using Mitsuba and Trimesh.
    """
    try_set_mitsuba_variant()

    # ---- Build Mitsuba scene dictionary ----
    scene_dict = {"type": "scene"}
    scene_dict["receiver_sphere"] = {
        "type": "sphere",
        "radius": rx_radius,
        "center": rx_center,
        "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.0, 0.0, 1.0]}},
        "id": "receiver_sphere"
    }

    dielectric_ids = set()
    material_map = {}

    for obj in objects:
        obj_id = obj["id"]
        mat_type = obj.get("material", "conductor")
        eta = obj.get("eta", 1.5)

        # --- Material ---
        if mat_type == "conductor":
            bsdf = {"type": "roughconductor"}
        else:
            bsdf = {"type": "dielectric", "int_ior": eta}
            dielectric_ids.add(obj_id)

        # --- Transformation matrix ---
        if "to_world" not in obj:
            raise ValueError(f"Object {obj_id} missing to_world 4x4 matrix")
        T_np = np.array(obj["to_world"], dtype=float)
        if T_np.shape != (4, 4):
            raise ValueError(f"Object {obj_id} to_world must be a 4x4 matrix")

        T_mi = mi.ScalarTransform4f(T_np)

        shape_dict = {
            "type": "obj",
            "filename": obj["path"],
            "to_world": T_mi,
            "bsdf": bsdf,
            "id": obj_id,
            "face_normals": False
        }
        scene_dict[obj_id] = shape_dict
        material_map[obj_id] = (mat_type, eta)

    scene: mi.Scene = mi.load_dict(scene_dict)

    # ---- Sample emission directions ----
    dirs = sample_directions_in_cone(num_rays, cone_deg, forward=np.array(tx_dir, dtype=float))
    tx_pos_np = np.array(tx_pos, dtype=float)

    hit_flags, ray_paths = [], []
    rng = np.random.RandomState(42)
    eps, far_miss = 1e-12, 60.0

    for i in range(num_rays):
        d0 = np.array(dirs[i], dtype=float)
        ray = mi.Ray3f(mi.Point3f(*tx_pos_np), mi.Vector3f(*d0))
        path_pts, b, hit = [tx_pos_np.copy()], 0, False

        while b <= max_bounces:
            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
            if not si.is_valid():
                path_pts.append(path_pts[-1] + np.array([ray.d.x, ray.d.y, ray.d.z]) * far_miss)
                break

            p_hit = np.array([si.p.x, si.p.y, si.p.z], dtype=float)
            n_hit = np.array([si.n.x, si.n.y, si.n.z], dtype=float)
            path_pts.append(p_hit)

            si_mesh: mi.Mesh = si.shape
            sid = si_mesh.id()

            if sid == "receiver_sphere":
                hit = True
                break

            d_in = np.array([ray.d.x, ray.d.y, ray.d.z], dtype=float)
            d_in /= np.linalg.norm(d_in)

            if sid in dielectric_ids:
                mat_type, eta_obj = material_map[sid]
                cosi = -np.dot(d_in, n_hit)
                if cosi < 0:
                    n_face, eta_i, eta_t = -n_hit, eta_obj, air_eta
                    cosi = -np.dot(d_in, n_face)
                else:
                    n_face, eta_i, eta_t = n_hit, air_eta, eta_obj

                R = fresnel_schlick(max(0.0, min(1.0, cosi)), eta_i, eta_t)
                if rng.rand() < R:
                    d_out = reflect(d_in, n_face)
                else:
                    ok, t_dir = refract(d_in, n_face, eta_i, eta_t)
                    d_out = t_dir if ok else reflect(d_in, n_face)
            else:
                d_out = reflect(d_in, n_hit)

            origin = p_hit + d_out * eps
            ray = mi.Ray3f(mi.Point3f(*origin), mi.Vector3f(*d_out))
            b += 1

        hit_flags.append(hit)
        ray_paths.append(np.array(path_pts))

    print(f"Receiver sphere hit {np.sum(hit_flags)} / {num_rays} rays emitted")

    # ---- Trimesh Visualization ----
    meshes = []
    for obj in objects:
        obj_tm = trimesh.load(obj["path"])
        T_np = np.array(obj["to_world"], dtype=float)

        if isinstance(obj_tm, trimesh.Scene):
            for g in obj_tm.geometry.values():
                g.apply_transform(T_np)
                meshes.append(g)
        else:
            obj_tm.apply_transform(T_np)
            meshes.append(obj_tm)

    tx_ball = trimesh.creation.icosphere(radius=0.08, subdivisions=3)
    tx_ball.apply_translation(tx_pos_np)
    tx_ball.visual.face_colors = [255, 220, 0, 240]
    meshes.append(tx_ball)

    rx_ball = trimesh.creation.icosphere(radius=rx_radius, subdivisions=3)
    rx_ball.apply_translation(np.array(rx_center, dtype=float))
    rx_ball.visual.face_colors = [60, 120, 255, 180]
    meshes.append(rx_ball)

    green, red = np.array([30, 200, 60, 230], np.uint8), np.array([220, 60, 60, 50], np.uint8)
    for path, ok in zip(ray_paths, hit_flags):
        color = green if ok else red
        for s, e in zip(path[:-1], path[1:]):
            cyl = cylinder_between(s, e, ray_radius, color)
            if cyl is not None:
                meshes.append(cyl)

    scene_tri = trimesh.Scene(meshes)
    scene_tri.show()

if __name__ == "__main__":
    R = np.eye(4)
    theta = np.deg2rad(90)
    R[:3,:3] = [[np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]]
    T = np.eye(4)
    T[:3, 3] = [2, 1, 0]
    car_pose_1 = T @ R

    T[:3, 3] = [2, 1, -1]
    car_pose_2 = T @ R

    objects = [
        {
            "id": "car1",
            "path": "pyradar_sandbox/assets/truck/untitled_quardfaced.obj",
            "to_world": car_pose_1,
            "material": "conductor"
        },
        # {
        #     "id": "car2",
        #     "path": "pyradar_sandbox/assets/truck/untitled_quardfaced.obj",
        #     "to_world": car_pose_2,
        #     "material": "conductor"
        # },
        {
            "id": "road",
            "path": "pyradar_sandbox/assets/Road/untitled.obj",
            "to_world": np.eye(4),
            "material": "conductor"
        }
    ]

    main_mitsuba(
        objects=objects,
        num_rays=800,
        tx_pos=(-5, 1.5, -0.25),
        tx_dir=(1, 0, 0),
        cone_deg=30,
        rx_center=(-5, 1.5, 0.25),
        rx_radius=0.3
    )