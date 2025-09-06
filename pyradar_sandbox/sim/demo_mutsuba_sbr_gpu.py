
import math
import numpy as np
import mitsuba as mi
import trimesh

# -------------------- Mitsuba 变体 --------------------
def try_set_mitsuba_variant():
    # 尝试 GPU 优先
    for v in ["cuda_ad_rgb", "cuda_rgb", "llvm_ad_rgb", "scalar_rgb"]:
        try:
            mi.set_variant(v)
            print(f"[Mitsuba] Using variant: {v}")
            return
        except Exception:
            pass
    raise RuntimeError("无法设置 Mitsuba 变体，请检查 Mitsuba 安装。")


# -------------------- 方向采样：圆锥内均匀 --------------------
def sample_directions_in_cone(n, cone_angle_deg, forward=np.array([0,0,-1], dtype=float)):
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
    # 把 local z=[0,0,1] 旋到 forward
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

# -------------------- 光学：反射、折射、Fresnel --------------------
def reflect(dir_in, normal):
    # dir_in、normal 要单位化；公式返回单位向量
    d = dir_in / np.linalg.norm(dir_in)
    n = normal / np.linalg.norm(normal)
    return d - 2.0 * np.dot(d, n) * n

def refract(dir_in, normal, eta_i, eta_t):
    """
    Snell 折射：从介质 eta_i 进入 eta_t
    返回 (success, dir_out)
    如果全内反射，success=False
    """
    d = dir_in / np.linalg.norm(dir_in)
    n = normal / np.linalg.norm(normal)

    # 保证 n 指向“与 d 相反”（即入射角余弦为正）
    cosi = -np.dot(d, n)
    if cosi < 0:
        # 从“里向外”，翻转法线，并交换折射率
        n = -n
        cosi = -np.dot(d, n)
        eta_i, eta_t = eta_t, eta_i

    eta = eta_i / eta_t
    k = 1.0 - eta**2 * (1.0 - cosi**2)
    if k < 0.0:
        return False, None  # 全内反射
    cost = math.sqrt(k)
    t = eta * d + (eta * cosi - cost) * n
    t /= np.linalg.norm(t)
    return True, t

def fresnel_schlick(cos_theta, eta_i, eta_t):
    # Schlick 近似：反射系数
    r0 = ((eta_t - eta_i) / (eta_t + eta_i))**2
    return r0 + (1 - r0) * (1 - cos_theta)**5

# -------------------- 可视化：用圆柱段画射线 --------------------
def cylinder_between(p0, p1, radius, rgba):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-9:
        return None
    vn = v / L
    # 默认圆柱沿 z 轴、中心在原点（[-h/2,h/2]）
    cyl = trimesh.creation.cylinder(radius=radius, height=L, sections=12)
    # 对齐到方向 vn
    T = trimesh.geometry.align_vectors([0,0,1], vn)
    cyl.apply_transform(T)
    # 平移到中点
    mid = (p0 + p1) * 0.5
    cyl.apply_translation(mid)
    cyl.visual.face_colors = rgba
    return cyl

# -------------------- 主流程 --------------------
def main_mitsuba(
    obj_path: str,
    num_rays: int = 800,
    max_bounces: int = 6,
    tx_pos=(0.0, 1.5, 5.0),
    tx_dir=(0.0, 0.0, -1.0),
    cone_deg=60.0,
    # 接收球
    rx_center=(0.0, 1.0, 0.0),
    rx_radius=0.5,
    # 材质设置
    car_material="conductor",  # 'conductor' or 'dielectric'
    car_eta=1.5,               # 车体若为 dielectric 的折射率
    air_eta=1.0,
    # 可视化
    ray_radius=0.015
):
    try_set_mitsuba_variant()

    # ---- 场景构建：车 + 接收球（接收球参与求交，命中即结束）----
    car_bsdf = {'type': 'roughconductor'} if car_material == "conductor" else {
        'type': 'dielectric', 'int_ior': car_eta
    }

    scene_dict = {
        'type':'scene',
        'car': {
            'type':'obj',
            'filename': obj_path,
            'bsdf': car_bsdf,
            'id': 'car'
        },
        'receiver_sphere': {
            'type': 'sphere',
            'center': rx_center,
            'radius': rx_radius,
            # 让接收球不再反射/折射，避免继续弹射
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0,0,0]}},
            'id': 'receiver_sphere'
        }
    }
    scene = mi.load_dict(scene_dict)

    # 方向
    dirs = sample_directions_in_cone(num_rays, cone_deg, forward=np.array(tx_dir, dtype=float))
    tx_pos_np = np.array(tx_pos, dtype=float)

    # 记录
    hit_flags = []       # True=命中接收球
    ray_paths  = []      # 每条主射线的分段点序列

    # 哪些 shape 需要折射：如果车体设置成 dielectric，则加入
    dielectric_ids = set()
    if car_material == "dielectric":
        dielectric_ids.add('car')

    rng = np.random.RandomState(42)
    eps = 1e-4
    far_miss = 60.0

    for i in range(num_rays):
        d0 = np.array(dirs[i], dtype=float).flatten()  # Ensure 1D array
        ray = mi.Ray3f(
            mi.Point3f(*(float(x) for x in tx_pos_np.astype(np.float32))),
            mi.Vector3f(*(float(x) for x in d0.astype(np.float32)))
        )
        path_pts = [tx_pos_np.copy()]
        b = 0
        hit = False

        while b <= max_bounces:
            si = scene.ray_intersect(ray)
            if not si.is_valid():
                # 飞出场景：拉一段到远处
                far_point = path_pts[-1] + np.array([ray.d.x, ray.d.y, ray.d.z], dtype=float).flatten() * far_miss
                path_pts.append(far_point)
                break

            # 命中点与法线，确保 1D
            p_hit = np.array([si.p.x, si.p.y, si.p.z], dtype=float).flatten()
            n_hit = np.array([si.n.x, si.n.y, si.n.z], dtype=float).flatten()
            path_pts.append(p_hit.flatten())

            sid = si.shape.label  # ← Mitsuba 3 replacement for .id()

            # 命中接收球：计为成功并结束
            if sid == 'receiver_sphere':
                hit = True
                break

            # 入射方向（单位向量）
            d_in = np.array([ray.d.x, ray.d.y, ray.d.z], dtype=float).flatten()
            d_in /= np.linalg.norm(d_in)

            # 反射 or 折射
            if sid in dielectric_ids:
                # 折射介质：根据入射侧决定 eta_i/eta_t
                cosi = -np.dot(d_in, n_hit)
                if cosi < 0:
                    n_face = -n_hit
                    eta_i, eta_t = car_eta, air_eta
                    cosi = -np.dot(d_in, n_face)
                else:
                    n_face = n_hit
                    eta_i, eta_t = air_eta, car_eta

                R = fresnel_schlick(max(0.0, min(1.0, cosi)), eta_i, eta_t)
                if rng.rand() < R:
                    d_out = reflect(d_in, n_face)
                else:
                    ok, t_dir = refract(d_in, n_face, eta_i, eta_t)
                    d_out = t_dir if ok else reflect(d_in, n_face)
            else:
                d_out = reflect(d_in, n_hit)

            # 下一跳射线
            origin = p_hit + d_out * eps
            ray = mi.Ray3f(
                mi.Point3f(*(float(x) for x in origin)),
                mi.Vector3f(*(float(x) for x in d_out))

            )
            b += 1

        hit_flags.append(hit)
        ray_paths.append(np.stack(path_pts))

    hit_count = int(np.sum(hit_flags))
    print(f"接收球命中 {hit_count} / 发射 {num_rays} 条")

    # -------------------- Trimesh 可视化 --------------------
    meshes = []

    # 车模（trimesh 载入）
    car_tm = trimesh.load(obj_path)
    if isinstance(car_tm, trimesh.Scene):
        for g in car_tm.geometry.values():
            meshes.append(g)
    else:
        meshes.append(car_tm)

    # 发射器小球（黄色）
    tx_ball = trimesh.creation.icosphere(radius=0.08, subdivisions=3)
    tx_ball.apply_translation(tx_pos_np)
    tx_ball.visual.face_colors = [255, 220, 0, 240]
    meshes.append(tx_ball)

    # 接收球（蓝色半透明）
    rx_ball = trimesh.creation.icosphere(radius=rx_radius, subdivisions=3)
    rx_ball.apply_translation(np.array(rx_center, dtype=float))
    rx_ball.visual.face_colors = [60, 120, 255, 180]
    meshes.append(rx_ball)

    # 射线段（命中=绿，未命中=红），用细圆柱表示
    green = np.array([30, 200, 60, 230], dtype=np.uint8)
    red   = np.array([220, 60, 60, 50], dtype=np.uint8)

    for path, ok in zip(ray_paths, hit_flags):
        color = green if ok else red
        for s, e in zip(path[:-1], path[1:]):
            cyl = cylinder_between(s, e, ray_radius, color)
            if cyl is not None:
                meshes.append(cyl)

    scene_tri = trimesh.Scene(meshes)
    scene_tri.show()

    

# -------------------- 运行入口 --------------------
if __name__ == "__main__":
    # 换成你的 Car 模型路径
    # obj_path = "C:\\Project_HOME\\pyradar\\pyradar_sandbox\\assets\\Road\\untitled.obj"
    obj_path = "G:/Other computers/PC-HOME/Project_HOME/pyradar/pyradar_sandbox/assets/Road/untitled.obj"

    # obj_path = "C:/Project_HOME/pyradar/pyradar_sandbox/assets/P911GT/Porsche_911_GT2.obj"


    # 备注：想看折射，把 car_material 改成 'dielectric'
    main_mitsuba(
        obj_path=obj_path,
        num_rays=1000,
        max_bounces=6,
        tx_pos=(-5.0, 5.0, 0.0),
        tx_dir=(1.0, -1.0, 0.0),
        cone_deg=20.0,
        rx_center=(5.0, 5.0, 0.0),
        rx_radius=0.8,
        car_material="conductor",    # ← 换成 "dielectric" 可以看到折射路径
        car_eta=1.5,
        air_eta=1.0,
        ray_radius=0.015
    )
