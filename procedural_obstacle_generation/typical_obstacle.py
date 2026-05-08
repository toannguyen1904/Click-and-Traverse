import numpy as np

def build_obstacles(scene, grids):
    # The | operator is to aggregate multiple obstacles, it's like the logical OR operator
    X, Y, Z = grids
    if scene == "pillar":
        return obs_pillar(X, Y, Z)
    elif scene == "narrow0":
        return obs_narrow(X, Y, Z)
    elif scene == "narrow1":
        return obs_narrow(X, Y, Z, gap_width=0.25)
    elif scene == "narrow0_low":
        return obs_narrow(X, Y, Z, z_high=1.0)
    elif scene == "narrow1_low":
        return obs_narrow(X, Y, Z, gap_width=0.25, z_high=0.7)
    elif scene == "bar0":
        return obs_easy_bar(X, Y, Z)
    elif scene == "bar1":
        return obs_shin_bar(X, Y, Z)
    elif scene == "bar2":
        return obs_hard_bar(X, Y, Z)
    elif scene == "bar3":
        return obs_hard_bar(X, Y, Z, r=0.15, z=0.1,)
    elif scene == "ceil0":
        return obs_ceiling(X, Y, Z)
    elif scene == "ceil1":
        return obs_ceiling(X, Y, Z, z_low=1.0)
    elif scene == "ceilbar0":
        return obs_ceiling(X, Y, Z) | obs_easy_bar(X, Y, Z)
    elif scene == "ceilbar1":
        return obs_ceiling(X, Y, Z, z_low=1.0) | obs_shin_bar(X, Y, Z)
    elif scene == "Mceilbar0":
        return obs_ceiling(X, Y, Z) | obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mceilbar1":
        return obs_ceiling(X, Y, Z, z_low=1.05) | obs_shin_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "hole":
        return obs_ceiling(X, Y, Z) | obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.5)
    elif scene == "empty":
        return obs_empty(X, Y, Z)
    elif scene == "Mceil0":
        return obs_ceiling(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mbar0":
        return obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mceil1":
        return obs_ceiling(X, Y, Z, z_low=1.0) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mbar1":
        return obs_shin_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Mbar2":
        return obs_hard_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.9)
    elif scene == "Nbar0":
        return obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.5)
    elif scene == "Nbar1":
        return obs_easy_bar(X, Y, Z) | obs_narrow(X, Y, Z, gap_width=0.3)
    elif scene == "doubar":
        return obs_double_knee_bars(X, Y, Z)
    elif scene == "bend":
        return obs_bend(X, Y, Z)
    elif scene == "lowcorner":
        return obs_low_corner(X, Y, Z)
    elif scene == "highcorner":
        return obs_high_corner(X, Y, Z)
    else:
        raise ValueError(f"Unknown scene: {scene}")

def _box(X, Y, Z, *, x0, x1, y0, y1, z0, z1):
    # create a box obstacle, limited by the x, y, z coordinates
    # The * is a Python syntax feature that forces the arguments after it to be keyword-only
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)

def _cylinder_along_y(X, Y, Z, *, x, z, r, y0, y1):
    # create a cylinder along the y axis, with predefien x and z coordinates, and radius
    return ((X - x)**2 + (Z - z)**2 <= r**2) & (Y >= y0) & (Y <= y1)

def obs_pillar(X, Y, Z, *, x=1.0, y=0.0, r=0.20, z_low=0.0, z_high=1.5):
    # create a pillar along the z axis, with predefined x and y coordinates, and radius
    return (((X - x)**2 + (Y - y)**2 <= r**2) &
            (Z >= z_low) & (Z <= z_high))


def obs_hard_bar(X, Y, Z, *, x_center=1.0, r=0.1, z=0.08, y_width=2.0):
    # create a hard bar, with z = 0.08
    y0, y1 = -y_width/2, +y_width/2
    return _cylinder_along_y(X, Y, Z, x=x_center, z=z, r=r, y0=y0, y1=y1)

def obs_shin_bar(X, Y, Z, *, x_center=1.0, r=0.08, z=0.05, y_width=2.0):
    # create a shin bar, with z = 0.05
    y0, y1 = -y_width/2, +y_width/2
    return _cylinder_along_y(X, Y, Z, x=x_center, z=z, r=r, y0=y0, y1=y1)

def obs_easy_bar(X, Y, Z, *, x_center=1.0, r=0.08, z=0.00, y_width=2.0):
    # create an easy bar, with z = 0.00
    y0, y1 = -y_width/2, +y_width/2
    return _cylinder_along_y(X, Y, Z, x=x_center, z=z, r=r, y0=y0, y1=y1)


def obs_double_knee_bars(X, Y, Z, *, x1=0.75, x2=1.25, r=0.08, z=0.05, y_width=2.0):
    # create two knee bars that are parallel to each other and along the y axis
    y0, y1 = -y_width/2, +y_width/2
    bar1 = _cylinder_along_y(X, Y, Z, x=x1, z=z, r=r, y0=y0, y1=y1)
    bar2 = _cylinder_along_y(X, Y, Z, x=x2, z=z, r=r, y0=y0, y1=y1)
    return bar1 | bar2


def obs_ceiling(X, Y, Z, *, x_center=1, length=0.2, thickness=1.0, z_low=1.15, y_width=2.0):
    # create a ceiling obstacle
    x0, x1 = x_center - length/2, x_center + length/2
    y0, y1 = -y_width/2, +y_width/2
    z0, z1 = z_low, z_low + thickness
    return _box(X, Y, Z, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)

def obs_narrow(X, Y, Z, x=1., wall_thickness=0.12, y = 0, gap_width=0.4, z_low=0.0, z_high=1.5):
    # create a narrow obstacle, with a wall and a gap
    wall = (X >= x - wall_thickness/2) & (X <= x + wall_thickness/2) & (Z >= z_low) & (Z <= z_high)
    slit = (np.abs(Y-y) <= gap_width/2)
    return wall & (~slit)


def obs_empty(X, Y, Z):
    # create a nearly empty obstacle
    return (X < -0.4) & (Z > 1.44) & ((Y > 0.9) | (Y < -0.9)) # nearly empty

def obs_ankle_block_field(X, Y, Z, *, xs=(0.6, 0.9, 1.2, 1.5), y_span=0.35, h=0.18, w=0.16):
    # A row of 4 short rectangular blocks spread along the X axis at x = 0.6, 0.9, 1.2, 1.5,
    # each only 18 cm tall and narrow in Y. The robot has to step over or between them — targeting ankle/foot level.
    # currently not used
    mask = np.zeros_like(X, dtype=bool)
    for xc in xs:
        mask |= _box(X, Y, Z,
                     x0=xc - w/2, x1=xc + w/2,
                     y0=-y_span,  y1=+y_span,
                     z0=0.0,      z1=h)
    return mask

def obs_bend(X, Y, Z, *, x1=0.9, x2=1.3, thickness=0.24, y_cover=0.05, z0=0.5, z1=1.1):
    x0 = 1.0
    fin1 = _box(X, Y, Z, x0=x0 - thickness/2, x1=x0 + thickness/2, y0=-1.0, y1=0.1, z0=z1, z1=2.0)
    fin2 = _box(X, Y, Z, x0=x0 - thickness, x1=x0 + thickness, y0=0.18, y1=1.0, z0=0, z1=z0)
    fin3 = _box(X, Y, Z, x0=x0 - thickness, x1=x0 + thickness, y0=-1.0, y1=-0.18, z0=0, z1=z0)
    return fin1 | fin2 | fin3

def obs_chest_2(X, Y, Z, *, x1=0.9, x2=1.3, thickness=0.12, y_cover=0.05, z0=0.4, z1=1.15):
    # A wall with an asymmetric opening — designed to require the robot to crouch or twist its torso, currently not used
    x0 = 1.0
    fin1 = _box(X, Y, Z, x0=x0 - thickness/2, x1=x0 + thickness/2, y0=-1.0, y1=1.0, z0=z1, z1=2.0)
    fin2 = _box(X, Y, Z, x0=x0 - thickness/2, x1=x0 + thickness/2, y0=0.05, y1=1.0, z0=0, z1=z0)
    fin3 = _box(X, Y, Z, x0=x0 - thickness/2, x1=x0 + thickness/2, y0=-1.0, y1=-0.25, z0=0, z1=2.0)
    return fin1 | fin2 | fin3

def obs_low_corner(X, Y, Z, *, x1=0.9, x2=1.3, thickness=0.12, y_cover=0.15, z0=0.4, z1=1.0):
    x0 = 1.0
    fin1 = _box(X, Y, Z, x0=-0.5, x1=1.1, y0=-1.0, y1=-0.5, z0=0, z1=z0)
    fin2 = _box(X, Y, Z, x0=1.1, x1=1.3, y0=-1.0, y1=0.1, z0=0, z1=z0)
    fin3 = _box(X, Y, Z, x0=0.5, x1=0.7, y0=-0.1, y1=1.0, z0=0, z1=z0)
    return fin1 | fin2 | fin3

def obs_high_corner(X, Y, Z, *, x1=0.9, x2=1.3, thickness=0.12, y_cover=0.15, z0=1.0, z1=1.0):
    x0 = 1.0
    fin1 = _box(X, Y, Z, x0=-0.5, x1=1.3, y0=-1.0, y1=-0.5, z0=0, z1=z0)
    fin2 = _box(X, Y, Z, x0=1.3, x1=1.5, y0=-1.0, y1=0.1, z0=0, z1=z0)
    fin3 = _box(X, Y, Z, x0=0.5, x1=0.7, y0=-0.0, y1=1.0, z0=0, z1=z0)
    return fin1 | fin2 | fin3


if __name__ == "__main__":
    # script to visualize the typical obstacles
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    SCENES = [
        "pillar", "narrow0", "narrow1", "narrow0_low", "narrow1_low",
        "bar0", "bar1", "bar2", "bar3",
        "ceil0", "ceil1",
        "ceilbar0", "ceilbar1",
        "Mceil0", "Mceil1",
        "Mceilbar0", "Mceilbar1",
        "Mbar0", "Mbar1",
        "Nbar0", "Nbar1",
        "doubar", "bend",
        "lowcorner", "highcorner",
        "empty",
    ]

    VOXEL = 0.04
    xv = np.arange(-0.5, 2.5, VOXEL)
    yv = np.arange(-1.0, 1.0, VOXEL)
    zv = np.arange(0.0,  1.6, VOXEL)
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')

    n = len(SCENES)
    cols = 5
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 4, rows * 3.5))

    for i, scene in enumerate(SCENES):
        mask = build_obstacles(scene, (X, Y, Z))

        # Sample occupied voxel centres for scatter plot
        idx = np.argwhere(mask)
        if len(idx) > 8000:
            idx = idx[np.random.choice(len(idx), 8000, replace=False)]
        pts = idx * VOXEL + np.array([-0.5, -1.0, 0.0])  # back to world coords

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=pts[:, 2],
                       cmap="plasma", depthshade=True)
        ax.set_title(scene, fontsize=9)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0.0, 1.6)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=20, azim=-60)

    fig.suptitle("Typical obstacle scenes", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("/home/tien/Code/Click-and-Traverse/typical_scenes.png", dpi=120, bbox_inches="tight")
    print("Image saved to /home/tien/Code/Click-and-Traverse/typical_scenes.png")
    plt.show()

