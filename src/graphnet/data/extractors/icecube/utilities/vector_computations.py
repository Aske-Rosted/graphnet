import numpy as np


def closest_approach_distance_vector(
    particle,
    dom_x,
    dom_y,
    dom_z,
):
    """
    Takes the Vector Position and Direction and Computes:
    -> Distance between array of x,y,z points and the vector
    -> Point Along the Track (Relative to Closest Approach Point to the Detector)
    """

    pos_x = particle.pos.x
    pos_y = particle.pos.y
    pos_z = particle.pos.z
    theta = particle.dir.zenith
    phi = particle.dir.azimuth

    e_x = -np.sin(theta) * np.cos(phi)
    e_y = -np.sin(theta) * np.sin(phi)
    e_z = -np.cos(theta)

    h_x = dom_x - pos_x
    h_y = dom_y - pos_y
    h_z = dom_z - pos_z

    s = e_x * h_x + e_y * h_y + e_z * h_z

    pos_cx = pos_x + s * e_x
    pos_cy = pos_y + s * e_y
    pos_cz = pos_z + s * e_z

    r = np.sqrt(
        (pos_cx - dom_x) ** 2 + (pos_cy - dom_y) ** 2 + (pos_cz - dom_z) ** 2
    )
    return r, pos_cx, pos_cy, pos_cz, s
