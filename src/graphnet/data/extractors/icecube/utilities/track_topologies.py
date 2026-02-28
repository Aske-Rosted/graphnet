"""
For Computing If An Event is A Starting Track Under Several Definitions
"""

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package():
    from icecube import MuonGun
    from icecube import (  
        icetray,
        dataclasses,
    )
import pandas as pd
import numpy as np
dom_list = pd.read_csv(
    '/cvmfs/icecube.opensciencegrid.org/users/mnakos/process_training_data/DOM_Positions.csv',
)

import numpy as np
import shapely.geometry as geom

MuonGunGCD= '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz'
detector_boundaries = MuonGun.ExtrudedPolygon.from_file(MuonGunGCD, padding=0)
detector_boundaries_100 = MuonGun.ExtrudedPolygon.from_file(MuonGunGCD, padding=100)
detector_boundaries_200 = MuonGun.ExtrudedPolygon.from_file(MuonGunGCD, padding=200)


def compute_lengths(first_pos, second_pos):

    is_skimming = False
    if np.isnan(first_pos) & np.isnan(second_pos):
        track_length_inside_detector = 0
        veto_length = 0
        is_skimming = True
    elif (first_pos < 0) & (second_pos <= 0):
        track_length_inside_detector = 0
        veto_length = second_pos - first_pos
    elif (first_pos < 0) & (second_pos > 0):
        track_length_inside_detector = second_pos
        veto_length = -first_pos
    else:
        track_length_inside_detector = second_pos - first_pos
        veto_length = 0

    return track_length_inside_detector, veto_length, is_skimming

def closest_to_center(
    frame,
    leading_particle,
):
        
    d_c, x_d, y_d, z_d, s = closest_approach_distance_vector(
        particle=leading_particle,
        dom_x=0,
        dom_y=0,
        dom_z=0,
    )

    frame['ClosestApproachPosition'] = dataclasses.I3Position(x_d, y_d, z_d)

def compute_skimming_event(
    frame,
    starting_inside,
    detector_boundaries = detector_boundaries,
    detector_boundaries_100 = detector_boundaries_100,
    detector_boundaries_200 = detector_boundaries_200,
):
    # Extra is the primary, primary is the primary composition
    extra, primary = get_leading_particle(frame)

    intersections = detector_boundaries.intersection(primary.pos, primary.dir)
    intersections_100 = detector_boundaries_100.intersection(primary.pos, primary.dir)
    intersections_200 = detector_boundaries_200.intersection(primary.pos, primary.dir)
    
    int_one, int_two = intersections.first, intersections.second
    int_one_100, int_two_100 = intersections_100.first, intersections_100.second
    int_one_200, int_two_200 = intersections_200.first, intersections_200.second

    length, veto_size, skimming = compute_lengths(int_one, int_two)
    length_100, veto_size_100, skimming_100 = compute_lengths(int_one_100, int_two_100)
    length_200, veto_size_200, skimming_200 = compute_lengths(int_one_200, int_two_200)

    frame['TrackLength_Inside_Detector'] = dataclasses.I3Double(length)
    frame['TrackLength_Near_Detector_100'] = dataclasses.I3Double(length_100)
    frame['TrackLength_Near_Detector_200'] = dataclasses.I3Double(length_200)
    frame['VetoLength_Inside_Detector'] = dataclasses.I3Double(veto_size)
    frame['VetoLength_Near_Detector_100'] = dataclasses.I3Double(veto_size_100)
    frame['VetoLength_Near_Detector_200'] = dataclasses.I3Double(veto_size_200)

    closest_to_center(
        frame,
        leading_particle=primary,
    )

    return skimming


OUTER_LAYER = [1,2,3,4,5,6,13,21,30,40,50,59,67,74,73,72,78,77,76,75,68,60,51,41,31,22,14,7]
INNER_LAYER = [8,9,10,11,12,20,29,39,49,58,66,65,64,71,70,69,61,52,42,32,23,15] 

import matplotlib.path as mpltPath

def generate_path(
    dom_positions = dom_list,
    string_numbers = None,
    buffer = 0,
):
    """
    Generates the Outer Boundaries of the Detector for Staring Tracks
    """

    if string_numbers == None:
        string_numbers = OUTER_LAYER

    df_bound = dom_positions[dom_positions['string'].isin(string_numbers)]

    order = {val: idx for idx, val in enumerate(string_numbers)}
    df_bound['order'] = df_bound['string'].map(order)
    df_bound = df_bound.sort_values(by='order').drop(columns='order')

    codes = [mpltPath.Path.MOVETO] + [mpltPath.Path.LINETO] * (df_bound[['x', 'y']].to_numpy().shape[0] - 2) + [mpltPath.Path.CLOSEPOLY]
    boundary = mpltPath.Path(df_bound[['x', 'y']].to_numpy(), codes)

    if buffer > 0:
        original_polygon = geom.Polygon(df_bound[['x', 'y']].to_numpy())

        # Expand the shape outward by 300 meters
        expanded_polygon = original_polygon.buffer(buffer, resolution=16)

        # Extract expanded vertices
        expanded_vertices = np.array(expanded_polygon.exterior.coords)

        expanded_vertices = np.array(expanded_polygon.exterior.coords)
        codes = [mpltPath.Path.MOVETO] + [mpltPath.Path.LINETO] * (expanded_vertices.shape[0] - 2) + [mpltPath.Path.CLOSEPOLY]
        boundary = mpltPath.Path(expanded_vertices, codes)

    return boundary
    

outer_boundary = generate_path(
    dom_positions = dom_list,
    string_numbers = OUTER_LAYER,
    buffer = 0,
)
# Make Boundary Plots using Starting Track List

from graphnet.data.extractors.icecube.utilities.vector_computations import closest_approach_distance_vector


def get_leading_particle(
    frame,
):
    
    primary = frame['PolyplopiaPrimary']
    pdg = frame['PolyplopiaPrimary'].pdg_encoding
    full_mctree = frame['I3MCTree']
    mctree = frame['I3MCTree_preMuonProp']
    if np.abs(pdg) in [12, 14, 16]:
        current = mctree[1]
        while mctree.number_of_children(current) > 0:
            current = mctree.first_child(current)
    else:
        current = mctree[frame['PolyplopiaPrimary']]
        highest_energy = -1
        bundle_particles = mctree.get_daughters(current)
        for particle in bundle_particles:
            if (particle.type_string in ['MuPlus', 'MuMinus'] and particle.location_type_string == 'InIce'):
                if particle.energy > highest_energy:
                    highest_energy = particle.energy
                    current = particle

    tracklist = frame['MMCTrackList']

    e_initial = 0
    for track in tracklist:
        #if full_mctree.is_in_subtree(primary, track.particle) == True: # Cleaning Coincidence Hits
        if track.Ei > e_initial:
            e_initial = track.Ei
            current = track.particle

    return primary, current

def indentify_interaction_vertex(
    frame,
):
    """
    Saves the interaction vertex of the event to a key
    """

    mctree = frame['I3MCTree_preMuonProp']

    current = mctree[1]

    while mctree.number_of_children(current) > 0:
        current = mctree.first_child(current)

    frame['InteractionVertexParticle'] = current

   
def interaction_vertex_position(
    frame,
    leading_choice
):
    
    paritcle = frame['InteractionVertexParticle'] 

    vr, vx, vy, vz, vs = closest_approach_distance_vector(
        leading_choice,
        paritcle.pos.x,
        paritcle.pos.y,
        paritcle.pos.z,
    )
    
    return vs

def add_closest_approach_vectors(
    leading_particle,
    pulses,
):
    
    r, cx, cy, cz, s = closest_approach_distance_vector(
        leading_particle,
        pulses['x'],
        pulses['y'],
        pulses['z'],
    )

    dr, dcx, dcy, dcz, ds = closest_approach_distance_vector(
        leading_particle,
        0,
        0,
        0,
    )

    pulses['r'] = r
    # Closest Approach Point to the Detector Origin is Zero
    pulses['s'] = s - ds
    pulses['r_det'] = np.sqrt(pulses['x']**2 + pulses['y']**2)
    return pulses

"""
Optimization Parameters
"""
def is_starting_visible(
    frame,
    df,
    int_vertex,
    optimzied = False,
    charge_bins = np.geomspace(1000, 5e5, 13),
    n_hits =[51, 48, 45, 45, 39, 43, 34, 30, 30, 27, 24, 29],
    r = 250,
):
    
    charge = frame['HQTOT']
    # Cut on 1 DOM within 250 meters of the dom

    df['pretex'] = df['s'] < int_vertex

    veto_region_size = len(df[(df['r'] < r) & (df['pretex'])])
    is_starting_vis = False
    if optimzied == False:
        if veto_region_size >= 0:
            is_starting_vis = True
    else:
        
        if charge <= charge_bins[0]:
            index = 0
        elif charge <= charge_bins[-1]:              
            #for i, hits in enumerate(n_hits):
            #    if (charge_bins[i] >= charge) & (charge_bins[i+1]  <= charge):
            #        index = n_hits[i]
            #        break
            max_charge = charge_bins[charge_bins < charge].max()
            index = np.where(charge_bins == max_charge)[0][0]
        else:
            index = -1

        veto_region_threshold = n_hits[index]

        if veto_region_size >= veto_region_threshold:
            is_starting_vis = True
        

    return is_starting_vis



def get_topology_metrics(
    frame,
    dom_list = dom_list,
    boundary = outer_boundary
):
    
    indentify_interaction_vertex(
        frame,
    )
    
    primary, leading_lepton = get_leading_particle(
        frame,
    )

    # Convert to DOM Positions to Relative to the Track
    primary_df = add_closest_approach_vectors(
        leading_lepton,
        dom_list.copy(),
    )

    leading_df = add_closest_approach_vectors(
        primary,
        dom_list.copy(),
    )

    primary_s = interaction_vertex_position(
        frame,
        primary,
    )
    
    leading_s = interaction_vertex_position(
        frame,
        leading_lepton,
    )


    # Find the Cuts

    primary_high_rate = is_starting_visible(frame, int_vertex=primary_s, df=primary_df, optimzied=False)
    primary_low_rate = is_starting_visible(frame, int_vertex=primary_s, df=primary_df, optimzied=True)
    del primary_df
    leading_high_rate = is_starting_visible(frame, int_vertex=leading_s, df=leading_df, optimzied=False)
    leading_low_rate = is_starting_visible(frame, int_vertex=leading_s, df=leading_df, optimzied=True)
    del leading_df

    # Conventional Satrting
    int_vert = frame['InteractionVertexParticle']
    if (((int_vert.pos.z <=max(dom_list.z)) and (int_vert.pos.z>=min(dom_list.z)))) and boundary.contains_points([(int_vert.pos.x, int_vert.pos.y)]):
        starting_detector = True
    else:
        starting_detector = False

    return starting_detector, primary_high_rate, primary_low_rate, leading_high_rate, leading_low_rate


    