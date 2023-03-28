
'''
MAP Motion Tracking Schema
'''

import datajoint as dj
import numpy as np
import pandas as pd

from . import experiment, lab
from . import get_schema_name, create_schema_settings

schema = dj.schema(get_schema_name('tracking'), **create_schema_settings)
[experiment]  # NOQA flake8


@schema
class TrackingDevice(dj.Lookup):
    definition = """
    tracking_device:                    varchar(20)     # device type/function
    ---
    tracking_position:                  varchar(20)     # device position
    sampling_rate:                      decimal(8, 4)   # sampling rate (Hz)
    tracking_device_description:        varchar(100)    # device description
    """
    contents = [
        ('Camera 0', 'side', 1/0.0034, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
        ('Camera 1', 'bottom', 1/0.0034, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
        ('Camera 2', 'body', 1/0.01, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
        ('Camera 3', 'side', 1 / 0.0034, 'Blackfly S BFS-U3-04S2M-CS (FLIR)'),
        ('Camera 4', 'bottom', 1 / 0.0034, 'Blackfly S BFS-U3-04S2M-CS (FLIR)'),
        ('Camera 5', 'body', 1 / 0.01, 'Blackfly S BFS-U3-04S2M-CS (FLIR)'),
    ]


@schema
class Tracking(dj.Imported):
    '''
    Video feature tracking.
    Position values in px; camera location is fixed & real-world position
    can be computed from px values.
    '''

    definition = """
    -> experiment.SessionTrial
    -> TrackingDevice
    ---
    tracking_samples: int             # number of events (possibly frame number, relative to the start of the trial)
    """
    
    class Frame(dj.Part):
        definition = """
        -> Tracking
        ---
        frame_time: longblob   # Global session-wise time (in sec)
        """

    class NoseTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        nose_x:                 longblob        # nose x location (px)
        nose_y:                 longblob        # nose y location (px)
        nose_likelihood:        longblob        # nose location likelihood
        """

    class TongueTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        tongue_x:               longblob        # tongue x location (px)
        tongue_y:               longblob        # tongue y location (px)
        tongue_likelihood:      longblob        # tongue location likelihood
        """
        
    class TongueSideTracking(dj.Part):
        definition = """
        -> Tracking
        side:               varchar(36)     # leftfront, rightfront, leftback, rightback, ...
        ---
        tongue_side_x:               longblob        # tongue x location (px)
        tongue_side_y:               longblob        # tongue y location (px)
        tongue_side_likelihood:      longblob        # tongue location likelihood
        """

    class JawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        jaw_x:                  longblob        # jaw x location (px)
        jaw_y:                  longblob        # jaw y location (px)
        jaw_likelihood:         longblob        # jaw location likelihood
        """

    class LeftPawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        left_paw_x:             longblob        # left paw x location (px)
        left_paw_y:             longblob        # left paw y location (px)
        left_paw_likelihood:    longblob        # left paw location likelihood
        """

    class RightPawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        right_paw_x:            longblob        # right paw x location (px)
        right_paw_y:            longblob        # right_paw y location (px)
        right_paw_likelihood:   longblob        # right_paw location likelihood
        """

    class LickPortTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        lickport_x:            longblob        # right paw x location (px)
        lickport_y:            longblob        # right_paw y location (px)
        lickport_likelihood:   longblob        # right_paw location likelihood
        """

    class WhiskerTracking(dj.Part):
        definition = """
        -> Tracking
        whisker_name:         varchar(36)
        ---
        whisker_x:            longblob        # whisker x location (px)
        whisker_y:            longblob        # whisker y location (px)
        whisker_likelihood:   longblob        # whisker location likelihood
        """
        
    class PupilSideTracking(dj.Part):
        definition = """
        -> Tracking
        side:     varchar(36)   # Up, Down, Left, Right
        ---
        pupil_side_x:       longblob   
        pupil_side_y:       longblob       
        pupil_side_likelihood:    longblob 
        """
        

    @property
    def tracking_features(self):
        return {'NoseTracking': Tracking.NoseTracking,
                'TongueTracking': Tracking.TongueTracking,
                'JawTracking': Tracking.JawTracking,
                'LeftPawTracking': Tracking.LeftPawTracking,
                'RightPawTracking': Tracking.RightPawTracking,
                'LickPortTracking': Tracking.LickPortTracking,
                'WhiskerTracking': Tracking.WhiskerTracking,
                
                # For foraging tracking
                'nose': Tracking.NoseTracking,
                'tongue': Tracking.TongueTracking,
                'tongue_side': Tracking.TongueSideTracking,
                'jaw': Tracking.JawTracking,
                'left_paw': Tracking.LeftPawTracking,
                'right_paw': Tracking.RightPawTracking,
                'whisker': Tracking.WhiskerTracking,     
                'pupil_side': Tracking.PupilSideTracking,           
                }


@schema
class TrackedWhisker(dj.Manual):
    definition = """
    -> Tracking.WhiskerTracking
    """

    class Whisker(dj.Part):
        definition = """
        -> master
        -> lab.Whisker
        """
        

@schema
class TrackingPupilSize(dj.Computed):
    """
    Least squares fit of pupil_side tracking
    The output parameters:
    ((x - x0) * cos(phi) + y * sin(phi))^2 / a^2 + ((x - x0) * sin(phi) - y * cos(phi))^2 / b^2 = 1
    
    size = pi * ap * bp
    """
    definition = """
    -> experiment.SessionTrial
    ---
    polygon_area:    longblob
    ellipse_area:    longblob
    x0:      longblob
    y0:      longblob
    a:      longblob
    b:      longblob
    phi:    longblob
    likelihood:   longblob   # product of likelihoods for all four sides
    """
    
    key_source = experiment.SessionTrial & Tracking.PupilSideTracking
    
    def make(self, key):

        sides = ['Down', 'Left', 'Up', 'Right']
        df_this_trial = (Tracking.PupilSideTracking & key).fetch(format='frame').reset_index()
        df_this_trial['side'] = pd.Categorical(df_this_trial['side'], categories=sides, ordered=True)
        df_this_trial = df_this_trial.sort_values('side')  # Important for poly_area to work
        
        results = []
        
        # For each time point, fit ellipse
        for i in range(len(df_this_trial.pupil_side_x.iloc[0])):

            x = np.array([[a[i]] for a in df_this_trial.pupil_side_x])
            y = np.array([[a[i]] for a in df_this_trial.pupil_side_y])

            x_0 = x.mean()
            y_0 = y.mean()

            coeffs = fit_ellipse(x, y)

            # print('Fitted parameters:')
            # print('a, b, c, d, e, f =', coeffs)
            
            x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
            x0 += x_0
            y0 += y_0
            
            # Use polygon
            polygon_area = poly_area(x.ravel(), y.ravel())
            
            results.append([x0, y0, ap, bp, e, phi, polygon_area])
            
        x0s, y0s, aps, bps, _, phis, polygon_areas =  np.array(results).T
    
        likelihood = np.vstack(df_this_trial.pupil_side_likelihood)
        likelihood_ellipse = likelihood.prod(axis=0)
        
        
        # Do insert
        self.insert1(dict(**key,
                          polygon_area=polygon_areas,
                          ellipse_area=np.pi * aps * bps,
                          x0=x0s,
                          y0=y0s,
                          a=aps,
                          b=bps,
                          phi=phis,
                          likelihood=likelihood_ellipse))


# ------------------------ Quality Control Metrics -----------------------


@schema
class TrackingQC(dj.Computed):
    definition = """
    -> Tracking
    tracked_feature: varchar(32)  # e.g. RightPaw, LickPort
    ---
    bad_percentage: float  # percentage of bad frames out of all frames 
    bad_frames: longblob  # array of frame indices that are "bad"
    """

    threshold_mapper = {('RRig2', 'side', 'NoseTracking'): 20,
                        ('RRig2', 'side', 'JawTracking'): 20,
                        ('RRig-MTL', 'side', 'JawTracking'): 20,
                        ('RRig-MTL', 'bottom', 'JawTracking'): 20}

    def make(self, key):
        rig = (experiment.Session & key).fetch1('rig')
        device, device_position = (TrackingDevice & key).fetch1('tracking_device', 'tracking_position')

        tracking_qc_list = []
        for feature_name, feature_table in Tracking().tracking_features.items():
            if feature_name in ('JawTracking', 'NoseTracking'):
                if not feature_table & key:
                    continue

                bad_threshold = self.threshold_mapper[(rig, device_position, feature_name)]
                tracking_data = (Tracking * feature_table & key).fetch1()

                attribute_prefix = feature_name.replace('Tracking', '').lower()

                x_diff = np.diff(tracking_data[attribute_prefix + '_x'])
                y_diff = np.diff(tracking_data[attribute_prefix + '_y'])
                bad_frames = np.where(np.logical_or(x_diff > bad_threshold, y_diff > bad_threshold))[0]
                bad_percentage = len(bad_frames) / tracking_data['tracking_samples'] * 100

                tracking_qc_list.append({**key, 'tracked_feature': feature_name,
                                         'bad_percentage': bad_percentage,
                                         'bad_frames': bad_frames})

        self.insert(tracking_qc_list)



def poly_area(x, y):
    # https://stackoverflow.com/a/30408825
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def fit_ellipse(x, y):
    # Direct fit ellipse using least squares
    # https://stackoverflow.com/a/47881806
    
    X = x - np.mean(x)
    Y = y - np.mean(y)

    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X) * 100
    coeffs = np.linalg.lstsq(A, b)[0].squeeze()
    
    return [*coeffs, -100]


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """
    # https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        # print('coeffs do not represent an ellipse: b^2 - 4ac must'
        #                  ' be negative!')
        return [np.nan] * 6

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y