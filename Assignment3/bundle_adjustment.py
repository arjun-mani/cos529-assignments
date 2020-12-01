import numpy as np
import pickle
import argparse
import copy
import time

# Project point with index pointid onto camera with id camid
def project_point(solution, camid, pointid):
    point = solution['points'][pointid]
    P = solution['poses'][camid]

    # compute P' = RP + t
    pprime = np.dot(P[0:3, 0:3], point) + P[:, 3]

    # multiply by focal length and normalize by third element of P'
    pfinal = solution['focal_lengths'][camid] * pprime[0:2] / pprime[2]
    return (pfinal, pprime)

# Return cross product matrix of the vector
def cp_matrix(P):
    Phat = [[0, -P[2], P[1]], [P[2], 0, -P[0]], [-P[1], P[0], 0]]
    return np.array(Phat)

# Compute the error terms (2K-length vector)
def compute_errors(solution):
    dim = 2 * len(solution['observations'])
    eop = np.zeros(dim)

    for index, point in enumerate(solution['observations']):
        (pj_pt, pprime) = project_point(solution, point[0], point[1])
        eop[index*2] = point[2] - pj_pt[0]
        eop[index*2 + 1] = point[3] - pj_pt[1]

    return eop

def LM_terminate(eop, update):
    return False

def compute_Jacobian(solution):
    n_cameras = solution['poses'].shape[0]
    n_points = solution['points'].shape[0]
    n_observations = len(solution['observations'])

    # initialize Jacobian matrix
    Jop = np.zeros((2*n_observations, 6*n_cameras + 3*n_points))

    for index, point in enumerate(solution['observations']):
        camid = point[0]
        pointid = point[1]
        # The projected point is a composition of two functions
        # 1) P' = Rexp(psi^)P + t
        # 2) p = (x, y) = (f * X'/Z', f * Y'/Z')

        (pj_pt, pprime) = project_point(solution, camid, pointid)

        f = solution['focal_lengths'][camid]

        # We first compute (partialx / partialP', partialy / partialP')
        dxdPpr = [f/pprime[2], 0, -f*pprime[0] / (pprime[2]*pprime[2])]
        dydPpr = [0, f/pprime[2], -f*pprime[1] / (pprime[2]*pprime[2])]
        dxdPpr, dydPpr = np.array(dxdPpr), np.array(dydPpr)

        # Compute the derivatives with respect to the translation vector
        # since partialP'/partialt = I, we can just use the same derivatives
        dxdt = dxdPpr
        dydt = dydPpr

        # Compute the derivative with respect to the points
        # partialP' / partialP = R, since we takt the derivative at psi = 0
        R = solution['poses'][camid]
        R = R[0:3, 0:3]
        dxdP = np.dot(dxdPpr, R)
        dydP = np.dot(dydPpr, R)

        # compute the derivative with respect to the rotation matrix
        # partialP' / partialpsi = -RP^ at psi = 0, where P^ = the cross product matrix of P
        P = solution['points'][pointid]
        Phat = cp_matrix(P)
        dPprdpsi = np.dot(-R, Phat)
        dxdpsi = np.dot(dxdPpr, dPprdpsi)
        dydpsi = np.dot(dydPpr, dPprdpsi)

        # find the location at which to insert the partial derivatives for the camera coordinates
        cam_loc = 6 * camid

        # partial derivatives w.r.t x coordinate
        Jop[2*index, cam_loc:cam_loc+3] = dxdpsi
        Jop[2*index, cam_loc+3:cam_loc+6] = dxdt

        # partial derivatives w.r.t y coordinate
        Jop[2*index+1, cam_loc:cam_loc+3] = dydpsi
        Jop[2*index+1, cam_loc+3:cam_loc+6] = dydt

        # find the location at which to insert the partial derivatives for the point
        pt_loc = 6 * n_cameras + 3 * pointid

        # insert for x and y coordinate
        Jop[2*index, pt_loc:pt_loc+3] = dxdP
        Jop[2*index+1, pt_loc:pt_loc+3] = dydP

    return Jop

def compute_Hb_calibrated(solution, eop):

    # compute JTJ incrementally by computing each row of J
    n_cameras = solution['poses'].shape[0]
    n_points = solution['points'].shape[0]
    n_observations = len(solution['observations'])

    JTJ = np.zeros((6*n_cameras + 3*n_points,  6*n_cameras + 3*n_points))
    JTe = np.zeros(6*n_cameras + 3*n_points)

    for index, point in enumerate(solution['observations']):
        Jop_row1 = np.zeros(6*n_cameras + 3*n_points)
        Jop_row2 = np.zeros(6*n_cameras + 3*n_points)

        camid = point[0]
        pointid = point[1]
        # The projected point is a composition of two functions
        # 1) P' = Rexp(psi^)P + t
        # 2) p = (x, y) = (f * X'/Z', f * Y'/Z')

        (pj_pt, pprime) = project_point(solution, camid, pointid)

        f = solution['focal_lengths'][camid]

        # We first compute (partialx / partialP', partialy / partialP')
        dxdPpr = [f/pprime[2], 0, -f*pprime[0] / (pprime[2]*pprime[2])]
        dydPpr = [0, f/pprime[2], -f*pprime[1] / (pprime[2]*pprime[2])]
        dxdPpr, dydPpr = np.array(dxdPpr), np.array(dydPpr)

        # Compute the derivatives with respect to the translation vector
        # since partialP'/partialt = I, we can just use the same derivatives
        dxdt = dxdPpr
        dydt = dydPpr

        # Compute the derivative with respect to the points
        # partialP' / partialP = R, since we takt the derivative at psi = 0
        R = solution['poses'][camid]
        R = R[0:3, 0:3]
        dxdP = np.dot(dxdPpr, R)
        dydP = np.dot(dydPpr, R)

        # compute the derivative with respect to the rotation matrix
        # partialP' / partialpsi = -RP^ at psi = 0, where P^ = the cross product matrix of P
        P = solution['points'][pointid]
        Phat = cp_matrix(P)
        dPprdpsi = np.dot(-R, Phat)
        dxdpsi = np.dot(dxdPpr, dPprdpsi)
        dydpsi = np.dot(dydPpr, dPprdpsi)

        # find the location at which to insert the partial derivatives for the camera coordinates
        cam_loc = 6 * camid

        # partial derivatives w.r.t x coordinate
        Jop_row1[cam_loc:cam_loc+3] = dxdpsi
        Jop_row1[cam_loc+3:cam_loc+6] = dxdt

        # partial derivatives w.r.t y coordinate
        Jop_row2[cam_loc:cam_loc+3] = dydpsi
        Jop_row2[cam_loc+3:cam_loc+6] = dydt

        # find the location at which to insert the partial derivatives for the point
        pt_loc = 6 * n_cameras + 3 * pointid

        # insert for x and y coordinate
        Jop_row1[pt_loc:pt_loc+3] = dxdP
        Jop_row2[pt_loc:pt_loc+3] = dydP

        # update JTe with the newly computed row
        JTe += eop[2*index] * Jop_row1
        JTe += eop[2*index+1] * Jop_row2 
        
        # fill in update matrices (np.dot takes too long, and we know what they look like)
        cam_derivx = np.reshape(np.hstack([dxdpsi, dxdt]), (-1, 1))
        pt_derivx = np.reshape(dxdP, (-1, 1))
        JTJ[cam_loc:cam_loc+6, cam_loc:cam_loc+6] += np.dot(cam_derivx, cam_derivx.T)
        JTJ[cam_loc:cam_loc+6, pt_loc:pt_loc+3] += np.dot(cam_derivx, pt_derivx.T)
        JTJ[pt_loc:pt_loc+3, cam_loc:cam_loc+6] += np.dot(pt_derivx, cam_derivx.T)
        JTJ[pt_loc:pt_loc+3, pt_loc:pt_loc+3] += np.dot(pt_derivx, pt_derivx.T)

        cam_derivy = np.reshape(np.hstack([dydpsi, dydt]), (-1, 1))
        pt_derivy = np.reshape(dydP, (-1, 1))
        JTJ[cam_loc:cam_loc+6, cam_loc:cam_loc+6] += np.dot(cam_derivy, cam_derivy.T)
        JTJ[cam_loc:cam_loc+6, pt_loc:pt_loc+3] += np.dot(cam_derivy, pt_derivy.T)
        JTJ[pt_loc:pt_loc+3, cam_loc:cam_loc+6] += np.dot(pt_derivy, cam_derivy.T)
        JTJ[pt_loc:pt_loc+3, pt_loc:pt_loc+3] += np.dot(pt_derivy, pt_derivy.T)

    return (JTJ, JTe)

def compute_Hb_uncalibrated(solution, eop):
    # compute JTJ incrementally by computing each row of J
    n_cameras = solution['poses'].shape[0]
    n_points = solution['points'].shape[0]
    n_observations = len(solution['observations'])

    JTJ = np.zeros((7*n_cameras + 3*n_points,  7*n_cameras + 3*n_points))
    JTe = np.zeros(7*n_cameras + 3*n_points)

    for index, point in enumerate(solution['observations']):
        Jop_row1 = np.zeros(7*n_cameras + 3*n_points)
        Jop_row2 = np.zeros(7*n_cameras + 3*n_points)

        camid = point[0]
        pointid = point[1]
        # The projected point is a composition of two functions
        # 1) P' = Rexp(psi^)P + t
        # 2) p = (x, y) = (f * X'/Z', f * Y'/Z')

        (pj_pt, pprime) = project_point(solution, camid, pointid)

        f = solution['focal_lengths'][camid]

        # We first compute (partialx / partialP', partialy / partialP')
        dxdPpr = [f/pprime[2], 0, -f*pprime[0] / (pprime[2]*pprime[2])]
        dydPpr = [0, f/pprime[2], -f*pprime[1] / (pprime[2]*pprime[2])]
        dxdPpr, dydPpr = np.array(dxdPpr), np.array(dydPpr)

        # Compute the derivatives with respect to the translation vector
        # since partialP'/partialt = I, we can just use the same derivatives
        dxdt = dxdPpr
        dydt = dydPpr

        # Compute the derivative with respect to the points
        # partialP' / partialP = R, since we takt the derivative at psi = 0
        R = solution['poses'][camid]
        R = R[0:3, 0:3]
        dxdP = np.dot(dxdPpr, R)
        dydP = np.dot(dydPpr, R)

        # compute the derivative with respect to the rotation matrix
        # partialP' / partialpsi = -RP^ at psi = 0, where P^ = the cross product matrix of P
        P = solution['points'][pointid]
        Phat = cp_matrix(P)
        dPprdpsi = np.dot(-R, Phat)
        dxdpsi = np.dot(dxdPpr, dPprdpsi)
        dydpsi = np.dot(dydPpr, dPprdpsi)

        # compute the derivative with respect to the focal length
        dxdf = pprime[0]/pprime[2]
        dydf = pprime[1]/pprime[2]

        # find the location at which to insert the partial derivatives for the camera coordinates
        cam_loc = 7 * camid

        # partial derivatives w.r.t x coordinate
        Jop_row1[cam_loc:cam_loc+3] = dxdpsi
        Jop_row1[cam_loc+3:cam_loc+6] = dxdt
        Jop_row1[cam_loc+6] = dxdf

        # partial derivatives w.r.t y coordinate
        Jop_row2[cam_loc:cam_loc+3] = dydpsi
        Jop_row2[cam_loc+3:cam_loc+6] = dydt
        Jop_row2[cam_loc+6] = dydf

        # find the location at which to insert the partial derivatives for the point
        pt_loc = 7 * n_cameras + 3 * pointid

        # insert for x and y coordinate
        Jop_row1[pt_loc:pt_loc+3] = dxdP
        Jop_row2[pt_loc:pt_loc+3] = dydP

        # update JTe with the newly computed row
        JTe += eop[2*index] * Jop_row1
        JTe += eop[2*index+1] * Jop_row2 

        # fill in update matrices (np.dot takes too long, and we know what they look like)
        cam_derivx = np.reshape(np.hstack([dxdpsi, dxdt, dxdf]), (-1, 1))
        pt_derivx = np.reshape(dxdP, (-1, 1))
        JTJ[cam_loc:cam_loc+7, cam_loc:cam_loc+7] += np.dot(cam_derivx, cam_derivx.T)
        JTJ[cam_loc:cam_loc+7, pt_loc:pt_loc+3] += np.dot(cam_derivx, pt_derivx.T)
        JTJ[pt_loc:pt_loc+3, cam_loc:cam_loc+7] += np.dot(pt_derivx, cam_derivx.T)
        JTJ[pt_loc:pt_loc+3, pt_loc:pt_loc+3] += np.dot(pt_derivx, pt_derivx.T)

        cam_derivy = np.reshape(np.hstack([dydpsi, dydt, dydf]), (-1, 1))
        pt_derivy = np.reshape(dydP, (-1, 1))
        JTJ[cam_loc:cam_loc+7, cam_loc:cam_loc+7] += np.dot(cam_derivy, cam_derivy.T)
        JTJ[cam_loc:cam_loc+7, pt_loc:pt_loc+3] += np.dot(cam_derivy, pt_derivy.T)
        JTJ[pt_loc:pt_loc+3, cam_loc:cam_loc+7] += np.dot(pt_derivy, cam_derivy.T)
        JTJ[pt_loc:pt_loc+3, pt_loc:pt_loc+3] += np.dot(pt_derivy, pt_derivy.T)

    return (JTJ, JTe)

def solve_LM(solution, eop, lambd):

    if(solution['is_calibrated']):
        (H, b) = compute_Hb_calibrated(solution, eop)

    else:
        (H, b) = compute_Hb_uncalibrated(solution, eop)

    I = np.eye(H.shape[0])
    H = H + lambd * I

    update = np.linalg.solve(H, b)
    return update

def perform_update_calibrated(solution, update):

    n_cameras = solution['poses'].shape[0]
    n_points = solution['points'].shape[0]

    # update parameters for all the cameras
    for i in range(0, n_cameras):
        curr_index = 6*i

        # get updates for the ith camera
        psi = update[curr_index:curr_index+3]
        t_update = update[curr_index+3:curr_index+6]

        psi = np.array(psi)
        psi = np.expand_dims(psi, axis=1)
        t_update = np.array(t_update)

        # have to actually compute this using the Rodrigues formula
        # R = (cos theta)I + (1 - cos theta)phi*phiT + sin theta * \phi^
        theta = np.linalg.norm(psi)+0.000001
        phi = psi / theta
        cos = np.cos(theta)
        sin = np.sin(theta)
        R_update = cos * np.eye(3) + (1-cos) * np.dot(phi, phi.T) + sin * cp_matrix(phi)

        # update for translation vector
        solution['poses'][i][:, 3] = solution['poses'][i][:, 3] + t_update

        # update for rotation matrix
        solution['poses'][i][0:3, 0:3] = np.dot(solution['poses'][i][0:3, 0:3], R_update)

    # update all the points
    for i in range(0, n_points):
        curr_index = 6*n_cameras + 3*i

        pt_update = update[curr_index:curr_index+3]

        solution['points'][i] = solution['points'][i] + pt_update

def perform_update_uncalibrated(solution, update):
    n_cameras = solution['poses'].shape[0]
    n_points = solution['points'].shape[0]

    # update parameters for all the cameras
    for i in range(0, n_cameras):
        curr_index = 7*i

        # get updates for the ith camera
        psi = update[curr_index:curr_index+3]
        t_update = update[curr_index+3:curr_index+6]
        f_update = update[curr_index+6]

        psi = np.array(psi)
        psi = np.expand_dims(psi, axis=1)
        t_update = np.array(t_update)

        # have to actually compute this using the Rodrigues formula
        # R = (cos theta)I + (1 - cos theta)phi*phiT + sin theta * \phi^
        theta = np.linalg.norm(psi)+0.000001
        phi = psi / theta
        cos = np.cos(theta)
        sin = np.sin(theta)
        R_update = cos * np.eye(3) + (1-cos) * np.dot(phi, phi.T) + sin * cp_matrix(phi)

        # update for translation vector
        solution['poses'][i][:, 3] += t_update

        # update for rotation matrix
        solution['poses'][i][0:3, 0:3] = np.dot(solution['poses'][i][0:3, 0:3], R_update)

        # update for focal length
        solution['focal_lengths'][i] += f_update

    # update all the points
    for i in range(0, n_points):
        curr_index = 7*n_cameras + 3*i

        pt_update = update[curr_index:curr_index+3]

        solution['points'][i] += pt_update

def perform_update(solution, update):
    if(solution['is_calibrated']):
        perform_update_calibrated(solution, update)

    else:
        perform_update_uncalibrated(solution, update)


def solve_ba_problem(problem):
    '''
    Solves the bundle adjustment problem defined by "problem" dict

    Input:
        problem: bundle adjustment problem containing the following fields:
            - is_calibrated: boolean, whether or not the problem is calibrated
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        solution: dictionary containing the problem, with the following fields updated
            - poses: [n_cameras,3,4] numpy array of optimized camera extrinsics
            - points: [n_points,3] numpy array of optimized 3d points
            - (if is_calibrated==False) then focal lengths should be optimized too
                focal_lengths: [n_cameras] numpy array with optimized focal focal_lengths

    Your implementation should optimize over the following variables to minimize reprojection error
        - problem['poses']
        - problem['points']
        - problem['focal_lengths']: if (is_calibrated==False)

    '''

    solution = problem
    # YOUR CODE STARTS

    lambd = 1
    prev_loss = 1000000
    update_thresh = 0.0001

    # Levenberg-Marquandt optimization
    for i in range(0, 50):

        # compute error terms across the N cameras and M points
        eop = compute_errors(solution)

        # Adjust lambda based on whether previous loss is less than current loss
        # if loss increases, discard updates and double lambda
        curr_loss = np.linalg.norm(eop)
        if(prev_loss < curr_loss):
            solution = prev_sol
            lambd = lambd * 2

        # otherwise, halve lambda
        else:
            lambd = lambd / 2

        # Solve for the LM update
        update = solve_LM(solution, eop, lambd)

        # LM stop condition: if the update is less than a threshold
        if(np.linalg.norm(update) < update_thresh):
            print("Terminating algorithm.")
            break

        # Perform the update for all the variables
        prev_sol = copy.deepcopy(solution)
        perform_update(solution, update)

        prev_loss = curr_loss

    return solution

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="config file")
    args = parser.parse_args()

    problem = pickle.load(open(args.problem, 'rb'))

    solution = solve_ba_problem(problem)

    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
