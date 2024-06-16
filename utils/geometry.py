import math

import torch
import numpy as np
import autograd.numpy as at_np
from scipy.linalg import fractional_matrix_power
import copy
from autograd import grad

def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

'''
def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def matrix_to_axis_angle(R):
    #torch version
    R = R.double()
    sy = torch.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    alpha = torch.atan2(R[2,1] , R[2,2])
    beta = torch.atan2(-R[2,0], sy)
    gamma = torch.atan2(R[1,0], R[0,0])

    #check 
    print("convert Error: ", torch.norm(R - axis_angle_to_matrix(torch.tensor([alpha, beta, gamma]))))
    print("original: ", R, "converted: ", axis_angle_to_matrix(torch.tensor([alpha, beta, gamma])))
    return torch.tensor([alpha, beta, gamma]).float()
'''
#the above version get error when using negative determinant matrix
from scipy.spatial.transform import Rotation

def axis_angle_to_matrix(theta):
    '''
    R_x = torch.tensor([[1, 0, 0],
                    [0, torch.cos(theta[0]), -torch.sin(theta[0])],
                    [0, torch.sin(theta[0]), torch.cos(theta[0])]
                    ])
 
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])],
                    [0, 1, 0],
                    [-torch.sin(theta[1]), 0, torch.cos(theta[1])]
                    ])
 
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0],
                    [torch.sin(theta[2]), torch.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = torch.mm(R_z, torch.mm( R_y, R_x )).to(theta.device)
    return R
    '''
    #make this function differentiable
    R_x = torch.zeros((3,3), device=theta.device)
    R_y = torch.zeros((3,3), device=theta.device)
    R_z = torch.zeros((3,3), device=theta.device)
    R_x[0,0], R_y[1,1], R_z[2,2] = 1, 1, 1

    R_x[1, 1] = torch.cos(theta[0]) + R_x[1, 1]
    R_x[1, 2] = - torch.sin(theta[0]) + R_x[1, 2]
    R_x[2, 1] = torch.sin(theta[0]) + R_x[2, 1]
    R_x[2, 2] = torch.cos(theta[0]) + R_x[2, 2]

    R_y[0, 0] = torch.cos(theta[1]) + R_y[0, 0]
    R_y[0, 2] = torch.sin(theta[1]) + R_y[0, 2]
    R_y[2, 0] = - torch.sin(theta[1]) + R_y[2, 0]
    R_y[2, 2] = torch.cos(theta[1]) + R_y[2, 2]

    R_z[0, 0] = torch.cos(theta[2]) + R_z[0, 0]
    R_z[0, 1] = - torch.sin(theta[2]) + R_z[0, 1]
    R_z[1, 0] = torch.sin(theta[2]) + R_z[1, 0]
    R_z[1, 1] = torch.cos(theta[2]) + R_z[1, 1]
    R = torch.mm(R_z, torch.mm( R_y, R_x ))
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = torch.transpose(R, 0, 1)
    shouldBeIdentity = torch.mm(Rt, R)
    I = torch.eye(3, device=R.device)
    n = torch.norm(I - shouldBeIdentity)
    if n > 1e-4:
        print("isRotationMatrix Error: ", n)
    return n < 1e-4
 
def matrix_to_axis_angle(R) :
    
    thetax = torch.atan2(R[2,1] , R[2,2])
    thetay = torch.atan2(-R[2,0], torch.sqrt(R[2,1] * R[2,1] + R[2,2] * R[2,2]))
    thetaz = torch.atan2(R[1,0], R[0,0])
    theta = torch.tensor([thetax, thetay, thetaz])

    return theta.to(R.device)


def rigid_transform_Kabsch_3D_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    try:
        U, S, Vt = torch.linalg.svd(H)
    except:
        print("SVD Error: ", A, B)
        return torch.eye(3, device=A.device), centroid_B - centroid_A

    R = Vt.T @ U.T
    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

    #t = -R @ centroid_A + centroid_B
    t = centroid_B - centroid_A
    return R, t

def transform_norm_vector(vector_A, vector_B):
    vector_A, vector_B = vector_A.double(), vector_B.double()
    if torch.norm(vector_A[1] - vector_A[0]) < 1e-3 or torch.norm(vector_B[1] - vector_B[0]) < 1e-3:
        return torch.eye(3, device=vector_A.device).float(), (- vector_A[0] + vector_B[0]).float()
    norm_vec_A = (vector_A[1] - vector_A[0]) / torch.norm(vector_A[1] - vector_A[0])
    norm_vec_B = (vector_B[1] - vector_B[0]) / torch.norm(vector_B[1] - vector_B[0])
    
    cos = torch.dot(norm_vec_A, norm_vec_B)
    sin = torch.sqrt(1 - cos * cos)

    v = torch.cross(norm_vec_A, norm_vec_B) / torch.norm(torch.cross(norm_vec_A, norm_vec_B))

    rot_mat = torch.tensor([
        [v[0]**2 + (1 - v[0]**2) * cos, v[0] * v[1] * (1 - cos) - v[2] * sin, v[0] * v[2] * (1 - cos) + v[1] * sin],
        [v[0] * v[1] * (1 - cos) + v[2] * sin, v[1]**2 + (1 - v[1]**2) * cos, v[1] * v[2] * (1 - cos) - v[0] * sin],
        [v[0] * v[2] * (1 - cos) - v[1] * sin, v[1] * v[2] * (1 - cos) + v[0] * sin, v[2]**2 + (1 - v[2]**2) * cos]]
    )

    #not singlar matrix
    #rot_mat = Rotation.from_matrix(rot_mat.cpu().numpy()).as_matrix()
    #rot_mat = torch.tensor(rot_mat).to(vector_A.device).float()
    translation = - vector_A[0] + vector_B[0]
    #check nan in rot
    return rot_mat.float(), translation.float()

def angle_exp(angles, n):
    sign_n = np.sign(n)
    n = np.abs(n)
    rot_mat = axis_angle_to_matrix(angles).cpu().numpy()
    new_rot_mat = fractional_matrix_power(rot_mat, n).real
    if sign_n < 0:
        new_rot_mat = new_rot_mat.T
    new_rot_mat = torch.from_numpy(new_rot_mat)
    return matrix_to_axis_angle(new_rot_mat).to(angles.device)

def angle_sum(angles_A, angles_B):
    rot_mat_A = axis_angle_to_matrix(angles_A).cpu().numpy()
    rot_mat_B = axis_angle_to_matrix(angles_B).cpu().numpy()
    new_rot_mat = rot_mat_A @ rot_mat_B
    new_rot_mat = torch.from_numpy(new_rot_mat)
    return matrix_to_axis_angle(new_rot_mat).to(angles_A.device)

def rot_axis_to_matrix(v, angle):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    #rot_mat = torch.tensor([
    #    [v[0]**2 + (1 - v[0]**2) * cos, v[0] * v[1] * (1 - cos) - v[2] * sin, v[0] * v[2] * (1 - cos) + v[1] * sin],
    #    [v[0] * v[1] * (1 - cos) + v[2] * sin, v[1]**2 + (1 - v[1]**2) * cos, v[1] * v[2] * (1 - cos) - v[0] * sin],
    #    [v[0] * v[2] * (1 - cos) - v[1] * sin, v[1] * v[2] * (1 - cos) + v[0] * sin, v[2]**2 + (1 - v[2]**2) * cos]]
    #)
    #make this function differentiable
    rot_mat = torch.stack(
        [v[0]**2 + (1 - v[0]**2) * cos, v[0] * v[1] * (1 - cos) - v[2] * sin, v[0] * v[2] * (1 - cos) + v[1] * sin,
        v[0] * v[1] * (1 - cos) + v[2] * sin, v[1]**2 + (1 - v[1]**2) * cos, v[1] * v[2] * (1 - cos) - v[0] * sin,
        v[0] * v[2] * (1 - cos) - v[1] * sin, v[1] * v[2] * (1 - cos) + v[0] * sin, v[2]**2 + (1 - v[2]**2) * cos]
    )
    rot_mat = rot_mat.reshape(3,3)
    return rot_mat

def rot_axis_to_matrix_np(v, angle):
    cos = at_np.cos(angle)
    sin = at_np.sin(angle)
    rot_mat = at_np.stack(
        [v[0]**2 + (1 - v[0]**2) * cos, v[0] * v[1] * (1 - cos) - v[2] * sin, v[0] * v[2] * (1 - cos) + v[1] * sin,
        v[0] * v[1] * (1 - cos) + v[2] * sin, v[1]**2 + (1 - v[1]**2) * cos, v[1] * v[2] * (1 - cos) - v[0] * sin,
        v[0] * v[2] * (1 - cos) - v[1] * sin, v[1] * v[2] * (1 - cos) + v[0] * sin, v[2]**2 + (1 - v[2]**2) * cos]
    )
    rot_mat = rot_mat.reshape(3,3)
    return rot_mat

def loss_func(angle, points_A, points_B, axis, time):
    rot_mat = rot_axis_to_matrix_np(axis[1] - axis[0], angle)
    points_A = (points_A - axis[0]) @ rot_mat.T + axis[0]
    loss = ((points_A - points_B[0]) **2).mean() * (1 - time) + ((points_A - points_B[1]) **2).mean() * time
    return loss

def get_optimal_R(points_A, points_B, axis, time, iter=100, eps=1e-2, atol=0.1):
    device = points_A.device
    axis = copy.deepcopy(axis)
    axis[1] = axis[0] + (axis[1] - axis[0]) / torch.norm((axis[1] - axis[0]))
    opt_loss = 1e5
    decrease = 0
    points_A, points_B[0], points_B[1] = points_A.cpu().numpy(), points_B[0].cpu().numpy(), points_B[1].cpu().numpy()
    points_A, points_B[0], points_B[1] = at_np.array(points_A), at_np.array(points_B[0]), at_np.array(points_B[1])
    axis = at_np.array(axis.cpu().numpy())
    for _ in range(iter):
        loss_f = lambda angle: loss_func(angle, points_A, points_B, axis, time)
        angle_init = at_np.array(0.0)
        angle_grad = grad(loss_f)(angle_init)
        loss = loss_f(angle_init)
        if loss < opt_loss:
            opt_loss = loss
            decrease = 0
        else:
            decrease += 1
        if decrease > 10:
            break
        rot_mat_mod = rot_axis_to_matrix_np(axis[1] - axis[0], - angle_grad *eps)
        points_A = (points_A - axis[0]) @ rot_mat_mod.T + axis[0]
        if loss < atol:
            break
    points_A = torch.from_numpy(points_A).to(device).float()
    return points_A

def change_R_T_ref(tr, rot, ref, new_ref):
    #change the reference of the rotation and translation
    #tr: [3], rot:[3, 3], ref: [3], new_ref: [3]
    #the original transform is (x - ref) @ rot + ref + tr
    #the new transform is (x - new_ref) @ new_rot + new_ref + new_tr
    if isinstance(ref, np.ndarray):
        ref = torch.from_numpy(ref).float()
    if isinstance(new_ref, np.ndarray):
        new_ref = torch.from_numpy(new_ref).float()

    device = new_ref.device
    ref = torch.mean(ref, dim=0, keepdim=True)
    new_ref = torch.mean(new_ref, dim=0, keepdim=True)
    
    
    if tr.shape[0] == 1 and tr.shape[1] == 3:
        tr = tr.T
    elif len(tr.shape) == 1:
        tr = tr.unsqueeze(-1)
    new_tr = (new_ref - ref.to(device)) @ rot.to(device).T + tr.T.to(device) + ref.to(device) - new_ref
    
    return new_tr
