import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# world_plane 시각화 함수
def plot_plane(ax, normal_vector, d, plane_size=1.5, color='r', alpha=0.5):
    """
    평면을 시각화하는 함수
    Args:
        ax: Matplotlib의 Axes3D 객체
        normal_vector: 평면의 법선 벡터 (A, B, C)
        d: 평면 방정식의 D 값
        plane_size: 시각화할 평면의 크기
        color: 평면 색상
        alpha: 평면 투명도
    """
    xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10), 
                         np.linspace(-plane_size, plane_size, 10))
    A, B, C = normal_vector
    zz = -(A * xx + B * yy + d) / C
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)

# 포인트 클라우드 시각화 함수
def plot_point_cloud(ax, points, color='b'):
    """
    포인트 클라우드를 시각화하는 함수
    Args:
        ax: Matplotlib의 Axes3D 객체
        points: 포인트 클라우드 데이터 (Nx3)
        color: 포인트 색상
    """
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=1)

# 평면을 포인트 클라우드의 좌표계로 변환하는 함수
def transform_plane_to_point_cloud_coords(plane, point_cloud_mean, scale):
    """
    평면을 포인트 클라우드의 좌표계로 변환
    Args:
        plane (np.array): 평면 방정식의 파라미터 [A, B, C, D]
        point_cloud_mean (np.array): 포인트 클라우드의 평균 위치 (평면의 기준점을 맞추기 위해 필요)
        scale (float): 포인트 클라우드의 스케일 (스케일을 맞추기 위해 필요)
    Returns:
        transformed_plane (np.array): 포인트 클라우드 좌표계에 맞춘 평면 방정식
    """
    A, B, C = plane[:3]
    D = plane[3] / scale

    # 포인트 클라우드의 기준점(mean)을 반영하여 평면 방정식을 변환
    D_transformed = D - np.dot(plane[:3], point_cloud_mean) / scale

    return np.array([A, B, C, D_transformed])

# 포인트 클라우드의 스케일을 조정하는 함수
def scale_point_cloud(point_cloud, scale_factor):
    """
    포인트 클라우드의 스케일을 조정하는 함수
    Args:
        point_cloud (np.array): 포인트 클라우드 데이터 (Nx3)
        scale_factor (float): 스케일 팩터
    Returns:
        scaled_point_cloud (np.array): 스케일이 적용된 포인트 클라우드
    """
    return point_cloud * scale_factor

# 포인트 클라우드와 평면을 함께 시각화하는 함수
def visualize_object_with_plane(ply_file_path, world_plane_path, plane_size=1, scale_factor=1):
    """
    포인트 클라우드와 평면을 함께 시각화하는 함수
    Args:
        ply_file_path (str): .ply 파일 경로
        world_plane_path (str): .npy 파일 경로
        plane_size (float): 평면 크기
        scale_factor (float): 포인트 클라우드 스케일 팩터
    """
    # 포인트 클라우드 .ply 파일에서 로드
    mesh = trimesh.load(ply_file_path)
    points = np.array(mesh.vertices)  # 포인트 클라우드 데이터를 numpy 배열로 변환

    # 포인트 클라우드 스케일 조정
    points = scale_point_cloud(points, scale_factor)

    # 포인트 클라우드의 평균 위치 계산
    point_cloud_mean = points.mean(axis=0)

    # world_plane npy 파일에서 로드
    world_plane = np.load(world_plane_path)
    normal_vector = world_plane[:3]
    d = world_plane[3]

    # 평면을 포인트 클라우드의 좌표계로 변환
    transformed_plane = transform_plane_to_point_cloud_coords(world_plane, point_cloud_mean, scale_factor)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 포인트 클라우드 시각화
    plot_point_cloud(ax, points)

    # 변환된 평면 시각화
    plot_plane(ax, transformed_plane[:3], transformed_plane[3], plane_size=plane_size)

    # 시각화 설정
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

# # 예시 사용법
# ply_file_path = "../../../sds-complete/data_processing/redwood_dataset/point_clouds/09639.ply"
# world_plane_path = "../../../sds-complete/data_processing/redwood_dataset/world_planes/09639.npy"
ply_file_path = "../vol_dataset/ply_obj/23/23_23.ply"
world_plane_path = "23_23.npy"
# world_plane_path = "73_73_v3.npy"
# ply_file_path = "../../../sds-complete/data_processing/redwood_dataset/point_clouds/06188.ply"
# world_plane_path = "../../../sds-complete/data_processing/plane_test/plane.npy"

visualize_object_with_plane(ply_file_path, world_plane_path)

