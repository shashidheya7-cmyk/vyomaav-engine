"""
VYOMAAV Base Model Engine
Module: engine.ros2

ROS2 Gazebo Robotics Bridge & Joint Dynamics Compiler (Sprint 17).
Transforms SOMG scene graphs into:
1. Valid Unified Robot Description Format (URDF) XML trees.
2. ROS2 Gazebo simulation environments (.world files).
3. Complete deployable ROS2 workspace packages (URDF, Gazebo launch scripts, CMakeLists, package.xml).
"""

from dataclasses import dataclass, field
import json
import math
import os
from typing import List, Dict, Tuple, Optional, Any, Set
import xml.dom.minidom
import xml.etree.ElementTree as ET

from somg.scene import SceneState
from somg.entity import SOMGEntity
from somg.graph import RelationType


@dataclass
class InertiaTensor:
    """Rigid body 3D rotational inertia tensor components."""
    ixx: float = 1.0
    iyy: float = 1.0
    izz: float = 1.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    @classmethod
    def compute_box_inertia(cls, mass_kg: float, dx: float, dy: float, dz: float) -> "InertiaTensor":
        """Computes principal inertia moments for a solid rectangular box."""
        m = max(1e-4, mass_kg)
        dx, dy, dz = max(1e-3, dx), max(1e-3, dy), max(1e-3, dz)
        ixx = (1.0 / 12.0) * m * (dy**2 + dz**2)
        iyy = (1.0 / 12.0) * m * (dx**2 + dz**2)
        izz = (1.0 / 12.0) * m * (dx**2 + dy**2)
        return cls(ixx=ixx, iyy=iyy, izz=izz)


@dataclass
class URDFLink:
    """URDF <link> visual, collision, and inertial specifications."""
    name: str
    mass_kg: float
    inertia: InertiaTensor
    xyz: List[float]  # Centroid position [x, y, z]
    rpy: List[float]  # Euler angles [roll, pitch, yaw]
    size_box: List[float]  # Dimensions [dx, dy, dz]
    color_rgba: List[float] = field(default_factory=lambda: [0.7, 0.7, 0.7, 1.0])
    mesh_ref: Optional[str] = None


@dataclass
class URDFJoint:
    """URDF <joint> structural connection specification."""
    name: str
    joint_type: str  # fixed, revolute, continuous, prismatic
    parent_link: str
    child_link: str
    origin_xyz: List[float]  # [x, y, z]
    origin_rpy: List[float]  # [roll, pitch, yaw]
    axis_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    limit_effort: float = 100.0
    limit_velocity: float = 3.0
    limit_lower: float = -1.57
    limit_upper: float = 1.57
    damping: float = 0.1
    friction: float = 0.1


@dataclass
class URDFRobot:
    """URDF Robot assembly model container."""
    name: str
    links: List[URDFLink] = field(default_factory=list)
    joints: List[URDFJoint] = field(default_factory=list)

    def to_urdf_xml(self) -> str:
        """Serializes assembly to formatted XML string."""
        robot_elem = ET.Element("robot", name=self.name)

        # Build Links
        for link in self.links:
            link_elem = ET.SubElement(robot_elem, "link", name=link.name)

            # Inertial
            inertial = ET.SubElement(link_elem, "inertial")
            ET.SubElement(
                inertial,
                "origin",
                xyz=f"{link.xyz[0]:.4f} {link.xyz[1]:.4f} {link.xyz[2]:.4f}",
                rpy=f"{link.rpy[0]:.4f} {link.rpy[1]:.4f} {link.rpy[2]:.4f}"
            )
            ET.SubElement(inertial, "mass", value=f"{max(1e-3, link.mass_kg):.4f}")
            ET.SubElement(
                inertial,
                "inertia",
                ixx=f"{link.inertia.ixx:.6f}",
                ixy=f"{link.inertia.ixy:.6f}",
                ixz=f"{link.inertia.ixz:.6f}",
                iyy=f"{link.inertia.iyy:.6f}",
                iyz=f"{link.inertia.iyz:.6f}",
                izz=f"{link.inertia.izz:.6f}"
            )

            # Visual
            visual = ET.SubElement(link_elem, "visual")
            ET.SubElement(
                visual,
                "origin",
                xyz=f"{link.xyz[0]:.4f} {link.xyz[1]:.4f} {link.xyz[2]:.4f}",
                rpy=f"{link.rpy[0]:.4f} {link.rpy[1]:.4f} {link.rpy[2]:.4f}"
            )
            geometry = ET.SubElement(visual, "geometry")
            if link.mesh_ref:
                ET.SubElement(geometry, "mesh", filename=link.mesh_ref)
            else:
                ET.SubElement(
                    geometry,
                    "box",
                    size=f"{link.size_box[0]:.4f} {link.size_box[1]:.4f} {link.size_box[2]:.4f}"
                )
            material = ET.SubElement(visual, "material", name=f"{link.name}_mat")
            ET.SubElement(
                material,
                "color",
                rgba=f"{link.color_rgba[0]:.2f} {link.color_rgba[1]:.2f} {link.color_rgba[2]:.2f} {link.color_rgba[3]:.2f}"
            )

            # Collision
            collision = ET.SubElement(link_elem, "collision")
            ET.SubElement(
                collision,
                "origin",
                xyz=f"{link.xyz[0]:.4f} {link.xyz[1]:.4f} {link.xyz[2]:.4f}",
                rpy=f"{link.rpy[0]:.4f} {link.rpy[1]:.4f} {link.rpy[2]:.4f}"
            )
            col_geom = ET.SubElement(collision, "geometry")
            ET.SubElement(
                col_geom,
                "box",
                size=f"{link.size_box[0]:.4f} {link.size_box[1]:.4f} {link.size_box[2]:.4f}"
            )

        # Build Joints
        for joint in self.joints:
            joint_elem = ET.SubElement(robot_elem, "joint", name=joint.name, type=joint.joint_type)
            ET.SubElement(joint_elem, "parent", link=joint.parent_link)
            ET.SubElement(joint_elem, "child", link=joint.child_link)
            ET.SubElement(
                joint_elem,
                "origin",
                xyz=f"{joint.origin_xyz[0]:.4f} {joint.origin_xyz[1]:.4f} {joint.origin_xyz[2]:.4f}",
                rpy=f"{joint.origin_rpy[0]:.4f} {joint.origin_rpy[1]:.4f} {joint.origin_rpy[2]:.4f}"
            )

            if joint.joint_type in ("revolute", "prismatic", "continuous"):
                ET.SubElement(
                    joint_elem,
                    "axis",
                    xyz=f"{joint.axis_xyz[0]:.2f} {joint.axis_xyz[1]:.2f} {joint.axis_xyz[2]:.2f}"
                )
            if joint.joint_type in ("revolute", "prismatic"):
                ET.SubElement(
                    joint_elem,
                    "limit",
                    lower=f"{joint.limit_lower:.4f}",
                    upper=f"{joint.limit_upper:.4f}",
                    effort=f"{joint.limit_effort:.2f}",
                    velocity=f"{joint.limit_velocity:.2f}"
                )
            ET.SubElement(
                joint_elem,
                "dynamics",
                damping=f"{joint.damping:.4f}",
                friction=f"{joint.friction:.4f}"
            )

        # Pretty Print XML
        raw_xml = ET.tostring(robot_elem, encoding="utf-8")
        parsed = xml.dom.minidom.parseString(raw_xml)
        return parsed.toprettyxml(indent="  ")


class ROS2GazeboBridge:
    """Compiles SOMG scene graphs into ROS2 workspace packages and Gazebo world files."""

    @classmethod
    def somg_to_urdf_robot(cls, scene: SceneState, robot_name: Optional[str] = None) -> URDFRobot:
        """Transforms SOMG entities and relationship edges into a jointed URDFRobot assembly."""
        graph = scene.resolve_active_graph()
        r_name = robot_name or f"{scene.scene_id}_robot"
        robot = URDFRobot(name=r_name)

        if len(graph.nodes) == 0:
            # Fallback world link
            dummy_link = URDFLink(
                name="world_base",
                mass_kg=1.0,
                inertia=InertiaTensor(),
                xyz=[0.0, 0.0, 0.0],
                rpy=[0.0, 0.0, 0.0],
                size_box=[1.0, 1.0, 1.0]
            )
            robot.links.append(dummy_link)
            return robot

        # Map entities to URDFLinks
        for entity_id, entity in graph.nodes.items():
            b_min = entity.spatial.bbox_min
            b_max = entity.spatial.bbox_max

            dx = max(0.01, b_max[0] - b_min[0])
            dy = max(0.01, b_max[1] - b_min[1])
            dz = max(0.01, b_max[2] - b_min[2])

            cx = (b_min[0] + b_max[0]) / 2.0
            cy = (b_min[1] + b_max[1]) / 2.0
            cz = (b_min[2] + b_max[2]) / 2.0

            mass = entity.physics.mass_kg if entity.physics else 1.0
            inertia = InertiaTensor.compute_box_inertia(mass, dx, dy, dz)

            # Material color selection
            mat_colors = {
                "metal": [0.7, 0.7, 0.75, 1.0],
                "wood": [0.6, 0.4, 0.2, 1.0],
                "fabric": [0.2, 0.4, 0.8, 1.0],
                "generic": [0.5, 0.5, 0.5, 1.0]
            }
            color = mat_colors.get(entity.material.material_type, mat_colors["generic"])

            link = URDFLink(
                name=entity_id,
                mass_kg=mass,
                inertia=inertia,
                xyz=[cx, cy, cz],
                rpy=[0.0, 0.0, 0.0],
                size_box=[dx, dy, dz],
                color_rgba=color
            )
            robot.links.append(link)

        # Build URDFJoints from Spatial Graph Edges
        joint_idx = 0
        for src_id, edges in graph.outgoing_edges.items():
            for edge in edges:
                if edge.target_id in graph.nodes:
                    joint_name = f"joint_{src_id}_to_{edge.target_id}_{joint_idx}"
                    joint_idx += 1

                    j_type = "fixed"
                    if edge.relation_type == RelationType.ATTACHED_TO:
                        j_type = "revolute"
                    elif edge.relation_type == RelationType.SUPPORTED_BY:
                        j_type = "fixed"

                    joint = URDFJoint(
                        name=joint_name,
                        joint_type=j_type,
                        parent_link=edge.target_id,  # Target is parent in supported_by/attached
                        child_link=src_id,
                        origin_xyz=[0.0, 0.0, 0.0],
                        origin_rpy=[0.0, 0.0, 0.0],
                        damping=0.1,
                        friction=0.1
                    )
                    robot.joints.append(joint)

        return robot

    @classmethod
    def generate_gazebo_world_xml(cls, scene: SceneState) -> str:
        """Generates ROS2 Gazebo SDF simulation world XML string."""
        sdf_elem = ET.Element("sdf", version="1.6")
        world_elem = ET.SubElement(sdf_elem, "world", name=f"{scene.scene_id}_world")

        # Physics engine configuration
        physics = ET.SubElement(world_elem, "physics", type="ode")
        ET.SubElement(physics, "max_step_size").text = "0.001"
        ET.SubElement(physics, "real_time_factor").text = "1.0"
        ET.SubElement(physics, "real_time_update_rate").text = "1000"

        # Sun directional light
        sun = ET.SubElement(world_elem, "include")
        ET.SubElement(sun, "uri").text = "model://sun"

        # Ground plane
        ground = ET.SubElement(world_elem, "include")
        ET.SubElement(ground, "uri").text = "model://ground_plane"

        # Add inline models for SOMG entities
        graph = scene.resolve_active_graph()
        for entity_id, entity in graph.nodes.items():
            model = ET.SubElement(world_elem, "model", name=entity_id)
            ET.SubElement(model, "static").text = "true" if entity.physics.is_static else "false"

            b_min = entity.spatial.bbox_min
            b_max = entity.spatial.bbox_max
            cx = (b_min[0] + b_max[0]) / 2.0
            cy = (b_min[1] + b_max[1]) / 2.0
            cz = (b_min[2] + b_max[2]) / 2.0

            ET.SubElement(model, "pose").text = f"{cx:.4f} {cy:.4f} {cz:.4f} 0 0 0"

            link = ET.SubElement(model, "link", name="link")
            visual = ET.SubElement(link, "visual", name="visual")
            geom = ET.SubElement(visual, "geometry")
            dx = max(0.01, b_max[0] - b_min[0])
            dy = max(0.01, b_max[1] - b_min[1])
            dz = max(0.01, b_max[2] - b_min[2])
            ET.SubElement(geom, "box").text = f"<size>{dx:.4f} {dy:.4f} {dz:.4f}</size>"

        raw_xml = ET.tostring(sdf_elem, encoding="utf-8")
        parsed = xml.dom.minidom.parseString(raw_xml)
        return parsed.toprettyxml(indent="  ")

    @classmethod
    def export_ros2_package(
        cls,
        scene: SceneState,
        package_name: str,
        output_dir: str
    ) -> Dict[str, str]:
        """Exports complete deployable ROS2 workspace package file structure."""
        pkg_dir = os.path.join(output_dir, package_name)
        urdf_dir = os.path.join(pkg_dir, "urdf")
        worlds_dir = os.path.join(pkg_dir, "worlds")
        launch_dir = os.path.join(pkg_dir, "launch")

        os.makedirs(urdf_dir, exist_ok=True)
        os.makedirs(worlds_dir, exist_ok=True)
        os.makedirs(launch_dir, exist_ok=True)

        # 1. Generate URDF File
        urdf_robot = cls.somg_to_urdf_robot(scene, robot_name=package_name)
        urdf_xml = urdf_robot.to_urdf_xml()
        urdf_filepath = os.path.join(urdf_dir, f"{package_name}.urdf")
        with open(urdf_filepath, "w") as f:
            f.write(urdf_xml)

        # 2. Generate Gazebo .world File
        world_xml = cls.generate_gazebo_world_xml(scene)
        world_filepath = os.path.join(worlds_dir, f"{package_name}.world")
        with open(world_filepath, "w") as f:
            f.write(world_xml)

        # 3. Generate ROS2 Gazebo Launch Script (Python)
        launch_py = f"""import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('{package_name}')
    world_file = os.path.join(pkg_dir, 'worlds', '{package_name}.world')
    urdf_file = os.path.join(pkg_dir, 'urdf', '{package_name}.urdf')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
            ]),
            launch_arguments={{'world': world_file}}.items()
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            arguments=[urdf_file]
        )
    ])
"""
        launch_filepath = os.path.join(launch_dir, "gazebo.launch.py")
        with open(launch_filepath, "w") as f:
            f.write(launch_py)

        # 4. Generate package.xml
        package_xml = f"""<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>1.0.0</version>
  <description>VYOMAAV Compiled ROS2 Gazebo Simulation Package</description>
  <maintainer email="dev@vyomaav.ai">VYOMAAV Engine</maintainer>
  <license>Proprietary</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
"""
        pkg_xml_filepath = os.path.join(pkg_dir, "package.xml")
        with open(pkg_xml_filepath, "w") as f:
            f.write(package_xml)

        # 5. Generate CMakeLists.txt
        cmakelists = f"""cmake_minimum_required(VERSION 3.8)
project({package_name})

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch urdf worlds
  DESTINATION share/${{PROJECT_NAME}}
)

ament_package()
"""
        cmake_filepath = os.path.join(pkg_dir, "CMakeLists.txt")
        with open(cmake_filepath, "w") as f:
            f.write(cmakelists)

        return {
            "package_dir": pkg_dir,
            "urdf": urdf_filepath,
            "world": world_filepath,
            "launch": launch_filepath,
            "package_xml": pkg_xml_filepath,
            "cmakelists": cmake_filepath
        }