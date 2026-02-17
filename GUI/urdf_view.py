from pathlib import Path
import yourdfpy as urdf
import numpy as np
from trimesh.viewer import SceneViewer
import dearpygui.dearpygui as dpg
import threading
from robot_descriptions.loaders.yourdfpy import load_robot_description

# ----------------------------
# Paths
# ----------------------------
HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "cora_description_v2" / "cora.urdf"

# ----------------------------
# Load URDF
# ----------------------------
robot = load_robot_description("cora_description_v2", URDF_PATH.parent)
urdf_model = urdf.URDF.load(robot.URDF_PATH)

joint_names = urdf_model.actuated_joint_names
num_joints = urdf_model.num_actuated_joints
cfg = urdf_model.cfg.copy()

# ----------------------------
# Function to update URDF from sliders
# ----------------------------
def update_joint_callback(sender, app_data, user_data):
    joint_index = user_data
    cfg[joint_index] = app_data
    urdf_model.update_cfg(cfg)
    # update scene
# ----------------------------
# Create non-blocking viewer in separate thread
# ----------------------------
scene = urdf_model.scene

def run_viewer():
    # This starts the blocking viewer in its own thread
    SceneViewer(scene, start_loop=True, background=(0,0,0,0), use_raymond_lighting=True, callback_period=0.03)

viewer_thread = threading.Thread(target=run_viewer, daemon=True)
viewer_thread.start()

# ----------------------------
# Setup DearPyGUI sliders
# ----------------------------
dpg.create_context()
with dpg.window(label="Joint Control", width=300, height=50*num_joints):
    for i, joint_name in enumerate(joint_names):
        dpg.add_slider_float(
            label=joint_name,
            min_value=-3.14,
            max_value=3.14,
            default_value=cfg[i],
            callback=update_joint_callback,
            user_data=i
        )

dpg.create_viewport(title='Joint Control', width=300, height=50*num_joints)
dpg.setup_dearpygui()
dpg.show_viewport()

# ----------------------------
# Main GUI loop
# ----------------------------
try:
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
finally:
    dpg.destroy_context()
