# UR5 Startup Guide (ROS 2 + MoveIt)

## ✅ Step 1: Source the ROS Environment

```bash
source /opt/ros/humble/setup.bash
```

---

## ✅ Step 2: Launch the UR5 Driver Node

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=172.18.6.20
```

>  Replace `robot_ip` with the actual IP address of your UR5 control box.

---

## ✅ Step 3: Verify Controller Operation

```bash
ros2 launch ur_robot_driver test_scaled_joint_trajectory_controller.launch.py
```

> Successful motion confirms proper driver connectivity.

---

## ✅ Step 4: Execute the External Control Program on UR5

- Run your preconfigured External Control program on the UR5 Teach Pendant.
- Ensure the IP address in the program matches the host computer (currently `172.18.6.20`).

---

## ✅ Step 5: Launch MoveIt (Optional)

For motion planning and execution:

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5 \
  robot_ip:=172.18.6.20 \
  launch_rviz:=true
```

---

## ✅ Step 6: Inspect Active Controllers

To review active controllers via the controller manager:

```bash
ros2 control list_controllers
```

---

## ✅ Step 7: Plan and Execute via MoveIt

1. Click **Plan** in RViz to visualize the trajectory (purple path).
2. Confirm the path avoids collision with the workspace.
3. Click **Execute** to move the robot.

---

## 💡 Notes

- A red trajectory indicates collision or planning failure.
- The robot will not respond unless the External Control program is running and IP settings are correct.
