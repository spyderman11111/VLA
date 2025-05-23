#%% imports

import rtde_control
import rtde_receive
import time
import threading


class URController():
    def __init__(self, ip="127.0.0.1"):
        # robot ip
        self.ip = ip
        # instantiate rtde control and receiver objects
        self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
        # print(dir(self.rtde_c))
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
        # set default speed and acceleration
        self.speed = .2
        self.acceleration = .2

        self._infreeDrive = False
        self._is_moving = False
        self._stop = False
        self._joint_positions = []

        self.thread = threading.Thread(target=self.track_positions)
        self.thread.daemon = True

    def start_thread(self):
        self.thread.start()

    def get_tcp_pose(self):
        tcp_pose = self.rtde_r.getActualTCPPose()
        return tcp_pose

    def get_current_joints_position(self) -> list:
        """
        """
        joints_position = self.rtde_r.getActualQ()
        return joints_position
    
    def set_speed(self, speed=.2) -> None:
        self.speed = speed
    
    def set_acceleration(self, acceleration=.2) -> None:
        self.acceleration = acceleration

    def move_joints(self, joints_position, speed, acceleration) -> None:
        self._is_moving = True
        print(f"Moving to target joints position {joints_position}.")
        self.rtde_c.moveJ(joints_position, speed=speed, acceleration=acceleration)
        self._is_moving = False

    def startFreeDrive(self) -> None:
        self.rtde_c.teachMode()
        self._infreeDrive = True

    def stopFreeDrive(self) -> None:
        self.rtde_c.endTeachMode()
        self._infreeDrive = False

    def inFreeDriveMode(self) -> bool:
        return self._infreeDrive
    
    def stopScript(self):
        self._stop = True
        self.rtde_c.stopScript()
        self.stopFreeDrive()
        if self.thread.is_alive():
            self.thread.join()

    def track_positions(self):
        while True:
            if self._stop:
                break

            if self._is_moving:
                position = self.get_current_joints_position()
                self._joint_positions.append(position)
                time.sleep(1)
            else:
                time.sleep(0.1)
    
    def get_interpolated_jp(self) -> list:
        """
        """
        return self._joint_positions


'''
if __name__ == '__main__':

    robot_ip = "172.18.6.23" # default local host (for ursim)
    joint_position = [0, -1.57, 4, 0, 4, 0] # target position

    ur_controller = URController(robot_ip)
    ur_controller.move_joints(joints_position=joint_position, speed=.2, acceleration=.2)
'''