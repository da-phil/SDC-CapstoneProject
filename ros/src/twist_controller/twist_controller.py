import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity,
                 brake_deadband, decel_limit, accel_limit,
                 wheel_radius, wheel_base, steer_ratio,
                 max_lat_accel, max_steer_angle):

        self.throttle_kp = 0.6
        self.throttle_ki = 0.1
        self.throttle_kd = 0.2
        self.throttle_min = 0.0 # min. throttle value
        self.throttle_max = 0.5 # max. throttle value
        self.throttle_controller = PID(self.throttle_kp, self.throttle_ki, self.throttle_kd,
                                       self.throttle_min, self.throttle_max)

        min_speed = 10.
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed,
                                            max_lat_accel, max_steer_angle)

        self.steer_kp = 0.8
        self.steer_ki = 0.1
        self.steer_kd = 0.3
        self.steer_min = -1.0 # min. steer value
        self.steer_max = 1.0 # max. steer value
        self.steer_controller = PID(self.steer_kp, self.steer_ki, self.steer_kd,
                                    self.steer_min, self.steer_max)


        tau = 0.01 # 1/(2*pi*tau) = filter cutoff frequency
        ts = 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.last_time = rospy.get_time()


    def control(self, current_vel, dbw_enable, linear_vel_desired, angular_vel_desired):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enable:
            self.throttle_controller.reset()
            self.steer_controller.reset()

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        #current_vel = self.vel_lpf.filt(current_vel)
        #rospy.logwarn("Cur velocity value: {}".format(current_vel))

        steering = self.yaw_controller.get_steering(linear_vel_desired, angular_vel_desired, current_vel)
        steering = self.steer_controller.step(steering, sample_time)

        #corrective_steer = self.steer_controller.step(angular_vel_desired, sample_time)
        #predictive_steer = self.yaw_controller.get_steering(linear_vel_desired, angular_vel_desired, current_vel)
        #steering = corrective_steer + predictive_steer
        
        #rospy.logwarn("Cur steering value: {}".format(steering))

        vel_error = linear_vel_desired - current_vel

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0.0
        """
        if throttle > 0.0:
            # positive control input -> accelerating
            throttle = max(self.throttle_min, min(self.throttle_max, throttle))
        else:
            # negative control input -> decelerating
            throttle = 0.0
            decel = max(vel_error, self.decel_limit)
            # apply at least 700 Nm of torque 
            brake = max(700.0, abs(decel) * self.vehicle_mass * self.wheel_radius) 
        """
        if linear_vel_desired == 0.0 and current_vel < 0.25:
            throttle = 0.0 
            brake = 700.0 # Nm - torque to keep car standing still
        elif throttle < 0.1 and vel_error < 0.0:
            throttle = 0.0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # torque in Nm

        # clip steering
        #steering = max(self.steer_min, min(self.steer_max, steering))

        return throttle, brake, steering