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

		throttle_kp = 0.5
		throttle_ki = 0.05
		throttle_kd = 0.1
		throttle_min = 0.0 # min. throttle value
		throttle_max = 0.8 # max. throttle value
		self.throttle_controller = PID(throttle_kp, throttle_ki, throttle_kd,
									   throttle_min, throttle_max)

		min_speed = 10.
		self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed,
											max_lat_accel, max_steer_angle)

		steer_kp = 0.8
		steer_ki = 0.1
		steer_kd = 0.3
		steer_min = -1.0 # min. steer value
		steer_max = 1.0 # max. steer value
		self.steer_controller = PID(steer_kp, steer_ki, steer_kd,
									steer_min, steer_max)


		tau = 0.5 # 1/(2*pi*tau) = filter cutoff frequency
		ts = 0.02 # sample time
		self.vel_lpf = LowPassFilter(tau, ts)

		self.vehicle_mass = vehicle_mass
		self.fuel_capacity = fuel_capacity
		self.brake_deadband = brake_deadband
		self.decel_limit = decel_limit
		self.accel_limit = accel_limit
		self.wheel_radius = wheel_radius
		self.last_time = rospy.get_time()
		self.last_steering = 0.0


	def control(self, current_vel, dbw_enable, linear_vel, angular_vel):
		# TODO: Change the arg, kwarg list to suit your needs
		# Return throttle, brake, steer
		if not dbw_enable:
			self.throttle_controller.reset()
			return 0.0, 0.0, self.last_steering

		current_time = rospy.get_time()
		sample_time = current_time - self.last_time
		self.last_time = current_time

		current_vel = self.vel_lpf.filt(current_vel)
		steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
		steering = self.steer_controller.step(steering, sample_time)
		#rospy.logwarn("Cur steering value: {}".format(steering))

		vel_error  = linear_vel - current_vel

		throttle = self.throttle_controller.step(vel_error, sample_time)
		brake = 0

		if linear_vel == 0.0 and current_vel < 0.1:
			throttle = 0 
			brake = 700 # Nm - torque to keep car standing still
		elif throttle < 0.1 and vel_error < 0:
			throttle = 0
			decel = max(vel_error, self.decel_limit)
			brake = abs(decel) * self.vehicle_mass * self.wheel_radius # torque in Nm

		self.last_steering = steering
		
		return throttle, brake, steering