import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from capsule import Capsule
from dipole import Dipole


class Simulator():
    def __init__(self,
                 radium=400,  # mm
                 num_samples=10,
                 axes_acc=3,
                 axes_mag=3,
                 mgt_m=66.0,  # A*m^2
                 noise=True):
        self.r = radium / 1000
        self.num_samples = num_samples
        self.axes_acc = axes_acc
        self.axes_mag = axes_mag
        self.mgt_m = mgt_m

        self.capsule = Capsule(axes_acc=self.axes_acc,
                               axes_mag=self.axes_mag,
                               radius=self.r)
        self.dipole = Dipole(radium=self.r, mgt_m=self.mgt_m)
        self.noise = noise

        # Pre-generate noise for accelerometer and magnetometer readings
        self.acc_noise = np.random.normal(0, 0.002, (self.num_samples, 3)) if noise else np.zeros((self.num_samples, 3))
        self.mag_noise = np.random.normal(0, 10.6 * 1e-6, (self.num_samples, 3)) if noise else np.zeros((self.num_samples, 3))

    def magnetic_field_model(self, pc, me):
        mu0 = 4 * np.pi * 1e-7
        norm_pc = np.linalg.norm(pc)
        norm_me = np.linalg.norm(me)

        if pc.ndim == 1:
            pc = pc.reshape(-1, 1)
        if me.ndim == 1:
            me = me.reshape(-1, 1)

        B = (mu0 * norm_me) / (4 * np.pi * norm_pc**3)
        term = (3 * np.dot(pc, pc.T) / norm_pc**2) - np.identity(3)
        b = B * (term @ (me / norm_me))
        return b

    def rotate_vector(self, vec, axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        cross_product = np.cross(axis, vec)
        dot_product = np.dot(axis, vec)
        rotated_vector = cos_theta * vec + sin_theta * cross_product + (1 - cos_theta) * dot_product * axis
        return rotated_vector

    def generate_rotating_me(self, ):
        initial_dipole = np.array([0, 0, self.mgt_m])
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        angles = np.linspace(0, np.pi / 2, 50)
        dipole_moments = []

        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                for angle_i in angles:
                    for angle_j in angles:
                        # First rotate around one axis
                        intermed_dipole = self.rotate_vector(initial_dipole, axes[i], angle_i)
                        # Then rotate the result around another axis
                        rotated_dipole = self.rotate_vector(intermed_dipole, axes[j], angle_j)
                        dipole_moments.append(rotated_dipole)
        return dipole_moments

    def calculate_angles(self, g):
        gx, gy, gz = g
        theta = np.arctan2(-gx, np.sqrt(gy**2 + gz**2))
        phi = np.arctan2(gy, gz)

        return theta, phi

    def rotation_matrix(self, theta, phi, psi):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi), np.cos(phi)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi), np.cos(psi), 0],
                       [0, 0, 1]])

        return Rz.T @ Ry.T @ Rx.T

    def calculate_bm(self, pos, me, phi, theta, psi, type='x-y'):
        if type == 'x-y':
            P = np.array([[1, 0, 0], [0, 1, 0]])
        elif type == 'y-z':
            P = np.array([[0, 1, 0], [0, 0, 1]])
        else:
            raise ValueError("Invalid type specified. Use 'x-y' or 'y-z'.")

        b = self.magnetic_field_model(pos, me)
        rotation = self.rotation_matrix(theta, phi, psi)
        bm = P @ rotation @ b

        return bm

    def objective_function(self, x, pc, mes, ro_truth, type='x-y', idx=0):
        pos = x[:3]  # Updated position
        psi = x[3]  # Updated psi rotation angle
        g_reading = self.capsule.generate_acc_data(ro_truth)
        g_reading_noise = self.acc_noise[idx] + g_reading

        # Calculate angles from accelerometer data
        theta, phi = self.calculate_angles(g_reading)
        theta_n, phi_n = self.calculate_angles(g_reading_noise)

        residuals = []
        for me in mes:
            me_noise = self.mag_noise[idx] if self.noise else np.zeros(3)

            # Calculate modeled magnetometer data based on the current position and pose
            Bm = self.calculate_bm(pos, me + me_noise, phi_n, theta_n, psi, type)

            # Objective: minimize the difference between actual magnetic field and modeled field
            Be = self.calculate_bm(pc, me, phi, theta, ro_truth[-1], type)
            residuals.extend(Bm - Be)

        return np.array(residuals).flatten()

    def simulate(self, ):
        # generate test points
        test_positions = self.capsule.generate_position_data(self.num_samples)
        ro_truth = (np.radians(0), np.radians(0), np.radians(10))
        results = []

        for idx, pc in enumerate(test_positions):
            print("--------------------------------------")
            print("Ground Truth: {}".format(pc))
            dipole_moments = self.generate_rotating_me()

            # six init values
            X = [np.array([-81*1e-3, -81*1e-3, -81*1e-3, 0]),
                 np.array([110*1e-3, -30*1e-3, -81*1e-3, 0]),
                 np.array([-30*1e-3, 110*1e-3, -81*1e-3, 0]),
                 np.array([-81*1e-3, -81*1e-3, -81*1e-3, np.radians(180)]),
                 np.array([110*1e-3, -30*1e-3, -81*1e-3, np.radians(180)]),
                 np.array([-30*1e-3, 110*1e-3, -81*1e-3, np.radians(180)])]
            flag = 0
            for x0 in X:
                print("current test position: {}".format(x0))
                result = least_squares(lambda x: self.objective_function(x, pc, dipole_moments, ro_truth, 'x-y', idx),
                                       x0, method='lm', verbose=2)

                if result.x[-2] > 0:
                    for i in range(3):
                        result.x[i] = -result.x[i]

                distance = np.linalg.norm(result.x[:3] - pc)
                print("the estimated: {} the distance: {}".format(result.x, distance))
                if distance < 1e-4:
                    flag = 1
                    print("the estimated: {}".format(result.x))
                    break

            if flag == 0:
                print("Don't find the global minimum")

            results.append(result.x)
            print("--------------------------------------")

        self.plot(test_positions, np.array(results))
        return results

    def plot(self, truth_pos, estimated_pos):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(truth_pos[:, 0]*1e3, truth_pos[:, 1]*1e3, truth_pos[:, 2]*1e3, c='b', label='Truth')
        ax.scatter(estimated_pos[:, 0]*1e3, estimated_pos[:, 1]*1e3, estimated_pos[:, 2]*1e3, c='r', label='Estimated')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Truth vs Estimated Positions')
        ax.legend()
        plt.savefig("../figures/sim_noise.png" if self.noise else "../figures/sim.png")
        plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    simulator = Simulator(noise=False)
    results = simulator.simulate()
