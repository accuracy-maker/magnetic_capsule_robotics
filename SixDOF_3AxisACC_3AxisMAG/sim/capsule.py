import numpy as np

class Capsule():
    def __init__(self,
                 axes_acc = 3,
                 axes_mag = 3,
                 radius = 400):
        self.axes_acc = axes_acc
        self.axes_mag = axes_mag
        self.g = 9.81
        self.r = radius
        

    def generate_acc_data(self, ground_truth_ang:tuple):
        phi,theta,psi = ground_truth_ang
        Rx = np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])
        g = np.array([0, 0, -self.g])
        
        return Rz @ Ry @ Rx @ g
    
    
    def generate_position_data(self,num_points):
        radius = self.r
        min_z = -radius
        max_z = 0
        
        positions = []
        
        while len(positions) < num_points:
            x = np.random.uniform(-radius, radius)
            y = np.random.uniform(-radius, radius)
            z = np.random.uniform(min_z, max_z)

            # Check if within the sphere
            if x**2 + y**2 + z**2 <= radius**2:
                # Ensure magnitude constraint is met and z-component constraint
                pc_magnitude = np.sqrt(x**2 + y**2 + z**2)
                if pc_magnitude > 20*1e-3 and -z * pc_magnitude > 25.4*1e-3:
                    positions.append((x, y, z))

        return np.array(positions)
    
    
#test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    capsule = Capsule()
    num_samples = 20
    test_positions = capsule.generate_position_data(num_samples)
    
    # Set up the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 400 * np.outer(np.cos(u), np.sin(v))
    y = 400 * np.outer(np.sin(u), np.sin(v))
    z = 400 * np.outer(np.ones(np.size(u)), np.cos(v))  # Only the lower hemisphere
    ax.plot_surface(x, y, z, color='b', alpha=0.1)  # Transparent sphere

    # Plotting the test positions
    ax.scatter(test_positions[:, 0], test_positions[:, 1], test_positions[:, 2], color='red')

    # Axis labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    # Set aspect ratio
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    # Show the plot
    plt.show()
