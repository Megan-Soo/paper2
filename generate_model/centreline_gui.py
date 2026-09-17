# centreline v4 takes 12 nodes selected from raw airway image, assembles a 14-node centreline template of the upper airway, and writes out the exnode ipnode files.
import SimpleITK as sitk
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.image as mpimg
import numpy as np
from matplotlib.widgets import Slider
import generate

# Scrollable plot of 2D image slices w/ mpldatacursor
class IndexTracker(object):
    def __init__(self, ax, amp_slider, X, aspect_ratio, view='c'):
        self.count = 0 # initialise counter for number of coords collected
        self.view = view # need this for other functions in this class
        self.ax = ax
        self.amp_slider = amp_slider
        self.ax.set_title('Use scroll wheel to navigate images', loc='center', wrap=True)

        self.X = X
        if self.view == 's':
            _, _, self.slices = self.X.shape  # for sagittal view
        elif self.view == 'c':
            _, self.slices, _, = self.X.shape  # for coronal view
        elif self.view == 'a':
            self.slices, _, _ = self.X.shape  # for axial view
        else:  # default to coronal view
            _, self.slices, _, = self.X.shape
        self.ind = self.slices // 2

        if self.view == 's':
            self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', aspect=str(aspect_ratio))  # for sagittal view
        elif self.view == 'c':
            self.im = ax.imshow(self.X[:, self.ind, :], cmap='gray', aspect=str(aspect_ratio))  # for coronal view
        elif self.view == 'a':
            self.im = ax.imshow(self.X[self.ind, :, :], cmap='gray', aspect=str(aspect_ratio))  # for axial view
        else:  # default to coronal view
            self.im = ax.imshow(self.X[:, self.ind, :], cmap='gray', aspect=str(aspect_ratio))
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()
        self.amp_slider.set_val(self.ind)

    def update(self):
        if self.view == 's':
            self.im.set_data(self.X[:, :, self.ind])  # for sagittal view
        elif self.view == 'c':
            self.im.set_data(self.X[:, self.ind, :])  # for coronal view
        elif self.view == 'a':
            self.im.set_data(self.X[self.ind, :, :])  # for axial view
        else:  # default to coronal view
            self.im.set_data(self.X[:, self.ind, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    ## Function to collect coords on cursor click.
    def onclick(self, event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        # z = self.tracker.ind
        z = self.ind # slice number

        xmin = 0
        ymin = 0
        xmax = self.ax.dataLim.bounds[2]
        ymax = self.ax.dataLim.bounds[3]

        try:
            # Change values of selected pixel and its neighbours
            x = int(ix)  # + x_offset
            y = int(iy)  # + y_offset
            z = int(z)
            if x<=xmin or x>=xmax or y<=ymin or y>=ymax:  # if click outside image slice, don't store coords
                return
        except TypeError:
            return

        try:
            if self.view == 's':
                self.X[x][(y - 2):(y + 2), (x - 2):(x + 2)] = 9000  # for sagittal view
            elif self.view == 'c':
                self.X[y][(z - 2):(z + 2), (x - 2):(x + 2)] = 9000  # for coronal view
            elif self.view == 'a':
                self.X[z][(y - 2):(y + 2), (x - 2):(x + 2)] = 9000  # for axial view
            else:  # default to coronal view
                self.X[y][(z - 2):(z + 2), (x - 2):(x + 2)] = 9000

            self.count = self.count+1
            print(f'    {self.count}. ', end='')
            if self.view == 's':
                print(f'    x = {z},       y = {int(ix)},                z = {int(iy)}') # for sagittal view
            elif self.view == 'c':
                print(f'    x = {int(ix)},       y = {z},                z = {int(iy)}')  # for coronal view
            elif self.view == 'a':
                print(f'    x = {int(ix)},       y = {int(iy)},                z = {z}') # for axial view
            else: # default to coronal view
                print(f'    x = {x},          y = {z},                z = {y}')

            # global coords
            if self.view == 's':
                indices.append((z, ix, iy)) # for sagittal view
            elif self.view == 'c':
                indices.append((ix, z, iy))  # for coronal view
            elif self.view == 'a':
                indices.append((ix, iy, z)) # for axial view
            else: # default to coronal view
                indices.append((ix, z, iy))

        except IndexError: # if clicked out of bounds of img slice, no coords to collect, return
            return

        return indices

class RunGUI:
    def __init__(self, image, view='c',title=None):

        X = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()

        global indices
        indices = []  # initialise array to store clicked coords

        size = image.GetSize()
        if view=='a':
            slices = size[2]
            aspect_ratio = spacing[1]/spacing[0]
        elif view=='c':
            slices = size[1]
            aspect_ratio = spacing[2]/spacing[0]
        elif view=='s':
            slices = size[0]
            aspect_ratio = spacing[2]/spacing[1]
        else:
            slices = size[1]
            aspect_ratio = spacing[2]/spacing[0]

        # open a figure window
        self.fig, ax = plt.subplots()
        if title is not None:
            self.fig.suptitle(title)
        # adjust the main plot to make room for the sliders
        self.fig.subplots_adjust(left=0.25)
        # Make a vertically oriented slider to control the amplitude
        axamp = self.fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        self.amp_slider = Slider(
            ax=axamp,
            label="Slice",
            valmin=0,
            valmax=slices-1,
            valinit=0,
            orientation="vertical",
            initcolor='None',
            valfmt='%0.0f' # set to discrete integer values
        )

        self.amp_slider.on_changed(self.slider_update) # register the update function with each slider

        ## Call scrollable slice viewer
        self.tracker = IndexTracker(ax, self.amp_slider, X, aspect_ratio, view) # X is the image array
        self.fig.canvas.mpl_connect('scroll_event', self.tracker.onscroll) # connect plot to scrollable slice viewer
        self.fig.canvas.mpl_connect('button_press_event', self.tracker.onclick) # connect plot to coord collector

        plt.show()
    
    # The function to be called anytime a slider's value changes
    def slider_update(self, val):
        
        self.tracker.ind = int(self.amp_slider.val)
        self.fig.canvas.mpl_connect('scroll_event', self.tracker.update())  # Reconnect plot to scrollable slice viewer

class IndexProcessor:
    
    @staticmethod
    def global_coords(coords, spacing, origin):

        coords = np.asarray(coords)
        
        z_spacing = spacing[1]
        z_offset = origin[2]
        y_spacing = spacing[2]
        y_offset = origin[1]
        x_spacing = spacing[0]
        x_offset = origin[0]

        for i in range(len(coords)):
            # translate z coord
            coords[i][2] = coords[i][2] * z_spacing + z_offset
            # translate y coord
            coords[i][1] = coords[i][1] * y_spacing + y_offset
            # translate x coord
            coords[i][0] = coords[i][0] * x_spacing + x_offset

        return coords
    
    @staticmethod
    def get_physical_location(coords, image):
        for i in range(len(coords)):
            coords[i] = image.TransformContinuousIndexToPhysicalPoint(coords[i])
        coords = np.array(coords)
        return coords

    @staticmethod
    def sort_centreline_nodes(indices):
        try:
            ## Arrange selected landmarks in order of: trachea nodes, left lung nodes, right lung nodes.
            centreline = np.zeros((14,3))

            # Sort along z axis in ascending order. Airway entry is the 1st node; as node number increases, z value increases
            indices = np.array(indices)
            indices = indices[indices[:, 2].argsort()] # sorts in ascending order
            # indices = np.flip(indices, axis=0) # flip array vertically

            # Assign selected trachea nodes
            ## in CMISS template, node 1 is airway opening, node 3 is trachea midpoint, node 5 is trachea bifurcation/carina
            centreline[0] = indices[0]
            centreline[2] = indices[1]
            centreline[4] = indices[2]

            # Generate trachea nodes
            ## find Node 2, the midpoint of Node 1 & Node 3
            centreline[1] = [(centreline[0][0]+centreline[2][0])/2.0, (centreline[0][1]+centreline[2][1])/2.0, (centreline[0][2]+centreline[2][2])/2.0]
            # find Node 4, the midpoint of Node 3 & Node 5
            centreline[3] = [(centreline[2][0]+centreline[4][0])/2.0, (centreline[2][1]+centreline[4][1])/2.0, (centreline[2][2]+centreline[4][2])/2.0]
            print(" Trachea nodes assigned.\n")

            # delete processed trachea landmarks from coords
            indices = np.delete(indices, slice(0,3), axis=0)


            # Orientation set to RAS
            # Separate right & left lung nodes based on trachea bifurcation node.
            right_lung_nodes = []
            left_lung_nodes = []
            for i in range(len(indices)):
                if indices[i][0] < centreline[4][0]:
                    right_lung_nodes.append(indices[i])
                else:
                    left_lung_nodes.append(indices[i])

            ## convert to np array
            right_lung_nodes = np.asarray(right_lung_nodes)
            left_lung_nodes = np.asarray(left_lung_nodes)

            #--------------------- Start right lung. -----------------
            # Sort along z axis in ascending order.
            right_lung_nodes = right_lung_nodes[right_lung_nodes[:, 2].argsort()] # sorts in ascending order
            # right_lung_nodes = np.flip(right_lung_nodes, axis=0) # flip array vertically

            # Top 2 nodes are right bronchi bifurcation & RUL node
            ## in CMISS template, node 8 is right bronchi bif, node 11 is RUL node.
            ## sort by x value - x_RUL > x_RBronchi
            if right_lung_nodes[0][0] < right_lung_nodes[1][0]:
                centreline[10] = right_lung_nodes[0]
                centreline[7] = right_lung_nodes[1]
            elif right_lung_nodes[0][0] > right_lung_nodes[1][0]:
                centreline[10] = right_lung_nodes[1]
                centreline[7] = right_lung_nodes[0]
            print(' Right bronchus bifurcation and RUL node assigned.')

            # RML-RLL bifurcation node is 3rd node
            centreline[11] = right_lung_nodes[2]
            print(' RML-RLL bifurcation assigned.')

            # Bottom 2 nodes are RML & RLL
            ## in CMISS template, node 13 is RML, node 14 is RLL. Elem 12 is RML, Elem 13 is RLL.
            ## sort by y value = y_RML > y_RLL
            if right_lung_nodes[3][1] < right_lung_nodes[4][1]:
                centreline[12] = right_lung_nodes[3]
                centreline[13] = right_lung_nodes[4]
            elif right_lung_nodes[3][1] > right_lung_nodes[4][1]:
                centreline[12] = right_lung_nodes[4]
                centreline[13] = right_lung_nodes[3]
            print(' RML & RLL nodes assigned.')
            print(' All right lung nodes sorted.\n')

            #--------------------- Right lung done. -----------------
            #--------------------- Start left lung. -----------------

            # Sort along z axis in ascending order.
            left_lung_nodes = left_lung_nodes[left_lung_nodes[:, 2].argsort()] # sorts in ascending order
            # left_lung_nodes = np.flip(left_lung_nodes, axis=0) # flip array vertically

            # Top node is left bronchus node.
            ## in CMISS template, node 6 is left bronchi node.
            centreline[5] = left_lung_nodes[0]
            print(' Left bronchus node assigned.')
            left_lung_nodes = np.delete(left_lung_nodes, 0, axis=0)

            ## in CMISS template, node 7 is left bronchi bif, node 9 is LUL, node 10 is LLL. Elem 8 is LUL, Elem 9 is LLL.
            ## sort by x value - x_LBifurcation < x_LUL & x_LLL
            idx = np.argmin(left_lung_nodes, axis=0)[0]
            centreline[6] = left_lung_nodes[idx] # argmin finds the idx of min vals in each col i.e., min x, y,z values. 0th col is x col, so take [0]
            print(' Left bronchus bifurcation assigned.')

            # remove processed nodes
            left_lung_nodes = np.delete(left_lung_nodes, idx, axis=0)

            ## sort LUL & LLL nodes by z values - z_LUL > z_LLL
            if left_lung_nodes[0][2] < left_lung_nodes[1][2]:
                centreline[8] = left_lung_nodes[0]
                centreline[9] = left_lung_nodes[1]
            elif left_lung_nodes[0][2] > left_lung_nodes[1][2]:
                centreline[8] = left_lung_nodes[1]
                centreline[9] = left_lung_nodes[0]
            print(' LUL & LLL nodes assigned.')
            print(' All left lung nodes sorted.\n')

            #--------------------- Left lung done. -----------------

        except IndexError:  # if wrong number of nodes, exit() to cancel operation
            print(' Cancelled. IndexError.')
            exit()

        return centreline

    @staticmethod
    def sort_artery_centreline(indices):
        # 4 selected nodes to sort: entrypoint Main Pulmonary Artery (MPA measured here), MPA birfucation, Left Pulmonary Artery (1), Right Pulmonary Artery (1)
        # Then connect to upper airway centreline at airway nodes 7 (left) & 8 (right) downstream
        try:
            # Sort along x (medial-lateral) axis in ascending order.
            indices = np.array(indices)
            indices = indices[indices[:, 0].argsort()]
            
            # initialise pulmonary centreline template
            artery_centreline = []

            # Find the index of the point with the smallest y-coordinate
            idx_MPA = min(enumerate(indices), key=lambda p: p[1][1])[0]
            artery_centreline.append(indices[idx_MPA]) # assign MPA node (most anterior node)
            indices = np.delete(indices,idx_MPA, axis=0) # remove assigned point. MPA bifurcation is now the middle point

            artery_centreline.append(indices[1]) # assign MPA bifurcation
            artery_centreline.append(indices[0]) # assign RPA node (rightmost ie smallest x value)
            artery_centreline.append(indices[-1]) # assign LPA node (most left ie largest x value)
        
        except IndexError:  # if wrong number of nodes, exit() to cancel operation
            print(' Cancelled. IndexError.')
            return False
        
        return np.array(artery_centreline)

    @staticmethod
    def get_indices():
        return indices


class GetUpperAirway:
        
    def __init__(self, image, orientation, view):
        print(' Selecting landmarks...\n\n Index: Right -> Left, Anterior -> Posterior, Superior -> Inferior')
        
        image = sitk.DICOMOrient(image, orientation)
        
        RunGUI(image,view)
        print("\n Collected upper airway centreline indices.")
        self.centreline = IndexProcessor.sort_centreline_nodes(indices)
        spacing = image.GetSpacing()
        self.centreline = self.centreline * spacing
        if not generate.left_hand_system(self.centreline): # our universe works in the left hand coordinate system
            self.centreline = generate.convert_left_right_system(self.centreline)
        print(' Derived upper airway centreline template.\n')
        
class GetUpperArtery:

    def __init__(self, image, orientation, view, upper_airway_centreline):
        print(' Select MPA, MPA bifurcation, LPA, and RPA...\n\n Index: Right -> Left, Anterior -> Posterior, Superior -> Inferior')

        image = sitk.DICOMOrient(image, orientation)
        
        RunGUI(image,view)
        if len(indices) != 4:
            return Exception        
        print("\n Collected upper artery centreline indices.")
        self.centreline = IndexProcessor.sort_artery_centreline(indices)
        spacing = image.GetSpacing()
        # print(spacing)
        self.centreline = self.centreline * spacing
        if not generate.left_hand_system(self.centreline): # our universe works in the left hand coordinate system
            self.centreline = generate.convert_left_right_system(self.centreline)

        # interpolate Nodes 2-4 between 1st and 2nd selected nodes
        # Define the two points
        point1 = self.centreline[0]  # Replace with actual coordinates
        point2 = self.centreline[1]  # Replace with actual coordinates

        # Number of intermediate points to interpolate
        num_points = 3

        # Generate linearly spaced factors between 0 and 1
        factors = np.linspace(0, 1, num_points + 2)

        # Interpolate points and combine with original points
        pre_bifurcation = np.array([point1 + factor * (point2 - point1) for factor in factors])

        # Attach LPA node, equivalent to node 6 in upper_airway
        self.centreline = np.append(pre_bifurcation, [self.centreline[-1]], axis=0)
            
        # detach nodes 1-6 from upper airway centreline
        del upper_airway_centreline[0:6]

        # attach nodes 7-14 downstream of upper artery centreline
        self.centreline = np.append(self.centreline, upper_airway_centreline, axis=0) # axis=0 appends along rows
        print(' Derived upper artery centreline template.\n')