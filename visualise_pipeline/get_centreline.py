import os

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import SimpleITK as sitk


# ==============================================================
# Slice viewer (Shift + wheel switches axis, Ctrl toggles wheel scrolling)
# ==============================================================

class SliceViewer:
    """Shows three orthogonal slices of a volume.

    Shift + wheel -> switch the active axis (X -> Y -> Z)
    Ctrl (tap)    -> toggle scrolling mode: the wheel then scrolls the slices
                     of the active axis
    The active axis gets a checkbox (show/hide) and a slider (slice index).
    """

    NAMES = ("X", "Y", "Z")

    # polyscope zooms the camera on every wheel event before our callback runs,
    # so in scrolling mode we restore the previous frame's view to cancel it
    LOCK_ZOOM_WHILE_SCROLLING = True

    def __init__(self, volume, spacing):

        # float32 + contiguous once, so nothing is converted on every slice change
        self.volume = np.ascontiguousarray(volume, dtype=np.float32)
        self.shape = self.volume.shape
        self.spacing = np.asarray(spacing, dtype=float)

        # fixed intensity range so contrast doesn't jump from slice to slice
        lo, hi = np.percentile(self.volume, (0.5, 99.5))
        self.vminmax = (float(lo), float(hi))

        # geometry that never changes is stored once per axis:
        # node dims (2 nodes = one voxel thick along the slice axis) and the
        # full-extent bounds. Only the slice-axis bounds move with the slice.
        extent = np.array(self.shape) * self.spacing
        self.node_dims = []
        self.bound_low = []
        self.bound_high = []
        for axis in range(3):
            dims = np.array(self.shape) + 1
            dims[axis] = 2
            self.node_dims.append(tuple(int(d) for d in dims))
            self.bound_low.append(np.zeros(3))
            self.bound_high.append(extent.copy())

        # state
        self.slices = [s // 2 for s in self.shape]  # [x, y, z]
        self.axis_enabled = [True, True, True]
        self.active_axis = 2

        # scrolling mode (toggled by tapping Ctrl)
        self.scroll_mode = False
        self._ctrl_down = False
        self._ctrl_used = False   # Ctrl was combined with a click / wheel
        self._view_snapshot = None

        for axis in range(3):
            self.update_slice_view(axis)

    # ----------------------------------------------------------
    # Main update (call once per frame from the polyscope callback)
    # ----------------------------------------------------------

    def update(self):

        self.handle_scroll_toggle()
        self.handle_wheel()
        self.draw_ui()

        # remember the camera so the next wheel event can be un-zoomed
        if self.scroll_mode and self.LOCK_ZOOM_WHILE_SCROLLING:
            self._view_snapshot = ps.get_view_as_json()

    # ----------------------------------------------------------
    # Tap Ctrl (press + release, no click/wheel in between) toggles scrolling.
    # Ctrl+drag (box selection) and Ctrl+click (typing into a slider) are
    # therefore left alone.
    # ----------------------------------------------------------

    def handle_scroll_toggle(self):

        io = psim.GetIO()

        if io.KeyCtrl:

            if not self._ctrl_down:
                self._ctrl_down = True
                self._ctrl_used = False

            if (io.MouseWheel != 0
                    or psim.IsMouseClicked(0) or psim.IsMouseClicked(1)):
                self._ctrl_used = True

        elif self._ctrl_down:

            self._ctrl_down = False

            if not self._ctrl_used:
                self.scroll_mode = not self.scroll_mode
                self._view_snapshot = (
                    ps.get_view_as_json() if self.scroll_mode else None
                )

    # ----------------------------------------------------------
    # Mouse wheel:
    #   Shift + wheel        -> switch the active axis
    #   wheel (scroll mode)  -> scroll the slices of the active axis
    # ----------------------------------------------------------

    def handle_wheel(self):

        io = psim.GetIO()

        # ignore the wheel when the cursor is over an ImGui window
        if io.WantCaptureMouse:
            return

        wheel = io.MouseWheel

        if wheel == 0:
            return

        step = int(np.sign(wheel))

        if io.KeyShift:
            self.active_axis = (self.active_axis + step) % 3
            return

        if not self.scroll_mode:
            return

        # cancel the camera zoom polyscope already applied this frame
        if self.LOCK_ZOOM_WHILE_SCROLLING and self._view_snapshot is not None:
            ps.set_view_from_json(self._view_snapshot, fly_to=False)

        axis = self.active_axis

        self.slices[axis] = int(np.clip(
            self.slices[axis] + step, 0, self.shape[axis] - 1
        ))

        if self.axis_enabled[axis]:
            self.update_slice_view(axis)

    # ----------------------------------------------------------
    # Update ONE axis only (the one that changed)
    # ----------------------------------------------------------

    def update_slice_view(self, axis):

        name = self.NAMES[axis]
        idx = self.slices[axis]
        sp = self.spacing[axis]

        # only the slice-axis bounds change; everything else is precomputed
        low = self.bound_low[axis]
        high = self.bound_high[axis]
        low[axis] = idx * sp
        high[axis] = (idx + 1) * sp

        # slice as a view, keeping a singleton axis -> e.g. (1, Ny, Nz)
        sl = [slice(None)] * 3
        sl[axis] = slice(idx, idx + 1)
        arr = np.ascontiguousarray(self.volume[tuple(sl)])

        vg = ps.register_volume_grid(
            name, self.node_dims[axis], tuple(low), tuple(high)
        )
        vg.add_scalar_quantity(
            "intensity", arr, defined_on="cells", enabled=True,
            cmap="gray", vminmax=self.vminmax,
        )
        # polyscope remembers the enabled state by structure name, so a
        # re-registered grid stays hidden unless we enable it explicitly
        vg.set_enabled(self.axis_enabled[axis])

    # ----------------------------------------------------------
    # UI display
    # ----------------------------------------------------------

    def draw_ui(self):

        axis = self.active_axis
        name = self.NAMES[axis]

        psim.Separator()
        if self.scroll_mode:
            psim.Text(f"{name} axis (scrolling; Ctrl to disable)")
        else:
            psim.Text(f"{name} axis (Ctrl to enable scrolling)")
        psim.Text("Shift + wheel to switch axis")

        # show / hide this axis' slice
        changed, self.axis_enabled[axis] = psim.Checkbox(
            f"{name} slice", self.axis_enabled[axis]
        )
        if changed:
            if self.axis_enabled[axis]:
                self.update_slice_view(axis)  # catch up with the slider
            else:
                ps.get_volume_grid(name).set_enabled(False)

        # slice index
        changed, idx = psim.SliderInt(
            f"{name} index", self.slices[axis], 0, self.shape[axis] - 1
        )
        if changed:
            self.slices[axis] = idx
            if self.axis_enabled[axis]:
                self.update_slice_view(axis)


# ==============================================================
# Curve network editor (annotation only, no knowledge of the image)
# ==============================================================

class CurveNetworkEditor:
    """Annotate a curve network on top of whatever is in the scene.

    Double-click a surface      -> add a point (linked to the active point)
    Click a node                -> select it (and make it active)
    Shift + click a node        -> add / remove it from the selection
    Ctrl + drag                 -> box-select (hold Shift too to add)
    Drag a selected node        -> move all selected nodes (in the view plane)
    Delete / Backspace          -> delete selected nodes (and their edges)
    Escape                      -> clear the selection
    "Set Inlet" button          -> store the selected nodes in `self.inlet`
                                   (node indices; `inlet_points` gives the
                                   positions)
    """

    OWN_STRUCTURES = {"Curve Points", "Curve Network"}

    # appearance (radii are relative to the scene length scale)
    NODE_RADIUS = 0.006
    EDGE_RADIUS = 0.002
    NODE_COLOR = (1.0, 0.8, 0.1)      # yellow nodes
    SELECTED_COLOR = (1.0, 0.2, 0.9)  # magenta selected nodes
    INLET_COLOR = (0.2, 1.0, 0.3)     # green inlet nodes
    ACTIVE_COLOR = (1.0, 0.0, 0.0)    # red active node
    EDGE_COLOR = (0.2, 0.6, 1.0)      # blue lines

    # polyscope rotates the camera on any left-drag before our callback runs,
    # so while box-selecting / moving nodes we restore the view captured at
    # the start of the drag to cancel that (set False if it causes trouble)
    LOCK_CAMERA_WHILE_DRAGGING = True

    def __init__(self):

        self.enabled = False

        # annotation data
        self.points = []
        self.edges = []

        self.active_point = None   # new points get linked to this one
        self.selected = set()      # indices of selected nodes
        self.inlet = []            # indices of the nodes stored by "Set Inlet"

        # box selection state
        self.box_dragging = False
        self.box_start = None
        self.box_end = None

        # node moving state
        self.moving = False
        self.move_normal = None       # drag plane normal (view direction)
        self.move_anchor = None       # a point on the drag plane
        self.move_start_hit = None    # where the mouse ray first hit the plane
        self.move_origins = {}        # index -> position when the drag began

        self._view_snapshot = None

    # ==========================================================
    # Main update
    # ==========================================================

    def update(self):

        changed, self.enabled = psim.Checkbox(
            "Curve Network Editor",
            self.enabled
        )

        if not self.enabled:
            self.moving = False
            self.box_dragging = False
            return

        self.handle_keys()
        self.handle_point_selection()
        self.handle_point_creation()
        self.handle_box_selection()
        self.handle_point_move()

        self.draw_selection_box()
        self.draw_ui()

    # ==========================================================
    # Camera helpers (world <-> screen), from polyscope's camera parameters
    # Assumes a perspective camera.
    # ==========================================================

    def _camera(self):

        params = ps.get_view_camera_parameters()

        pos = np.array(params.get_position(), dtype=float)
        look = np.array(params.get_look_dir(), dtype=float)
        up = np.array(params.get_up_dir(), dtype=float)

        look /= np.linalg.norm(look)
        right = np.cross(look, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, look)

        tan_half = np.tan(np.radians(params.get_fov_vertical_deg()) / 2.0)
        width, height = ps.get_window_size()

        return pos, look, right, up, tan_half, float(width), float(height)

    def world_to_screen(self, pts):
        """(N,3) world points -> ((N,2) screen coords, (N,) depth along view)."""

        pos, look, right, up, tan_half, w, h = self._camera()

        d = np.asarray(pts, dtype=float) - pos
        depth = d @ look

        with np.errstate(divide="ignore", invalid="ignore"):
            ndc_x = (d @ right) / (depth * tan_half * (w / h))
            ndc_y = (d @ up) / (depth * tan_half)

        screen = np.stack([(ndc_x * 0.5 + 0.5) * w,
                           (0.5 - ndc_y * 0.5) * h], axis=1)

        return screen, depth

    def ray_plane_hit(self, screen, anchor, normal):
        """Intersect the mouse ray with the plane through `anchor`."""

        pos, look, right, up, tan_half, w, h = self._camera()

        ndc_x = 2.0 * screen[0] / w - 1.0
        ndc_y = 1.0 - 2.0 * screen[1] / h

        d = look + right * (ndc_x * tan_half * (w / h)) + up * (ndc_y * tan_half)

        denom = d @ normal
        if abs(denom) < 1e-9:
            return None

        t = ((anchor - pos) @ normal) / denom
        if t <= 0:
            return None

        return pos + t * d

    def lock_camera(self):

        if self.LOCK_CAMERA_WHILE_DRAGGING and self._view_snapshot is not None:
            ps.set_view_from_json(self._view_snapshot, fly_to=False)

    # ==========================================================
    # Keyboard: delete / clear selection
    # ==========================================================

    @staticmethod
    def key_pressed(names):

        for n in names:
            key = getattr(psim, n, None)
            if key is not None and psim.IsKeyPressed(key, False):
                return True
        return False

    def handle_keys(self):

        if psim.GetIO().WantCaptureKeyboard:
            return

        if self.key_pressed(("ImGuiKey_Delete", "ImGuiKey_Backspace")):
            self.delete_selected()

        if self.key_pressed(("ImGuiKey_Escape",)):
            self.clear_selection()

    # ==========================================================
    # Click a node to select it (Shift: toggle) and start moving it
    # ==========================================================

    def handle_point_selection(self):

        io = psim.GetIO()

        # Ctrl+drag is reserved for box selection
        if io.KeyCtrl or io.WantCaptureMouse:
            return

        if not psim.IsMouseClicked(0):
            return

        pick = ps.pick(screen_coords=io.MousePos)

        if not pick.is_hit or pick.structure_name != "Curve Points":
            return

        # for a point cloud, local_index is the index of the picked point
        idx = int(pick.local_index)

        if not (0 <= idx < len(self.points)):
            return

        if io.KeyShift:
            # additive selection (Shift+drag pans the camera, so no moving here)
            if idx in self.selected:
                self.selected.discard(idx)
            else:
                self.selected.add(idx)
                self.active_point = idx
        else:
            # keep the selection if the node is already part of it,
            # so the whole group can be dragged
            if idx not in self.selected:
                self.selected = {idx}
            self.active_point = idx
            self.begin_move(idx, io.MousePos)

        self.update_geometry()

    # ==========================================================
    # Double-click add point
    # ==========================================================

    def handle_point_creation(self):

        if not psim.IsMouseDoubleClicked(0):
            return

        io = psim.GetIO()

        # don't pick through ImGui windows
        if io.WantCaptureMouse:
            return

        pick = ps.pick(screen_coords=io.MousePos)

        if not pick.is_hit:
            return

        # ignore hits on the editor's own geometry
        if pick.structure_name in self.OWN_STRUCTURES:
            return

        # 3D hit position in world space (any structure is accepted)
        world_pos = np.array(pick.position, dtype=float)

        new_idx = len(self.points)

        self.points.append(world_pos)

        # connect TO ACTIVE POINT
        if self.active_point is not None:
            self.edges.append(
                [self.active_point, new_idx]
            )

        self.active_point = new_idx
        self.selected = {new_idx}

        self.update_geometry()

    # ==========================================================
    # Moving selected nodes (drag in the plane facing the camera)
    # ==========================================================

    def begin_move(self, idx, mouse_pos):

        normal = self._camera()[1]           # view direction
        anchor = self.points[idx].copy()

        hit = self.ray_plane_hit(mouse_pos, anchor, normal)
        if hit is None:
            return

        self.moving = True
        self.move_normal = normal
        self.move_anchor = anchor
        self.move_start_hit = hit
        self.move_origins = {i: self.points[i].copy() for i in self.selected}
        self._view_snapshot = ps.get_view_as_json()

    def handle_point_move(self):

        if not self.moving:
            return

        if not psim.IsMouseDown(0):
            self.moving = False
            return

        self.lock_camera()

        hit = self.ray_plane_hit(
            psim.GetIO().MousePos, self.move_anchor, self.move_normal
        )
        if hit is None:
            return

        delta = hit - self.move_start_hit

        if not np.any(delta):
            return

        for i, p0 in self.move_origins.items():
            self.points[i] = p0 + delta

        self.update_geometry()

    # ==========================================================
    # Deleting / clearing
    # ==========================================================

    def delete_selected(self):

        if not self.selected:
            return

        n = len(self.points)

        keep = np.ones(n, dtype=bool)
        keep[list(self.selected)] = False

        # old index -> new index
        new_index = -np.ones(n, dtype=int)
        new_index[keep] = np.arange(keep.sum())

        self.points = [p for i, p in enumerate(self.points) if keep[i]]

        # edges touching a deleted node are dropped (no automatic re-linking)
        self.edges = [
            [int(new_index[a]), int(new_index[b])]
            for a, b in self.edges
            if keep[a] and keep[b]
        ]

        # inlet nodes that were deleted disappear, the rest are re-indexed
        self.inlet = [int(new_index[i]) for i in self.inlet if keep[i]]

        if self.active_point is not None and keep[self.active_point]:
            self.active_point = int(new_index[self.active_point])
        else:
            self.active_point = None

        self.selected = set()
        self.moving = False

        self.update_geometry()

    def set_inlet(self):

        if not self.selected:
            return

        self.inlet = sorted(self.selected)
        self.update_geometry()

    @property
    def inlet_points(self):
        """World positions of the inlet nodes, shape (len(inlet), 3)."""

        return np.array([self.points[i] for i in self.inlet], dtype=float)

    def clear_selection(self):

        if self.selected:
            self.selected = set()
            self.update_geometry()

    # ==========================================================
    # Geometry updates
    # ==========================================================

    def update_geometry(self):

        if len(self.points) == 0:
            if ps.has_point_cloud("Curve Points"):
                ps.remove_point_cloud("Curve Points")
            if ps.has_curve_network("Curve Network"):
                ps.remove_curve_network("Curve Network")
            return

        pts = np.asarray(self.points, dtype=float)

        # Nodes: a point cloud (also works before the first edge exists).
        # Big spheres, coloured per point: selected nodes and the active node
        # stand out.
        cloud = ps.register_point_cloud("Curve Points", pts)
        cloud.set_radius(self.NODE_RADIUS)

        colors = np.tile(self.NODE_COLOR, (len(pts), 1))
        if self.inlet:
            colors[self.inlet] = self.INLET_COLOR
        if self.selected:
            colors[list(self.selected)] = self.SELECTED_COLOR
        if self.active_point is not None:
            colors[self.active_point] = self.ACTIVE_COLOR
        cloud.add_color_quantity("node colors", colors, enabled=True)

        # Lines: a thinner curve network in a different colour.
        if len(self.edges) > 0:

            net = ps.register_curve_network(
                "Curve Network",
                pts,
                np.asarray(self.edges, dtype=np.int64)
            )
            net.set_radius(self.EDGE_RADIUS)
            net.set_color(self.EDGE_COLOR)

        elif ps.has_curve_network("Curve Network"):
            ps.remove_curve_network("Curve Network")

    # ==========================================================
    # Box selection (Ctrl + drag; Ctrl + Shift + drag adds to selection)
    # ==========================================================

    def handle_box_selection(self):

        io = psim.GetIO()

        if (not self.box_dragging and io.KeyCtrl
                and not io.WantCaptureMouse and psim.IsMouseClicked(0)):

            self.box_dragging = True
            self.box_start = tuple(psim.GetMousePos())
            self.box_end = self.box_start
            self._view_snapshot = ps.get_view_as_json()
            return

        if not self.box_dragging:
            return

        self.lock_camera()

        self.box_end = tuple(psim.GetMousePos())

        if psim.IsMouseReleased(0):

            self.box_dragging = False
            self.select_points_in_box(additive=io.KeyShift)

    # ==========================================================
    # Draw rectangle overlay
    # ==========================================================

    def draw_selection_box(self):

        if not self.box_dragging:
            return

        draw = psim.GetForegroundDrawList()

        draw.AddRect(
            self.box_start,
            self.box_end,
            psim.IM_COL32(0, 255, 0, 255),
            thickness=2.0
        )

    # ==========================================================
    # Select every node inside the box
    # ==========================================================

    def select_points_in_box(self, additive=False):

        if len(self.points) == 0:
            return

        x0, y0 = self.box_start
        x1, y1 = self.box_end

        # a plain Ctrl+click (tiny box) is not a selection
        if abs(x1 - x0) < 3 and abs(y1 - y0) < 3:
            return

        screen, depth = self.world_to_screen(np.asarray(self.points))

        inside = (
            (depth > 0)
            & (screen[:, 0] >= min(x0, x1)) & (screen[:, 0] <= max(x0, x1))
            & (screen[:, 1] >= min(y0, y1)) & (screen[:, 1] <= max(y0, y1))
        )

        hits = set(np.flatnonzero(inside).tolist())

        self.selected = (self.selected | hits) if additive else hits

        if hits:
            self.active_point = max(hits)   # most recently added of the hits

        self.update_geometry()

    # ==========================================================
    # UI display
    # ==========================================================

    def draw_ui(self):

        psim.Separator()

        psim.Text(f"Points: {len(self.points)}")
        psim.Text(f"Edges: {len(self.edges)}")
        psim.Text(f"Selected: {len(self.selected)}")

        if self.active_point is not None:
            psim.Text(f"Active: {self.active_point}")

        psim.Text(f"Inlet: {self.inlet if self.inlet else 'not set'}")

        if psim.Button("Set Inlet"):
            self.set_inlet()

        psim.SameLine()

        if psim.Button("Delete selected"):
            self.delete_selected()

        psim.SameLine()

        if psim.Button("Clear selection"):
            self.clear_selection()

        psim.Text("Click: select | Shift+click: add")
        psim.Text("Ctrl+drag: box (+Shift: add)")
        psim.Text("Drag selected: move | Del: delete")
        psim.Text("Double-click surface: add point")


# ==============================================================
# Image loading (drag & drop a file or a DICOM folder onto the window)
# ==============================================================

# optionally load an image at startup, e.g.
# INITIAL_IMAGE = '../MoCoLoR/001/struct_gradwarp/Images_mocolor_vent0.nii.gz'
INITIAL_IMAGE = None


def load_volume(path):
    """Read a 3D image (any format SimpleITK reads, or a DICOM folder).

    Returns (volume [X,Y,Z], spacing [X,Y,Z]) in LPS orientation.
    """

    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(path)
        if not files:
            raise ValueError("no DICOM series found in this folder")
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(path)

    if img.GetDimension() != 3:
        raise ValueError(f"expected a 3D image, got a {img.GetDimension()}D one")

    img = sitk.DICOMOrient(img, 'LPS')

    spacing = np.array(img.GetSpacing())  # SimpleITK spacing is already [X,Y,Z]
    arr_img = sitk.GetArrayFromImage(img)  # numpy array in [Z,Y,X]
    volume = arr_img.transpose(2, 1, 0)  # rearrange [Z,Y,X] to [X,Y,Z]

    return volume, spacing


slice_viewer = None
status = ""


def load_image(path):
    """Load `path` and (re)build the slice viewer. Keeps the old image on failure."""

    global slice_viewer, status

    name = os.path.basename(os.path.normpath(path))

    try:
        volume, spacing = load_volume(path)
    except Exception as e:
        print(f"Could not load {path}:\n{e}")   # full SimpleITK message
        status = f"Could not load {name}: {str(e).strip().splitlines()[-1]}"
        return

    slice_viewer = SliceViewer(volume, spacing)
    ps.reset_camera_to_home_view()

    status = f"Loaded {name}  {volume.shape}"


def on_files_dropped(paths):

    load_image(paths[0])

    global status
    if len(paths) > 1:
        status += f"  (ignored {len(paths) - 1} other dropped item(s))"

ps.init()

ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0, 0, 0, 0])
ps.set_navigation_style("free")
ps.set_ground_plane_mode("none")

ps.set_files_dropped_callback(on_files_dropped)

curve_editor = CurveNetworkEditor()

if INITIAL_IMAGE is not None:
    load_image(INITIAL_IMAGE)


def callback():

    if status:
        psim.Text(status)

    if slice_viewer is None:
        psim.Text("Drag & drop a 3D image (file or DICOM folder) here")
    else:
        slice_viewer.update()

    curve_editor.update()


ps.set_user_callback(callback)
ps.show()