

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from threading import Lock

import viser
from typing_extensions import override

from mjlab.sim.sim import Simulation
from mjlab.viewer.base import (
  BaseViewer,
  EnvProtocol,
  PolicyProtocol,
  VerbosityLevel,
)
from mjlab.viewer.viser.overlays import (
  ViserCameraOverlays,
  ViserContactOverlays,
  ViserDebugOverlays,
  ViserTermOverlays,
)
from mjlab.viewer.viser.scene import ViserMujocoScene


class UpdateReason(Enum):
  ACTION = auto()
  ENV_SWITCH = auto()
  SCENE_REQUEST = auto()


class ViserPlayViewer(BaseViewer):
  """Interactive Viser-based viewer with playback controls."""

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
    viser_server: viser.ViserServer | None = None,
    runner=None,
  ) -> None:
    super().__init__(env, policy, frame_rate, verbosity)
    self._runner = runner  # Optional OnPolicyRunner for live policy hot-swap
    self._term_overlays: ViserTermOverlays | None = None
    self._camera_overlays: ViserCameraOverlays | None = None
    self._debug_overlays: ViserDebugOverlays | None = None
    self._contact_overlays: ViserContactOverlays | None = None
    self._sim_lock = Lock()
    self._camera_update_last_ms: float = 0.0
    self._debug_queue_last_ms: float = 0.0
    self._scene_submit_enqueue_last_ms: float = 0.0
    self._scene_update_last_ms: float = 0.0
    self._timing_last_log_time: float = 0.0
    self._external_server = viser_server is not None
    self._server = viser_server or viser.ViserServer(host="0.0.0.0", label="mjlab")

  @override
  def setup(self) -> None:
    """Setup the viewer resources."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._counter = 0
    self._pending_update_reasons: set[UpdateReason] = set()

    # Create ViserMujocoScene for all 3D visualization (with debug visualization enabled).
    self._scene = ViserMujocoScene.create(
      server=self._server,
      mj_model=sim.mj_model,
      num_envs=self.env.num_envs,
    )

    self._scene.env_idx = self.cfg.env_idx
    self._scene.debug_visualization_enabled = False

    # ── Lighting ──────────────────────────────────────────────────────────
    # Disable default lights for full manual control.
    self._server.scene.configure_default_lights(enabled=False)
    # Very dim ambient so the scene is dark but not pitch black.
    self._server.scene.add_light_ambient(
      "/lights/ambient",
      color=(180, 200, 255),
      intensity=0.25,
    )
    # Spotlight from above pointing straight down onto the robot.
    import numpy as _np
    # wxyz for 90° rotation around X → spotlight faces -Z (down in Z-up frame).
    _q = (_np.cos(_np.pi / 4), _np.sin(_np.pi / 4), 0.0, 0.0)
    self._server.scene.add_light_spot(
      "/lights/spot_main",
      color=(255, 245, 220),
      intensity=8.0,
      angle=0.6,
      penumbra=0.3,
      decay=1.0,
      cast_shadow=True,
      position=(0.0, 0.0, 5.0),
      wxyz=_q,
    )

    # Background environment (options: apartment, city, dawn, forest, lobby, night, park, studio, sunset, warehouse)
    self._server.scene.configure_environment_map(hdri="forest", background=True, background_blurriness=0.4)

    # Title.
    self._server.gui.add_markdown(
        "<div style='text-align: center; margin-top: 5px;'>"
        "<h1 style='color: #4CAF50; margin: 0;'>🤖 STRIKE ROBOT</h1>"
        "<p style='color: #888; font-size: 14px; margin: 5px 0 15px 0;'>Interactive Physics & Motion Editor</p>"
        "</div>"
    )

    # Create tab group.
    tabs = self._server.gui.add_tab_group()

    # Main tab with simulation controls and display settings.
    with tabs.add_tab("Dashboard", icon=viser.Icon.DASHBOARD):
      self._status_html = self._server.gui.add_html("")

      # Playback controls.
      self._server.gui.add_markdown("### Playback Controls")
      self._pause_button = self._server.gui.add_button(
        "Play" if self._is_paused else "Pause",
        icon=viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE,
      )

      @self._pause_button.on_click
      def _(_) -> None:
        self.request_toggle_pause()

      reset_button = self._server.gui.add_button(
        "Reset Environment", icon=viser.Icon.REFRESH,
      )

      @reset_button.on_click
      def _(_) -> None:
        self.request_reset()

      speed_buttons = self._server.gui.add_button_group(
        "Speed",
        options=["Slower", "1x", "Faster"],
      )

      @speed_buttons.on_click
      def _(event) -> None:
        if event.target.value == "Slower":
          self.request_speed_down()
        elif event.target.value == "1x":
          self.request_reset_speed()
        else:
          self.request_speed_up()

      # Motion clips.
      env = self.env.unwrapped
      if env.command_manager.active_terms:
          self._server.gui.add_markdown("### Available Motion Clips")
          self._server.gui.add_markdown("_Select a reference trajectory clip for the policy._")
          env.command_manager.create_gui(self._server, lambda: self._scene.env_idx)

          # Inject policy hot-swap callbacks into any MotionCommand terms.
          if self._runner is not None:
            from mjlab.tasks.tracking.mdp.commands import MotionCommand as _MC
            for term in env.command_manager._terms.values():
              if isinstance(term, _MC):
                _runner = self._runner
                _viewer = self
                def _make_load_fn(r, v):
                  def _load(pt_path: str):
                    r.load(pt_path, load_cfg={"actor": True},
                           strict=True, map_location=r.device)
                    return r.get_inference_policy(device=r.device)
                  return _load
                term._load_policy_fn = _make_load_fn(_runner, _viewer)
                term._set_policy_fn  = lambda p: setattr(_viewer, "policy", p)

      # Video capture.
      self._server.gui.add_markdown("### Video Capture")
      self._is_recording = False
      self._record_frames = []
      self._record_button = self._server.gui.add_button(
        "Record Video", icon=viser.Icon.VIDEO,
      )

      @self._record_button.on_click
      def _(event) -> None:
        if not self._is_recording:
            self._is_recording = True
            self._record_frames = []
            self._initialize_renderer_request = True
            event.target.label = "Stop Recording"
            event.target.color = "red"
        else:
            self._is_recording = False
            event.target.label = "Saving..."
            event.target.disabled = True
            
            def save_video(frames, target):
                import time
                from pathlib import Path
                import mediapy as media
                import numpy as np
                
                if frames:
                    video_frames = []
                    for frame in frames:
                        frame = np.asarray(frame) if not isinstance(frame, np.ndarray) else frame
                        if frame.dtype != np.uint8:
                            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                        video_frames.append(frame)
                        
                    out_dir = Path("/home/acer/strike_demo_web/record_video")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = int(time.time())
                    out_path = out_dir / f"recording_{timestamp}.mp4"
                    media.write_video(str(out_path), video_frames, fps=60)
                    print(f"\\n[INFO] Saved video to {out_path.absolute()}")
                
                target.label = "Record Video"
                target.icon = viser.Icon.VIDEO
                target.color = None
                target.disabled = False
                
            import threading
            threading.Thread(target=save_video, args=(self._record_frames, event.target)).start()

      # Scene & Camera Controls.
      self._server.gui.add_markdown("### Camera Settings")
      with self._server.gui.add_folder("Scene Settings"):
        self._scene.create_visualization_gui(
          camera_distance=self.cfg.distance,
          camera_azimuth=self.cfg.azimuth,
          camera_elevation=self.cfg.elevation,
          show_debug_viz_control=False,
          debug_viz_extra_gui=None,
          show_contacts=False,
        )

      self._camera_overlays = ViserCameraOverlays(self._server, self.env, sim.mj_model)
      if self._camera_overlays.has_cameras:
        with self._server.gui.add_folder("Camera Feeds"):
          self._camera_overlays.setup_controls()

    # Interactive Joints.
    with tabs.add_tab("Pose", icon=viser.Icon.TOOL):
      self._server.gui.add_markdown("### Interactive Pose Editor")
      self._server.gui.add_markdown("_Pause the simulation to edit joint positions manually._")
      
      if hasattr(self.env.unwrapped, "scene") and "robot" in self.env.unwrapped.scene.entities:
          robot = self.env.unwrapped.scene["robot"]
          
          lower_limits = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
          upper_limits = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
          
          self._joint_sliders = []
          import numpy as np
          import torch
          initial_qpos = robot.data.joint_pos[0].clone()

          reset_pose_btn = self._server.gui.add_button("Reset to Zero Pose", icon=viser.Icon.REFRESH)
          
          @reset_pose_btn.on_click
          def _(_):
              if self._is_paused:
                  for i, s in enumerate(self._joint_sliders):
                      zero_val = float(np.clip(0.0, float(lower_limits[i]), float(upper_limits[i])))
                      s.value = zero_val
          
          with self._server.gui.add_folder("Joint Sliders"):
            for i, joint_name in enumerate(robot.joint_names):
                current_val = float(np.clip(0.0, float(lower_limits[i]), float(upper_limits[i])))
                initial_qpos[i] = current_val

                slider = self._server.gui.add_slider(
                    label=joint_name,
                    min=float(lower_limits[i]),
                    max=float(upper_limits[i]),
                    step=0.01,
                    initial_value=current_val
                )
                
                def make_callback(idx=i, s=slider):
                    @s.on_update
                    def _(_) -> None:
                        if self._is_paused:
                            import torch
                            current_qpos = robot.data.joint_pos[0].clone()
                            current_qpos[idx] = s.value
                            robot.write_joint_state_to_sim(
                                current_qpos.unsqueeze(0),
                                robot.data.joint_vel[[0]],
                                env_ids=torch.tensor([0], device=self.env.device)
                            )
                            sim = self.env.unwrapped.sim
                            sim.forward()
                            self._scene.needs_update = True
                            
                make_callback()
                self._joint_sliders.append(slider)
              
          robot.write_joint_state_to_sim(
              initial_qpos.unsqueeze(0),
              robot.data.joint_vel[[0]],
              env_ids=torch.tensor([0], device=self.env.device)
          )
          sim = self.env.unwrapped.sim
          sim.forward()
          self._scene.needs_update = True
      else:
          self._server.gui.add_markdown("_Robot entity not found for joint controls._")

    self._prev_env_idx = self._scene.env_idx

    self._term_overlays = ViserTermOverlays(self._server, self.env, self._scene)
    # Rewards/Metrics tabs hidden intentionally.
    self._debug_overlays = ViserDebugOverlays(self.env, self._scene)
    self._contact_overlays = ViserContactOverlays(self._scene)

  @override
  def _process_actions(self) -> None:
    """Process queued actions and sync UI state."""
    had_actions = bool(self._actions)
    super()._process_actions()
    if had_actions:
      self._pending_update_reasons.add(UpdateReason.ACTION)
      self._sync_ui_state()

  def _sync_ui_state(self) -> None:
    """Sync UI elements to current state after action processing."""
    self._pause_button.label = "Play" if self._is_paused else "Pause"
    self._pause_button.icon = (
      viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE
    )
    self._update_status_display()

  def _update_env_dependent_plots(self) -> None:
    """Refresh reward/metric plots and histories for the selected environment."""
    if self._scene.env_idx != self._prev_env_idx:
      self._prev_env_idx = self._scene.env_idx
      self._pending_update_reasons.add(UpdateReason.ENV_SWITCH)
      if self._term_overlays:
        self._term_overlays.on_env_switch()
      if self._debug_overlays:
        self._debug_overlays.on_env_switch()
      if self._contact_overlays:
        self._contact_overlays.on_env_switch()

    if self._term_overlays:
      self._term_overlays.update(self._is_paused)

  def _update_camera_feeds(self, sim: Simulation, has_pending_updates: bool) -> None:
    """Push camera sensor frames to GUI when needed."""
    t0 = time.perf_counter()
    if self._camera_overlays and self._should_update_cameras(
      self._is_paused, has_pending_updates
    ):
      self._camera_overlays.update(
        sim.data, self._scene.env_idx, self._scene._scene_offset
      )
    self._camera_update_last_ms = (time.perf_counter() - t0) * 1000.0

  def _queue_debug_visualizers(self) -> None:
    """Queue environment-specific debug draw calls into the scene.

    Acquires ``_sim_lock`` so the clear+requeue is atomic with respect
    to the background thread that reads the queues in ``scene.update``.
    """
    t0 = time.perf_counter()
    if self._debug_overlays:
      with self._sim_lock:
        self._debug_overlays.queue()
    self._debug_queue_last_ms = (time.perf_counter() - t0) * 1000.0

  def _submit_scene_update_if_needed(
    self, sim: Simulation, has_pending_updates: bool
  ) -> None:
    """Submit a scene sync job when the update policy allows it."""
    t_enqueue_start = time.perf_counter()
    if self._scene.needs_update:
      self._pending_update_reasons.add(UpdateReason.SCENE_REQUEST)

    if not self._should_submit_scene_update(
      self._counter, self._is_paused, has_pending_updates
    ):
      self._scene_submit_enqueue_last_ms = 0.0
      return

    def update_scene() -> None:
      with self._sim_lock:
        t0 = time.perf_counter()
        with self._server.atomic():
          self._scene.update(sim.data)
          self._server.flush()
        self._scene_update_last_ms = (time.perf_counter() - t0) * 1000.0

    self._threadpool.submit(update_scene)
    self._scene_submit_enqueue_last_ms = (
      time.perf_counter() - t_enqueue_start
    ) * 1000.0
    self._pending_update_reasons.clear()
    self._scene.needs_update = False

  def _maybe_log_debug_timings(self) -> None:
    """Log lightweight Viser pipeline timing in debug mode."""
    if self.verbosity < VerbosityLevel.DEBUG:
      return
    now = time.time()
    if now - self._timing_last_log_time < 1.0:
      return
    self._timing_last_log_time = now
    self.log(
      (
        "[DEBUG] Viser timings: "
        f"camera={self._camera_update_last_ms:.2f}ms, "
        f"debug={self._debug_queue_last_ms:.2f}ms, "
        f"submit_enqueue={self._scene_submit_enqueue_last_ms:.2f}ms, "
        f"scene_update={self._scene_update_last_ms:.2f}ms"
      ),
      VerbosityLevel.DEBUG,
    )

  @staticmethod
  def _should_update_cameras(paused: bool, has_pending_updates: bool) -> bool:
    """Camera feeds update continuously while running and on-demand while paused."""
    return (not paused) or has_pending_updates

  @staticmethod
  def _should_submit_scene_update(
    counter: int, paused: bool, has_pending_updates: bool
  ) -> bool:
    """Scene submits at 30Hz (every other 60Hz tick) with pause-aware gating."""
    if counter % 2 != 0:
      return False
    if paused and not has_pending_updates:
      return False
    return True

  @override
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    self._scene.paused = self._is_paused
    self._counter += 1
    if self._counter % 10 == 0:
      self._update_status_display()
    self._update_env_dependent_plots()
    has_pending_updates = bool(self._pending_update_reasons) or self._scene.needs_update
    self._update_camera_feeds(sim, has_pending_updates)
    # Queue debug visualizers only when a scene update will actually be
    # submitted.  Clearing the queues on skipped ticks creates a race
    # with the background thread that causes debug overlays to blink.
    if getattr(self, "_initialize_renderer_request", False):
        self._initialize_renderer_request = False
        env_unwrapped = self.env.unwrapped
        if getattr(env_unwrapped, "_offline_renderer", None) is None:
            from mjlab.viewer.offscreen_renderer import OffscreenRenderer
            sim = env_unwrapped.sim
            # Force 720p configuration for the video recording
            env_unwrapped.cfg.viewer.width = 1280
            env_unwrapped.cfg.viewer.height = 720
            renderer = OffscreenRenderer(
                model=sim.mj_model, cfg=env_unwrapped.cfg.viewer, scene=env_unwrapped.scene
            )
            renderer.initialize()
            env_unwrapped._offline_renderer = renderer
            env_unwrapped.render_mode = "rgb_array"

    if getattr(self, "_is_recording", False) and not self._is_paused:
        frame = self.env.unwrapped.render()
        if frame is not None:
            import numpy as np
            rgb_frame = frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
            self._record_frames.append(rgb_frame)

    will_submit = self._should_submit_scene_update(
      self._counter, self._is_paused, has_pending_updates
    )
    if will_submit:
      self._queue_debug_visualizers()
    self._submit_scene_update_if_needed(sim, has_pending_updates)
    self._maybe_log_debug_timings()

  @override
  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (e.g., perturbations)."""
    pass

  @override
  def reset_environment(self) -> None:
    """Extend BaseViewer.reset_environment to clear reward and metrics histories."""
    with self._sim_lock:
      super().reset_environment()
    if self._term_overlays:
      self._term_overlays.clear_histories()

  @override
  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    if self._term_overlays:
      self._term_overlays.cleanup()
    if self._camera_overlays:
      self._camera_overlays.cleanup()
    self._threadpool.shutdown(wait=True)
    if not self._external_server:
      self._server.stop()

  @override
  def is_running(self) -> bool:
    """Check if viewer is running."""
    return True  # Viser runs until process is killed.

  def _update_status_display(self) -> None:
    """Update the HTML status display."""
    status = self.get_status()
    actual_rt = status.actual_realtime
    rt_display = f"{actual_rt:.2f}x" if actual_rt > 0 else "—"
    capped = ' <span style="color:#e74c3c;">[CAPPED]</span>' if status.capped else ""
    error_line = ""
    if status.last_error:
      # Show last line of traceback to avoid flooding the panel.
      first_line = status.last_error.strip().splitlines()[-1]
      error_line = (
        f'<br/><span style="color:#e74c3c;"><strong>Error:</strong> {first_line}</span>'
      )
    self._status_html.content = f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
        <strong>Status:</strong> {"Paused" if status.paused else "Running"}{capped}<br/>
        <strong>Steps:</strong> {status.step_count}<br/>
        <strong>Speed:</strong> {status.speed_label}<br/>
        <strong>Target RT:</strong> {status.target_realtime:.2f}x<br/>
        <strong>Actual RT:</strong> {rt_display} ({status.smoothed_fps:.0f} FPS){error_line}
      </div>
      """
