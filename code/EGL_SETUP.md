# EGL setup guidance

EGL is a system-level dependency that must be installed via your operating system's package manager rather than by Python code. The training code only **detects** EGL and sets `MUJOCO_GL=egl` when the library is available; it does not download or install the underlying drivers.

## Linux (headless servers)
1. Install Mesa's EGL support and a software OpenGL stack:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libegl1-mesa-dev libgl1-mesa-dev mesa-utils-extra
   ```
2. Ensure `MUJOCO_GL` is set for MuJoCo tasks (the code will default to `egl` when it finds the library):
   ```bash
   export MUJOCO_GL=egl
   ```
3. You can verify the installation with the new unit test:
   ```bash
   pytest code/tests/unit/test_mujoco_egl_setup.py
   ```

On macOS or Windows, EGL is not required; leave `MUJOCO_GL` unset and rely on the platform's native OpenGL stack.
