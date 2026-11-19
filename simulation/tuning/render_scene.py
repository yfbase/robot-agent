"""Render both cameras of the tennis-pick scene to PNG for layout inspection.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/render_scene.py
"""

from pathlib import Path

import cv2
import mujoco

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "tuning" / "renders"


def main():
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    OUT.mkdir(exist_ok=True)
    renderer = mujoco.Renderer(model, 480, 640)
    for cam in ("top_view", "side_view"):
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        path = OUT / f"{cam}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("wrote", path)
    print("gripperframe:", data.site("gripperframe").xpos)


if __name__ == "__main__":
    main()
