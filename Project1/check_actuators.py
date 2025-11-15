#!/usr/bin/env python3
import mujoco

model = mujoco.MjModel.from_xml_path("integrated_model.xml")
print(f"\nTotal actuators: {model.nu}")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i}: {name}")
