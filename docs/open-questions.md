# Open questions for Ömer (day 1)

## Decided
- **Primary camera: OAK-D.** Drivers + DepthAI v3 SDK installed; device
  enumerates and streams RGB at ~30 FPS (640×400).
- **Training machine: laptop RTX 4070 (8 GB), venv ready, GPU verified.**

## Environment / infra
- [ ] Is there a standard company setup doc / preferred machine?
- [ ] Network / proxy restrictions on `pip` / model downloads (Roboflow,
      blobconverter's online converter — it calls a hosted service)?
- [ ] Jetson Nano: which JetPack version is flashed? Still in scope, or is the
      OAK-D VPU the only edge target for the demo?
- [ ] OAK-D is enumerating at **USB2 (HIGH) speed** — need the USB3 cable + a
      blue USB3 port for full RGB res / depth bandwidth. Is a USB3 port free on
      the target host?

## Project scope
- [ ] Approve the project idea and the verdict set (`docs/scenarios.md`).
- [ ] Space for a fixed camera rig — where, for how long?
- [ ] Deployment target: laptop-free / fully embedded (needs a host for the
      state logic), or OAK-D + laptop acceptable for the demo?
- [ ] Use OAK-D **depth** to help separate overlapping objects, or RGB only?

## Data
- [ ] Existing dataset available, or collect from scratch?
- [ ] Roboflow: company workspace/account, or personal?
- [ ] Any constraint on what the 3 objects can be / be shown as?

## Deliverables
- [ ] Expected form of the write-up (feeds Ömer's Edge AI article series?).
- [ ] C++ inference comparison — required, or nice-to-have for the last days?
