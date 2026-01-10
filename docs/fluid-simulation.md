# Node Concept

**Node name**

```
Fluid Image Advection
```

**Purpose**
Takes an input image and simulates fluid-like motion by injecting forces (“sticks”) into a velocity field and advecting the image forward in time.

This is an **image deformation node**, not a physically accurate CFD solver.

---

# Node Interface (ComfyUI style)

### Inputs

| Name             | Type  | Description                              |
| ---------------- | ----- | ---------------------------------------- |
| image            | IMAGE | Input image to distort                   |
| steps            | INT   | Number of simulation steps               |
| dt               | FLOAT | Time step size                           |
| resolution_scale | FLOAT | Internal sim resolution (0.25–1.0)       |
| force_count      | INT   | Number of force injections per step      |
| force_strength   | FLOAT | Magnitude of stick forces                |
| force_radius     | FLOAT | Radius of influence (pixels, normalized) |
| swirl_strength   | FLOAT | Rotational component of forces           |
| velocity_damping | FLOAT | Velocity decay per step                  |
| diffusion        | FLOAT | Optional velocity diffusion              |
| vorticity        | FLOAT | Curl enhancement                         |
| seed             | INT   | Random seed                              |
| wrap_mode        | ENUM  | clamp / wrap / mirror                    |

---

### Outputs

| Name                        | Type  | Description           |
| --------------------------- | ----- | --------------------- |
| image                       | IMAGE | Distorted image       |
| velocity_preview (optional) | IMAGE | Visualization of flow |

---

# Internal Data Model

### Simulation grid

Resolution:

```
sim_width  = image_width  * resolution_scale
sim_height = image_height * resolution_scale
```

### Fields

```python
velocity: float32[H, W, 2]
color:    float32[H, W, 3 or 4]
```

* Velocity stored in pixels per step
* Color sampled from image

---

# Core Algorithm

## Initialization

```python
rng = Random(seed)
velocity = zeros(H, W, 2)
color = downsample(image)
```

---

## Per-Step Loop

```pseudo
for step in range(steps):

    # 1. Inject forces ("sticks")
    for i in range(force_count):
        pos = random_position()
        dir = random_unit_vector()
        apply_force(
            velocity,
            pos,
            dir,
            force_strength,
            force_radius,
            swirl_strength
        )

    # 2. Optional velocity diffusion
    velocity = diffuse(velocity, diffusion)

    # 3. Vorticity confinement
    if vorticity > 0:
        velocity += vorticity_force(velocity) * vorticity

    # 4. Advect velocity
    velocity = advect(velocity, velocity, dt)

    # 5. Apply damping
    velocity *= velocity_damping

    # 6. Advect color
    color = advect(color, velocity, dt)
```

---

# Key Subsystems

## Force Injection ("Sticks")

```pseudo
for each cell x:
    r = x - pos
    d = length(r)
    if d < radius:
        falloff = exp(-(d/radius)^2)
        tangent = perpendicular(r)
        velocity[x] += (
            dir * strength +
            tangent * swirl_strength
        ) * falloff
```

This produces:

* Linear dragging
* Circular swirling
* Organic turbulence

---

## Semi-Lagrangian Advection

```pseudo
def advect(field, velocity, dt):
    for each cell x:
        prev_pos = x - velocity[x] * dt
        field_out[x] = bilinear_sample(field, prev_pos)
    return field_out
```

Stable, fast, and ideal for ComfyUI usage.

---

## Vorticity Confinement (Optional)

```pseudo
curl = ∂v_y/∂x - ∂v_x/∂y
N = normalize(∇|curl|)
force = perpendicular(N) * curl
```

Enhances swirling motion without instability.

---

# Boundary Modes

* **Clamp**: Sample edges
* **Wrap**: Toroidal domain
* **Mirror**: Reflect edges

Used during advection sampling.

---

# Performance Notes

* Resolution scaling is **critical**

* Typical good defaults:

  ```
  resolution_scale = 0.5
  steps = 10–30
  force_count = 2–5
  ```

* Vectorized NumPy or Torch implementation

* CUDA acceleration possible but not required

---

# ComfyUI Node Skeleton (Pseudo-Python)

```python
class FluidImageAdvection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "steps": ("INT", {"default": 20}),
                "dt": ("FLOAT", {"default": 1.0}),
                "resolution_scale": ("FLOAT", {"default": 0.5}),
                "force_count": ("INT", {"default": 3}),
                "force_strength": ("FLOAT", {"default": 5.0}),
                "force_radius": ("FLOAT", {"default": 0.1}),
                "swirl_strength": ("FLOAT", {"default": 2.0}),
                "velocity_damping": ("FLOAT", {"default": 0.98}),
                "diffusion": ("FLOAT", {"default": 0.0}),
                "vorticity": ("FLOAT", {"default": 0.0}),
                "seed": ("INT", {"default": 0}),
                "wrap_mode": (["clamp", "wrap", "mirror"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/warp"

    def run(self, image, **params):
        # Convert image → tensor
        # Run fluid sim
        # Upsample result
        return (output_image,)
```

---

# Extensions (Future Nodes)

* **External velocity input**
* **Mask-controlled forces**
* **User-drawn stick paths**
* **Temporal continuity across frames**
* **Pressure projection toggle**
* **GPU shader backend**

---

## Smoke Simulation Parameters (Optional Mode)

### Overview

Smoke is simulated by extending the fluid system with **buoyancy-driven motion** and **density evolution**. The image (or a derived density field) is treated as smoke density that rises due to buoyancy, cools over time, and forms characteristic billows through vorticity.

This mode prioritizes **visual plausibility and control**, not physical accuracy.

---

### Additional Fields

When `mode = "smoke"` is enabled, the solver maintains:

* **Density field** `ρ(x, y)`

  * Represents smoke concentration
    * Typically derived from image luminance or alpha

    * **Temperature field** `T(x, y)` (optional, but recommended)

      * Drives stronger initial upward motion
        * Decays over time (cooling)

        If temperature is disabled, buoyancy can be driven by density alone.

        ---

### New Node Parameters

| Parameter             | Type  | Description                      |
| --------------------- | ----- | -------------------------------- |
| smoke_mode            | BOOL  | Enables smoke simulation         |
| buoyancy              | FLOAT | Strength of upward buoyant force |
| ambient_updraft       | FLOAT | Constant upward airflow          |
| density_fade          | FLOAT | Density dissipation per step     |
| temperature_strength  | FLOAT | Initial temperature injection    |
| cooling_rate          | FLOAT | Temperature decay per step       |
| smoke_source_strength | FLOAT | Amount of density injected       |
| smoke_source_radius   | FLOAT | Radius of smoke injection        |
| smoke_source_mode     | ENUM  | image / random / mask            |

---

### Buoyancy Force Model

At each simulation step, a buoyancy force is added to the velocity field:

```pseudo
F_buoyancy = (β * T - α * ρ) * up_vector
velocity += F_buoyancy * dt
```

Where:

* `β` is controlled by **buoyancy**
* `α` is implicitly tied to density
* `up_vector = (0, -1)` in image space

This produces:

* Rising smoke columns
* Natural billowing
* Self-organizing plumes

---

### Density & Temperature Injection

Smoke can be injected in several ways:

#### 1. Image-derived (default)

* Density initialized from image luminance or alpha
* Bright areas → denser smoke

#### 2. Random sources

* Small stochastic puffs injected per step

#### 3. Mask-based

* External mask controls emission region

Example injection:

```pseudo
ρ += smoke_source_strength * falloff(radius)
T += temperature_strength * falloff(radius)
```

---

### Per-Step Smoke Reveal Order

When smoke mode is active, the simulation loop becomes:

```pseudo
inject_density()
inject_temperature()

apply_buoyancy()
apply_ambient_updraft()
apply_vorticity_confinement()

advect_velocity()
advect_density()
advect_temperature()

cool_temperature()
fade_density()
```

This order is important for maintaining coherent plumes.

---

### Parameter Defaults (Good Starting Points)

**Soft smoke**

```
buoyancy = 1.5
ambient_updraft = 0.1
vorticity = 0.3
density_fade = 0.01
cooling_rate = 0.05
```

**Fire / hot smoke**

```
buoyancy = 3.0
ambient_updraft = 0.3
vorticity = 0.8
temperature_strength = 2.0
cooling_rate = 0.1
```

---

### Interaction with Existing Parameters

* **Vorticity** becomes more visually important
* **Velocity damping** should be lower
* **Diffusion** should be minimal or zero
* **Resolution scaling** strongly affects plume detail

---

### Output Interpretation

* Density modulates image opacity or brightness
* Velocity field shapes plume motion
* Final image may be:

  * Density-only (grayscale smoke)
    * Density composited over original image
      * Color advected smoke

      ---

### Design Rationale

* Buoyancy explains *why* smoke rises
* Density creates structure
* Temperature creates energy and life
* Parameters map cleanly to artist intuition
* Fully compatible with the existing solver

