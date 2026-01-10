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

