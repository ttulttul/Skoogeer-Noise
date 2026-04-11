# The Atomic Prompting Method: A Guide to Building Infinite, Recursive AI Prompt Templates

This guide discusses how to get the most out of the `Mustache Template` and
`Mustache Variables` nodes within `Skoogeer-Noise`.

## Introducing: **The Atomic Prompting Method**

Want ridiculously diverse prompts that work well with newer flow matching
models like the Klein, Z.Image and Qwen Image families? **Atomic Prompting**
delivers this by deconstructing prompts into their most basic components (Atoms)
and recursively combining them into highly complex, non-repetitive descriptions
(Molecules and Scenes). 

This guide will teach you how to build a highly optimized, DRY (Don't Repeat
Yourself) prompt template architecture that generates millions of unique,
structurally sound prompts without breaking grammar.

---

## The 4-Level Architecture

Atomic Prompting is built on a strict hierarchy to prevent logic loops,
grammatical errors, and repetitive phrasing (tautologies). 

*   **Level 4 (Atoms):** Raw, single-concept word pools (colors, materials, adjectives, verbs).
*   **Level 3 (Molecules):** Small combinations of Atoms (a specific fabric + a color to make a piece of clothing).
*   **Level 2 (Entities & Actions):** Complex subjects, actions, and spatial relationships.
*   **Level 1 (Scene Components):** Complete, grammatically structured clauses. Enforces a "Division of Labor" so details aren't repeated.
*   **Level 0 (The Engine):** The final prompt assembly and probability controller.

---

## Step 1: Define Your Atoms (Level 4)

Atoms are the raw building blocks. The golden rule of Level 4 is to **never
include grammar, articles ("a", "an", "the"), or overlapping concepts.**

For example, instead of writing *"a rusted iron wall"*, break it down:

```yaml
# Paste this into your Mustache Variables node:

color_base:
  - "red"
  - "blue"
  - "neon green"

material_base:
  - "iron"
  - "concrete"
  - "glass"

material_modifier:
  - "rusted"
  - "polished"
  - "shattered"
```

### The "Wet and Wet" Problem (Combinatorial Collision)

If you put "wet" and "sweaty" in a `skin_state` pool, and your template says
`{{skin_state}} and {{skin_state}}`, you will eventually roll *"wet and wet"*.

**Solution:** Split overlapping concepts into distinct pools based on their
sub-type.

```yaml
moisture_state:  # The liquid condition
  - "wet"
  - "sweaty"
  - "damp"

moisture_finish: # The light reflection
  - "glistening"
  - "glossy"
  - "shiny"
```

---

## Step 2: Build Your Molecules (Level 3)

Molecules use your Atoms to create specific, highly variable concepts. By
mixing single variables and compound variables, you allow the engine to
generate simple or wildly complex descriptions.

*Note: In this guide, plain `{{variable}}` already uses lazy randomization by default. Use `:static` only when you want to bind to an already-resolved value instead of making a fresh lazy choice.*

```yaml
dominant_color:
  - "Crimson"
  - "Deep Blue"
  - "Neon Green"

skin_state:
  - "{{moisture_state}}"
  - "{{moisture_finish}}"
  - "{{moisture_state}} and {{moisture_finish}}" # Safe! Cannot output "wet and wet"

base_surface:
  - "{{material_base}}"
  - "{{material_modifier}} {{material_base}}"
  - "{{color_base}} {{material_base}}"

wardrobe:
  - "{{dominant_color:static,lowercase}} satin dress"
  - "{{dominant_color:static,lowercase}} leather corset"

accent_ribbon:
  - "{{dominant_color}} ribbon with {{dominant_color:repeat}} stitching"
```

Notice the operator choices:

*   `:static,lowercase` keeps `wardrobe` locked to the already-resolved `dominant_color`, but formats it naturally for mid-sentence use.
*   `:repeat` makes `"Crimson ribbon with Crimson stitching"` possible without a second random draw drifting to a different color.

---

## Step 3: Entities & Actions (Level 2)

Now we define *who* is in the scene and *what* they are doing. Always decouple
your verbs from your adverbs and targets to maximize variety.

```yaml
action_verb:
  - "leans"
  - "arches"
  - "presses"

action_modifier:
  - "forward"
  - "backwards"
  - "heavily against the wall"

posture_dynamic:
  - "{{action_verb}} {{action_modifier}}"

paired_subject:
  - "a {{subject_role}} and her {{subject_role:repeat}} counterpart"
```

Here `paired_subject` uses `:repeat` so a prompt can stay internally coherent when the same lazy choice should show up twice.

---

## Step 4: Scene Components & Division of Labor (Level 1)

**This is the most critical step.** Level 1 builds the actual sentences. You
must enforce a strict "Division of Labor" to prevent the template from
describing the same thing twice.

If you have a variable for `woman_look` and a variable for `duo_action`, make
sure `duo_action` **never** calls the wardrobe variables, or your prompt might
output: *"She wears a red bra. They kiss, her red bra catching the light."*

```yaml
# Strict Focus: Looks & Wardrobe
woman_look:
  - "{{spatial_position:propercase}}, a {{woman}} features {{hair_style}}, dressed in {{wardrobe}} with {{dominant_color:static,lowercase}} lipstick."

# Strict Focus: Actions
woman_solo_action:
  - "She {{posture_dynamic}}, {{facial_expression}}."

# Strict Focus: Environment
backdrop:
  - "set against {{base_surface}} walls in a {{atmosphere}}."

# Strict Focus: Micro details that need whitespace glue
lens_type:
  - "50mm lens"
  - "85mm lens"

lens_modifier:
  - "cracked "
  - "fogged "
  - "dusty "

lens_detail:
  - "{{lens_modifier:notrim}}{{lens_type}}"
```

This is where operators stop being cosmetic and start protecting grammar:

*   `:propercase` is perfect for sentence starts.
*   `:static,lowercase` keeps a reused resolved value readable in the middle of a clause.
*   `:notrim` lets you intentionally preserve glue spaces when two variables need to concatenate into one phrase.

---

## Step 5: The Final Generator (Level 0)

Level 0 strings your Level 1 components together into the final payload. This
is also where you control the **probability** of different scenes occurring
using weights (e.g., `:0.7` for 70%).

```yaml
# 1. Define the scene paths
women_only_prompt:
  - "{{woman_look}} {{woman_solo_action}} {{backdrop}}"

mixed_couple_prompt:
  - "A man {{man_action}} a woman. {{woman_look}} {{backdrop}}"

# 2. Control the probabilities
generated_prompt:
  - "{{women_only_prompt}}:0.8"   # 80% chance of a women-only scene
  - "{{mixed_couple_prompt}}:0.2" # 20% chance of a mixed couple scene

# 3. The Final Output
final_prompt:
  - "({{generated_prompt}}) ({{cinematography_style}}) (shot on {{camera_family:uppercase}})"
```

---

## Advanced Mustache Operators

To make your templates truly bulletproof, you must utilize inline operators.
You can append these to any variable call, and even chain them together with
commas (e.g., `{{variable:static,lowercase}}`).

### 1. Behavior Operators
*   `{{color}}` or `{{color:randomize}}` - *(Default)* Draws a fresh, random value from the list every single time it is called during lazy expansion.
*   `{{color:static}}` - Uses the already-resolved value for this variable instead of making a fresh lazy choice. If no resolved value exists yet, `static` cannot invent one.
*   `{{color:repeat}}` - Reuses the *exact same value* that was randomly chosen for this variable earlier in the prompt. 
    *   *Use Case:* Matching details. `"A {{race_ethnicity}} woman and her {{race_ethnicity:repeat}} sister."` (Ensures both women generate with the same ethnicity).

### 2. Formatting & Grammar Operators
*   `{{word:lowercase}}` - Forces the output to lowercase. Great for injecting nouns mid-sentence.
*   `{{word:propercase}}` - Capitalizes the first letter. Ideal when a variable happens to start a new sentence.
*   `{{word:uppercase}}` - Forces the output to ALL CAPS.
*   `{{word:notrim}}` - By default, engines trim leading and trailing whitespace from variables. If you intentionally leave a space inside your variable (e.g., `- "cracked "`), use `:notrim` so it blends perfectly into the next word: `{{modifier:notrim}}{{lens_type}}` outputs `"cracked 50mm lens"`.

### 2.5 Practical Operator Patterns
These patterns are where the operators become genuinely useful:

```yaml
headline:
  - "{{scene_type:propercase}} under {{lighting_color:lowercase}} light"

matched_palette:
  - "{{dominant_color}} silk with {{dominant_color:repeat}} satin trim"

coherent_makeup:
  - "{{dominant_color:static,lowercase}} lipstick"

technical_tag:
  - "{{camera_family:uppercase}} render"

joined_phrase:
  - "{{surface_prefix:notrim}}{{surface_material}}"
```

Interpretation:

*   `propercase` fixes sentence openings without forcing you to capitalize the whole source pool.
*   `lowercase` makes title-cased pools safe to inject into clauses.
*   `repeat` is for "same fresh choice again in this one template."
*   `static` is for "reuse the resolved variable that already belongs to this sampled setting."
*   `uppercase` is useful for technical tags, camera families, or stylized labels.
*   `notrim` is for deliberate token gluing, not general text cleanup.

### 3. Probability Weights
You can attach weights to individual list items by appending `:probability`. Fully weighted lists must sum to `1.0`. If only some entries are weighted, the remaining probability mass is split evenly across the unweighted entries.

```yaml
lens_effect:
  - " with shallow focus":0.2
  - " with motion blur":0.1
  - ":0.7"  # 70% chance of outputting absolutely nothing!
```

---

## Best Practices & Pitfall Avoidance

1.  **Watch out for Root Word Clashes:** If you hardcode `"reflecting the environment"` in Level 1, make sure the word `"reflective"` doesn't exist in your Level 4 pools, or you will generate *"reflective skin reflecting the environment"*.
2.  **Null Weights are your Friend:** Always include empty strings (`- ":0.5"`) in modifier lists. If every single item always has a modifier, your prompts become bloated and look "AI-generated." Silence is just as important as description.
3.  **Use Compound Nodes for High-End AI Triggers:** AI models love contrasting colors. Instead of just picking one color for lighting, build a molecule that occasionally feeds the AI two colors (e.g., `{{color}} and {{color}} lighting`). This triggers advanced cinematic grading in models like Midjourney and Stable Diffusion.
4.  **Reserve `static` for consistency, not variety:** If two clauses should talk about the same already-sampled concept, use `:static`. If you actually want a new roll, leave the placeholder plain.
5.  **Reserve `repeat` for same-template echoes:** `repeat` only makes sense after the same variable has already been drawn earlier in the same lazy render. Use it when you want "the same random choice again right now," not when you want to bind to a top-level resolved field.
