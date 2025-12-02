# Dataset Preparation Template

This document serves as a template for possible matching terms associated with the variables that connect to the `.csv` fields when building the annotations for training an image LoRA model using `dataset_prep_for_lora_training.py`.

Use these terms to populate the `STRONG_TERMS` dictionary in the main Python script. These terms act as filters for custom datasets.

**Note:** Core Categories and Tags work best as filters, while the other options (Attributes, Mood, etc.) are generally better used as modifiers or not included in the strict filter list at all.

---

## Field Reference

### **Primary Description Fields**

```python
PRIMARY_DESCRIPTION_FIELDS = [
    "llm_description",
    "human_description",
]
```

### **Term Source Fields**

```python
TERM_SOURCE_FIELDS = [
    "category",
    "attributes",
    "tags",
    "composition",
    "mood",
    "palette",
    "notes",
    "human_description",
    "llm_description",
]
```

---

## Strong Terms (Examples)

Copy and paste relevant sections below into the `STRONG_TERMS` dictionary in `dataset_prep_for_lora_training.py`.

Format:  
`"Term": (base_weight, [synonyms])`

```python
STRONG_TERMS = {   
    # ===== Core Categories =====
    "Abstract": (1.05, ["abstract"]),
    "Birds": (1.10, ["birds", "bird", "avian"]),
    "Canyons": (1.10, ["canyons", "canyon", "chasm"]),
    "Carl Sagan": (1.50, ["carl_sagan", "carl sagan", "sagan"]),
    "Columbia alternata": (1.60, ["columbia_alternata", "columbia alternata"]),
    "Darkness": (1.05, ["darkness", "dark"]),
    "Deity": (1.30, ["deity", "god", "goddess", "divine being", "divine"]),
    "Fantasy": (1.10, ["fantasy", "fantastical"]),
    "Futurism": (1.10, ["futurism", "futuristic", "sci-fi", "scifi"]),
    "Honest Abe": (1.40, ["honest_abe", "honest abe", "abraham lincoln", "abe lincoln"]),
    "Interiors": (1.05, ["interiors", "interior", "indoors", "indoor"]),
    "Lakes": (1.10, ["lakes", "lake"]),
    "Liminal": (1.25, ["liminal", "liminal space"]),
    "Misc": (1.00, ["misc", "miscellaneous"]),
    "Pyramids": (1.40, ["pyramids", "pyramid", "giza", "egyptian_pyramids", "egyptian pyramids"]),
    "Rollercoasters": (1.25, ["rollercoasters", "rollercoaster", "coaster"]),
    "Scenery": (1.05, ["scenery", "scenic"]),
    "Skies": (1.10, ["skies", "sky", "heavens"]),
    "Space": (1.40, ["space", "cosmic", "galactic", "astral", "celestial"]),
    "Stone ocean": (1.70, ["stone_ocean", "stone ocean"]),
    "Structures": (1.15, ["structures", "structure", "building", "buildings"]),
    "Temple": (1.25, ["temple", "temples", "shrine"]),
    "The appearing woman": (1.05, ["the_appearing_woman", "the appearing woman", "appearing woman"]),
    "The Bellman": (1.40, ["the_bellman", "the bellman", "bellman"]),
    "Trees": (1.25, ["trees", "tree", "forest", "woodland"]),
    "Other category": (1.00, ["other_category", "other category"]),

    # ===== Tags =====
    "Robot": (1.50, ["robot", "android", "mech", "mecha"]),
    "Pyramid": (1.35, ["pyramid", "pyramids"]),
    "Electricity": (1.50, ["electricity", "electric", "lightning", "arc", "arcs"]),
    "Portal": (1.30, ["portal", "stargate", "gate", "wormhole", "doorway"]),
    "Architecture": (1.25, ["architecture", "architectural"]),
    "Landscape": (1.10, ["landscape", "landscapes"]),
    "Turtle": (1.05, ["turtle", "tortoise"]),
    "Cityscape": (1.10, ["cityscape", "city", "skyline", "urban"]),
    "Cosmic": (1.08, ["cosmic"]),
    "Angel": (1.30, ["angel", "seraph", "seraphim"]),
    "Heart": (1.10, ["heart", "hearts"]),
    "Glyph": (1.50, ["glyph", "rune", "sigil", "inscription"]),
    "Mask": (1.05, ["mask", "masked"]),
    "Other tags": (1.00, ["other_tags", "other tags"]),

    # ===== Attributes =====
    "Realistic": (1.50, ["realistic", "photo-realistic", "photorealistic", "photo realistic"]),
    "Painterly": (1.10, ["painterly", "painted"]),
    "Vector flat": (1.00, ["vector_flat", "vector flat", "vector", "flat color", "flat-colour"]),
    "Comic / manga": (1.00, ["comic_manga", "comic / manga", "comic", "manga"]),
    "Photobash": (1.08, ["photobash", "photo-bash", "photo bash"]),
    "Fractal": (1.08, ["fractal", "fractals"]),
    "Pixel art": (1.00, ["pixel", "pixel_art", "pixel art"]),
    "3D render": (1.05, ["render_3d", "3d render", "3d-render", "cg render", "cg"]),
    "Other attributes": (1.00, ["other_attributes", "other attributes"]),

    # ===== Mood =====
    "Sacred": (1.50, ["sacred"]),
    "Eerie": (1.45, ["eerie", "uncanny"]),
    "Hopeful": (1.12, ["hopeful", "uplifting"]),
    "Melancholic": (1.12, ["melancholic", "melancholy"]),
    "Playful": (1.12, ["playful"]),
    "Epic": (1.50, ["epic", "grand"]),
    "Other mood": (1.00, ["other_mood", "other mood"]),

    # ===== Palette =====
    "Blue": (1.30, ["blue", "azure", "cobalt", "cerulean"]),
    "Green": (1.20, ["green", "emerald", "verdant", "jade"]),
    "Gold": (1.30, ["gold", "golden"]),
    "Neon": (1.15, ["neon", "glow", "glowing"]),
    "Pastel": (1.05, ["pastel", "soft palette"]),
    "Monochrome": (1.10, ["monochrome", "black and white", "grayscale", "greyscale"]),
    "High contrast": (1.12, ["high_contrast", "high contrast"]),
    "Other palette": (1.00, ["other_palette", "other palette"]),

    # ===== Composition =====
    "Close-up": (1.10, ["close_up", "close up", "close-up", "macro", "tight shot"]),
    "Medium": (1.20, ["medium", "mid shot", "waist-up"]),
    "Wide": (1.25, ["wide", "wide shot", "ultra wide"]),
    "Aerial": (1.10, ["aerial", "top-down", "top down", "bird's-eye", "birds-eye"]),
    "Isometric": (1.10, ["isometric", "iso"]),
    "Symmetrical": (1.08, ["symmetrical", "symmetry", "symmetrical composition"]),
    "Other composition": (1.00, ["other_composition", "other composition"]),
}
```

---

## Quality Scrubbing Lists

These terms help remove low‑quality artifacts from descriptions.

```python
ISSUE_TERMS = {
    "blur", "blurry", "blurred", "blur_noise", "noise", "noisy",
    "artifact", "artifacts", "artifacting",
    "watermark", "watermarked", "watermark_text",
    "cropped", "crop",
    "duplicate", "duplicated",
    "jpeg", "compression", "banding", "pixelation", "pixelated",
}

ISSUE_FIELDS = ["issue"]
ISSUE_BLOCKLIST = {"blur_noise", "artifacting", "watermark_text", "duplicate"}
```
