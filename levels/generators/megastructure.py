"""Generate levels/megastructure.json — nine districts, nine shape languages.

    python3 levels/generators/megastructure.py

The JSON is the artefact and is what ships; this is the SOURCE it was
written in, kept because nine gated mini-programs come to 60 ops and
1,100 lines of JSON, and every district has to repeat its band on every
op it owns. Editing that by hand is how a district ends up half in one
region and half in another.

Regenerating is expected to produce no diff unless you changed something
here. `cargo test -p voxel-engine` pins what the result has to be true
of: every district whole, none of them a sliver.
"""
import json, collections

# Terciles of each axis, measured rather than assumed. An FBM sum is
# bell-shaped, not uniform: cutting both axes at 0.455/0.545 looked even
# and gave the middle districts a twentieth of the world between them.
AX_A = {"lo": [0.0, 0.427], "mid": [0.427, 0.565], "hi": [0.565, 1.0]}
AX_B = {"lo": [0.0, 0.444], "mid": [0.444, 0.595], "hi": [0.595, 1.0]}
TAKES_MATERIAL = {"slabs_y", "pillars_xz", "walls", "beams", "fbm3", "coarse_solid"}

ops = []
D = collections.OrderedDict()


# Districts whose features are big enough to SURVIVE coarse voxels, and
# so build their own far silhouette instead of collapsing into the
# generic solid mass. Everything else gets `coarse_solid`, because a
# 12 m floor is gone by the time voxels are 4 m and an open district
# that renders as nothing at range is a hole in the world.
SELF_COARSE = {"cathedral", "silos"}


def district(name, mat, a, b, body):
    region = [*AX_A[a], *AX_B[b]]
    D[name] = (mat, a, b)
    if name not in SELF_COARSE:
        # Material 2, the neutral base, NOT the district's own — colour
        # stays with the band chain at the end. `surface_material_weight`
        # reads that chain to answer "which district is this", tracking
        # the base the bands repaint FROM; nine coarse masses each
        # claiming their own material leave it tracking whichever came
        # last, and every gate in the level then matches nothing.
        ops.append({"type": "coarse_solid", "material": 2, "region": region})
    for t, kw in body:
        e = {"type": t}
        e.update(kw)
        if t in TAKES_MATERIAL:
            e.setdefault("material", mat)
        # A district that owns its own coarse form needs its structure at
        # every LOD, not just the fine one.
        if name in SELF_COARSE and t in LOD_ALL_ABLE:
            e.setdefault("lod", "all")
        e["region"] = region
        ops.append(e)


LOD_ALL_ABLE = {"lattice_y", "slabs_y", "pillars_xz", "beams"}


# TWO octaves, not four. A region axis that gates COLOUR can be fractal
# and only looks hand-drawn; one that gates STRUCTURE cannot. At four
# octaves the finest is ~700 m, so every square kilometre is riddled with
# little islands of other districts, and since a floor simply stops where
# its district does, every floor in the world ended along a wiggling
# contour. The whole megastructure read as torn paper rather than
# architecture.
ops.append({"type": "region_axes", "scale": [0.00009, 0.00007],
            "offset": [2100.0, -880.0, -5400.0, 3300.0], "octaves": 2})

# 1. WARRENS — habitation cells. Low ceilings, walls both ways, doors
#    everywhere: the dwelling levels, claustrophobic at human scale.
district("warrens", 3, "lo", "lo", [
    ("lattice_y", {"spacing": 20.0}),
    ("slabs_y", {"half_thickness": 0.9}),
    ("walls", {"axis": "x", "spacing": 34.0, "half_thickness": 0.9, "chance": 0.72,
               "door": {"cell": 11.0, "chance": 0.55, "half": [3.0, 5.5, 2.4], "y": 5.0}}),
    ("walls", {"axis": "z", "spacing": 34.0, "half_thickness": 0.9, "chance": 0.72, "salt": 811,
               "door": {"cell": 11.0, "chance": 0.55, "half": [3.0, 5.5, 2.4], "y": 5.0, "salt": 37}}),
    ("pillars_xz", {"spacing": 26.0, "jitter": 5.0, "girth": [1.0, 1.4]}),
    ("grid_holes", {"cell": 13.0, "chance": 0.10, "half": [4.0, 3.0, 4.0]}),
    ("shafts_xz", {"spacing": 190.0, "jitter": 60.0, "radius": [9.0, 7.0]}),
])

# 2. CONDUIT — one wall axis only, near-certain and thick. Parallel
#    kilometre-long tunnels you travel ALONG and can never cross.
district("conduit", 4, "lo", "mid", [
    ("lattice_y", {"spacing": 26.0}),
    ("slabs_y", {"half_thickness": 1.8}),
    ("walls", {"axis": "z", "spacing": 30.0, "half_thickness": 2.6, "chance": 0.94,
               "door": {"cell": 96.0, "chance": 0.35, "half": [4.0, 7.0, 5.0], "y": 7.0}}),
    ("shafts_xz", {"spacing": 640.0, "jitter": 120.0, "radius": [26.0, 14.0]}),
])

# 3. STACKS — nothing but floors. A hundred storeys of empty plate at
#    twelve metres, holed through, with no walls to stop the eye.
district("stacks", 5, "lo", "hi", [
    ("lattice_y", {"spacing": 12.0}),
    ("slabs_y", {"half_thickness": 0.6}),
    ("grid_holes", {"cell": 22.0, "chance": 0.30, "half": [9.0, 4.0, 9.0]}),
    ("pillars_xz", {"spacing": 48.0, "jitter": 10.0, "girth": [0.8, 1.0]}),
])

# 4. GIRDERS — no slabs at all. An open steel lattice of columns and
#    catwalks over a drop that has no floor anywhere in it.
district("girders", 6, "mid", "lo", [
    ("lattice_y", {"spacing": 30.0}),
    ("pillars_xz", {"spacing": 17.0, "jitter": 4.0, "girth": [1.1, 1.6]}),
    ("shafts_xz", {"spacing": 150.0, "jitter": 40.0, "radius": [30.0, 18.0]}),
    ("beams", {"every": 1, "half_width": 2.6, "y": 1.0, "half_height": 0.8, "reach": 9.0}),
])

# 5. CATHEDRAL — the great halls: two-hundred-metre storeys on columns
#    the size of city blocks. The shot Blame! is remembered for.
district("cathedral", 7, "mid", "mid", [
    ("lattice_y", {"spacing": 195.0}),
    ("slabs_y", {"half_thickness": 6.0}),
    ("pillars_xz", {"spacing": 250.0, "jitter": 40.0, "girth": [16.0, 12.0]}),
    ("shafts_xz", {"spacing": 780.0, "jitter": 200.0, "radius": [55.0, 40.0]}),
    ("beams", {"every": 1, "half_width": 7.0, "y": 60.0, "half_height": 3.0, "reach": 26.0}),
])

# 6. SILOS — wells 150 m across bored through near-solid mass, 600 m
#    apart. Mostly you are inside the rock; the world IS the holes.
district("silos", 8, "mid", "hi", [
    ("lattice_y", {"spacing": 150.0}),
    ("slabs_y", {"half_thickness": 74.0}),
    ("shafts_xz", {"spacing": 600.0, "jitter": 150.0, "radius": [95.0, 55.0]}),
    ("beams", {"every": 1, "half_width": 4.0, "y": 74.0, "half_height": 1.6, "reach": 30.0}),
])

# 7. FOUNDRY — heavy industry: thick walls one way, deep floors, massive
#    beams. Asymmetric on purpose, so it never resolves into a grid.
district("foundry", 9, "hi", "lo", [
    ("lattice_y", {"spacing": 52.0}),
    ("slabs_y", {"half_thickness": 3.2}),
    ("walls", {"axis": "x", "spacing": 68.0, "half_thickness": 3.4, "chance": 0.82,
               "door": {"cell": 34.0, "chance": 0.6, "half": [5.0, 13.0, 8.0], "y": 13.0}}),
    ("walls", {"axis": "z", "spacing": 210.0, "half_thickness": 2.0, "chance": 0.4, "salt": 313}),
    ("shafts_xz", {"spacing": 300.0, "jitter": 90.0, "radius": [22.0, 16.0]}),
    ("beams", {"every": 2, "half_width": 5.0, "y": 3.0, "half_height": 2.0, "reach": 14.0}),
])

# 8. RUIN — the same building, eaten. A 3D noise carve takes bites out of
#    everything at a 250 m scale, so floors end in mid-air.
district("ruin", 10, "hi", "mid", [
    ("lattice_y", {"spacing": 42.0}),
    ("slabs_y", {"half_thickness": 1.6}),
    ("pillars_xz", {"spacing": 44.0, "jitter": 14.0, "girth": [2.0, 3.0]}),
    ("walls", {"axis": "z", "spacing": 88.0, "half_thickness": 1.4, "chance": 0.5, "salt": 55}),
    ("fbm3", {"scale": 0.004, "y_ratio": 1.6, "octaves": 3, "threshold": 0.06,
              "width": 26.0, "carve": True}),
])

# 9. BEDROCK — solid to the horizon, bored by capillaries. The negative
#    space between inhabited districts: a slab whose half thickness is
#    half its spacing has no gap between storeys at all.
district("bedrock", 11, "hi", "hi", [
    ("lattice_y", {"spacing": 64.0}),
    ("slabs_y", {"half_thickness": 32.0}),
    ("shafts_xz", {"spacing": 210.0, "jitter": 70.0, "radius": [11.0, 9.0]}),
])

# One cut for every district's shafts. The shaft registers are per-sample
# and only one district's `shafts_xz` can pass its gate at a point, so
# nine kinds of bore share a single carve.
ops.append({"type": "shafts_cut"})

# Colour follows the same bands, so the coarse mass a district shows at
# ten kilometres is already the colour its architecture will be up close.
for name, (mat, a, b) in D.items():
    ops.append({"type": "material_band", "from": 2, "material": mat,
                "a": AX_A[a], "b": AX_B[b]})

# --- materials ----------------------------------------------------------
# All concrete, as the source material is: the districts are told apart by
# their SHAPE, and the palette only has to keep them from reading as one
# grey mass. Values are LINEAR, so they look about twice as bright as the
# number suggests.
#
# `lights` is (spacing, level_spacing, chance) — an inhabited district is
# lit and an abandoned one is not, which at night is most of what tells
# you where you are.
def surface(mat, base, lights=None, warm=(1.30, 1.25, 1.05)):
    m = {"type": "surface", "id": mat, "base": list(base),
         "grime": {"tint": [0.55, 0.58, 0.55], "amount": 1.0}, "detail_fade": 0.004}
    if lights:
        sp, lsp, ch = lights
        m["emissive"] = {"color": list(warm), "intensity": 1.0, "spacing": sp,
                         "level_spacing": lsp, "chance": ch, "glow": 0.077}
    return m

COOL = (0.85, 0.98, 1.30)
SODIUM = (1.40, 1.05, 0.55)

materials = [
    # The base mass, and what a chunk is before any band claims it.
    surface(2, (0.0784, 0.0784, 0.0803), lights=(13.0, 22.0, 0.45)),
    # Inhabited, lit, grubby with sodium light.
    surface(3, (0.085, 0.070, 0.052), lights=(9.0, 20.0, 0.80), warm=SODIUM),
    # Wet dark tunnels, a light every hundred metres or so.
    surface(4, (0.030, 0.038, 0.034), lights=(26.0, 26.0, 0.30)),
    # Bone-pale open plate, cold strip lighting.
    surface(5, (0.135, 0.132, 0.120), lights=(11.0, 12.0, 0.55), warm=COOL),
    # Oxidised steel. Unlit — nothing lives in a lattice.
    surface(6, (0.062, 0.036, 0.022)),
    # The great halls: cold grey-blue, a few enormous lamps.
    surface(7, (0.100, 0.106, 0.120), lights=(60.0, 195.0, 0.9), warm=COOL),
    # Near-black basalt around the wells.
    surface(8, (0.020, 0.020, 0.023)),
    # Industry: iron with a warm cast, hot orange working lights.
    surface(9, (0.055, 0.048, 0.042), lights=(17.0, 52.0, 0.65), warm=SODIUM),
    # Bleached and mottled. Dead.
    surface(10, (0.072, 0.078, 0.060)),
    # Dull mass.
    surface(11, (0.044, 0.041, 0.037)),
]

level = {
    "lod": {"max_level": 8, "top_radius": 2, "top_y": [-3, 3],
            "split_k": 1.6, "merge_k": 2.1},
    "materials": materials,
    "generator": ops,
    "planning": {
        "stack": [
            {"kind": "biomes", "name": "districts",
             "table": [[name, mat] for name, (mat, _, _) in D.items()]},
            # Pockets are somebody's shelter, so they belong where somebody
            # could live: the warrens and the foundry, not the silos.
            {"kind": "scatter3", "name": "sites:pockets", "chance": 0.45,
             "snap_y_m": 44.0, "biome": "districts:warrens"},
            {"kind": "emit", "name": "pockets", "source": "sites:pockets",
             "cell_y_m": 132, "pad_m": 0.0,
             "emit": {"type": "site_structure3", "marker": "pocket", "structure": "pocket"},
             "max_chunk_edge_m": 128.0},
            {"kind": "connect3", "name": "links", "source": "sites:pockets"},
            {"kind": "emit", "name": "tubes", "source": "links",
             "cell_y_m": 132, "pad_m": 464.0, "emit": {"type": "tubes"},
             "max_chunk_edge_m": 128.0},
        ],
        "structures": json.load(open("levels/megastructure.json"))["planning"]["structures"],
    },
}

with open("levels/megastructure.json", "w") as f:
    json.dump(level, f, indent=2)
    f.write("\n")
print(f"{len(ops)} ops, {len(materials)} materials, {len(D)} districts")
