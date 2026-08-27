"""Generate levels/brutalist.json — one large brutalist building.

    python3 levels/generators/brutalist.py

The JSON is the artefact and is what ships; this is the SOURCE it was
written in, kept because half the numbers in it are DERIVED from the
other half and getting one of those wrong is not visible in the file.
The envelope is the union of the masses, the overhang skirt is the gap
between the low block's face and the envelope, and every face is a whole
number of bays from the origin — see BAY below for what happens when one
is not.

Editing the level in the in-game editor (F10) and saving supersedes
this: the file it writes is the level. Re-run this only to change one of
the derived numbers, and expect to lose hand edits when you do.
"""
import json

# --- the building, in metres -------------------------------------------
#
# The whole form is: two masses, hollowed, filled with an endless storey
# fabric, clipped ONCE to the box that bounds them, then carved.
#
# The elevation is not drawn by anything. A ribbon is cut out of the bare
# shell before the structure goes in, and the piers and floor bands then
# fill that ribbon back up wherever they cross it — so what a window is,
# is the part of the cut nothing stood in. PROUD widens the clip past the
# masses and pushes that whole frame out in front of the glazing; at 0 it
# is flush, and flush is what makes 180 m of it read as a wall with holes
# in it rather than as shelving.

STOREY = 5.6            # floor to floor
BAY = 7.5               # pier rhythm on the elevation
CROSS = 22.5            # cross-wall rhythm (three bays)
# The spine walls run the other way, and their spacing is deliberately
# NOT a divisor of either block's z half-width. A wall grid is
# world-space too: one whose plane lands exactly on an elevation skins
# the whole thing, and every window on that face becomes a doorway-shaped
# accident in an otherwise blank wall. 45 / 22.5 = 2 was exactly that.
SPINE = 26.0
WALL = 4.5              # external wall thickness
PROUD = 0.0             # how far the frame stands out from the glazing

# Every face is a whole number of BAYs from the origin, and that is not
# tidiness: the pier grid is world-space, so a face half a bay off its
# rhythm gets no pier in its window slot and the elevation comes out as
# open shelving instead of a punched wall. It is the single most
# load-bearing number in the file.
LOW = dict(c=[0.0, 51.0, 0.0], h=[90.0, 41.0, 30.0])       # y 10..92
HIGH = dict(c=[0.0, 117.0, 0.0], h=[90.0, 25.0, 45.0])     # y 92..142
# Their union, grown sideways by PROUD and by nothing vertically: a
# building has an elevation, not a proud roof.
ENV = dict(c=[0.0, 0.5 * (LOW["c"][1] - LOW["h"][1] + HIGH["c"][1] + HIGH["h"][1]), 0.0],
           h=[max(LOW["h"][0], HIGH["h"][0]) + PROUD,
              0.5 * ((HIGH["c"][1] + HIGH["h"][1]) - (LOW["c"][1] - LOW["h"][1])),
              max(LOW["h"][2], HIGH["h"][2]) + PROUD])


def box(name, src, mode, c, h, mat=None, lod=None):
    n = {"kind": "box", "name": name, "in": {"sdf": src},
         "center": [float(v) for v in c], "half": [float(v) for v in h]}
    if mode != "add":
        n["mode"] = mode
    if mat is not None:
        n["material"] = mat
    if lod is not None:
        n["lod"] = lod
    return n


nodes = []
prev = "void"


def add(node):
    global prev
    nodes.append(node)
    if node.get("name"):
        prev = node["name"]


nodes.append({"kind": "sdf_void", "name": "void"})
nodes.append({"kind": "height_zero", "name": "sea"})
nodes.append({"kind": "warp_none", "name": "unwarped"})

# The plaza it stands on: gentle, and deliberately so. A block this sheer
# wants a flat datum to be sheer against, and relief that reached the
# plinth would read as the building sinking into a hill.
nodes.append({"kind": "height_fbm", "name": "ground.1",
              "in": {"height": "sea", "warp": "unwarped"},
              "scale": 0.0006, "amp": 3.4, "octaves": 3})
nodes.append({"kind": "height_fbm", "name": "ground.2",
              "in": {"height": "ground.1", "warp": "unwarped"},
              "offset": [610.0, -220.0],
              "scale": 0.0052, "amp": 1.1, "octaves": 3})
nodes.append({"kind": "height_offset", "name": "ground.3",
              "in": {"height": "ground.2"}, "value": -3.5})

# --- mass ---------------------------------------------------------------
add(box("mass.low", prev, "add", LOW["c"], LOW["h"], mat=2))
add(box("mass.high", prev, "add", HIGH["c"], HIGH["h"], mat=3))
# Hollowed into shells, and the two voids MEET: solid left between them
# is solid the ribbon window then slots straight through, which reads as
# a slit of daylight halfway up the building.
add(box("void.low", prev, "cut", [0.0, 52.0, 0.0],
        [LOW["h"][0] - WALL, 38.5, LOW["h"][2] - WALL]))
add(box("void.high", prev, "cut", [0.0, 114.0, 0.0],
        [HIGH["h"][0] - WALL, 23.5, HIGH["h"][2] - WALL]))

# --- fabric -------------------------------------------------------------
nodes.append({"kind": "lattice_y", "name": "storeys", "spacing": STOREY})
# Glazing FIRST, structure second. The cut runs while the shell is the
# only thing standing, so what it opens is the wall alone; everything
# after it puts itself back into the opening, which is where the
# elevation's mullions and spandrel returns come from.
add({"kind": "slabs_y", "name": "windows", "in": {"sdf": prev, "lattice": "storeys"},
     "half_thickness": 1.1, "y": 1.9, "cut": True})
add({"kind": "slabs_y", "name": "plates", "in": {"sdf": prev, "lattice": "storeys"},
     "half_thickness": 0.7, "material": 6})
# Two wall grids, and the second is not decoration: with cross walls
# alone every window lines up with the one on the far elevation and the
# building is see-through.
add({"kind": "walls", "name": "cross", "in": {"sdf": prev, "lattice": "storeys"},
     "axis": "x", "spacing": CROSS, "half_thickness": 0.8, "chance": 1.0,
     "door": {"cell": 9.0, "chance": 0.75, "half": [1.0, 1.35, 2.6], "y": -1.55},
     "material": 6})
add({"kind": "walls", "name": "spine", "in": {"sdf": prev, "lattice": "storeys"},
     "axis": "z", "spacing": SPINE, "half_thickness": 0.7, "chance": 1.0,
     "salt": 617,
     "door": {"cell": 9.0, "chance": 0.7, "half": [0.9, 1.35, 2.6],
              "y": -1.55, "salt": 41},
     "material": 6})
add({"kind": "pillars_xz", "name": "columns", "in": {"sdf": prev},
     "spacing": BAY, "jitter": 0.0, "girth": [1.85, 0.0], "material": 7})

# One clip, and the building exists. Everything above it is unbounded.
add(box("envelope", prev, "clip", ENV["c"], ENV["h"]))

# --- carve --------------------------------------------------------------
# The overhang. The clip left fabric standing in the upper block's
# footprint all the way to the ground; these two take it away. That
# region is outside every mass, so they breach no shell doing it — which
# is the whole reason the massing is carved after the clip and not
# before.
skirt = 0.5 * ((LOW["h"][2] + PROUD) + (ENV["h"][2] + 2.0))
add(box("skirt.z+", prev, "cut", [0.0, 50.0, skirt],
        [95.0, 42.0, skirt - LOW["h"][2] - PROUD]))
add(box("skirt.z-", prev, "cut", [0.0, 50.0, -skirt],
        [95.0, 42.0, skirt - LOW["h"][2] - PROUD]))
# A base and a crown: two solid bands filled back in over the fabric, so
# 130 m of elevation begins and ends somewhere instead of running off
# both edges as one texture. Both go in BEFORE the entrance is cut, which
# is what lets the entrance be a hole in a solid wall rather than a
# missing piece of a grid.
add(box("base", prev, "add", [0.0, 19.0, 0.0],
        [LOW["h"][0] + PROUD, 9.0, LOW["h"][2] + PROUD], mat=2))
add(box("crown", prev, "add", [0.0, 135.0, 0.0],
        [HIGH["h"][0] + PROUD, 7.0, HIGH["h"][2] + PROUD], mat=3))
# The openings, and every one of them stays INSIDE a solid band. Cutting
# into the fabric instead leaves floor plates and columns sectioned in
# mid-air across the back of the recess, which reads as damage rather
# than as a door.
add(box("entrance", prev, "cut", [0.0, 19.5, 36.0], [24.0, 6.0, 14.0]))
add(box("service", prev, "cut", [0.0, 18.5, -36.0], [13.0, 5.0, 12.0]))
add(box("roof.court", prev, "cut", [0.0, 136.0, 0.0], [36.0, 6.5, 26.0]))

# --- appendages ---------------------------------------------------------
# Added after the window cut, so they stay blank: a stair tower has no
# storeys to show. Added after the clip, so the clip does not reach it.
# Two blank towers bracketing the slab, and one on its back. They are
# what a 180 m elevation of horizontal banding needs to stand still:
# something vertical, unrelieved, and taller than the roof.
add(box("tower.x-", prev, "add", [-101.0, 72.0, 0.0], [11.0, 80.0, 21.0], mat=4))
add(box("tower.x+", prev, "add", [101.0, 72.0, 0.0], [11.0, 80.0, 21.0], mat=4))
add(box("tower.stair", prev, "add", [0.0, 70.0, -49.0], [14.0, 78.0, 7.0], mat=4))
add(box("plinth", prev, "add", [0.0, 5.0, 0.0], [126.0, 5.0, 76.0], mat=5))
add(box("apron", prev, "add", [0.0, 1.0, 0.0], [158.0, 3.0, 104.0], mat=5))
# One stair mass off the front of the plinth. Anything thinner than this
# — a parapet, a kerb — reads as tape laid on the plaza at any distance
# the whole building is in frame at.
add(box("stair", prev, "add", [0.0, 5.0, 84.0], [54.0, 5.0, 10.0], mat=5))

# --- ground -------------------------------------------------------------
# Last, because a clip earlier in the chain would have taken it too.
add({"kind": "height_surface", "name": "plaza",
     "in": {"height": "ground.3", "sdf": prev}, "material": 1})

nodes.append({
    "kind": "population", "name": "blocks",
    "per_tile": 4, "chance": 0.3,
    "altitude": [-40.0, 30.0], "min_up": 0.9,
    "placement": {"align": "normal", "tilt_deg": 7.0},
    "sink_scaled": 0.3, "scale_bias": 1.9,
    "variants": [{"scale": [0.5, 2.6]}],
})


def surface(id, base, **kw):
    m = {"type": "surface", "id": id, "base": [float(v) for v in base]}
    m.update(kw)
    m.setdefault("detail_fade", 0.003)
    return m


level = {
    "environment": {
        # Low and along the long elevation. The only relief on this
        # facade is the depth of its window reveals, and a sun overhead
        # puts no shadow in one.
        "sun_direction": [0.86, 0.40, 0.31],
    },
    "lod": {
        "max_level": 10,
        "top_radius": 2,
        "top_y": [-1, 0],
        # Far higher than any other shipped level, and affordable for a
        # reason: the plaza is a heightfield and the building is bounded
        # by a `box` clip, so `eval_range` decides almost every chunk out
        # here as air or as rock and it never reaches a density pass.
        # What the refinement is spent on is the 2.2 m window reveals,
        # which need an ARRIS: they read as openings or as dents
        # depending on whether their edge is one voxel or four.
        #
        # A `detail` volume was the obvious tool and is the wrong one
        # here: the landmark IS the level, so a box around it is 340 m
        # across, and two levels over that box wanted 11.4k of the
        # slab's 11.4k slots on its own. Camera distance buys the same
        # sharpness where you are actually looking, for 8.6k.
        "split_k": 4.5,
        "merge_k": 5.3,
        "gated_finest_first": False,
    },
    "nodes": nodes,
    "materials": [
        # Seven greys, and they are all the level has: the shader's
        # surface recipe is a flat colour, so what tells one plane from
        # another here is which way it faces.
        surface(1, [0.086, 0.084, 0.078], grain=0.4,
                grime={"tint": [0.5, 0.49, 0.45], "amount": 0.5}),
        surface(2, [0.132, 0.130, 0.123], grain=0.5,
                band={"freq": 0.75, "amp": 0.22, "lo": 0.05, "hi": 0.45, "warp": 0.3},
                grime={"tint": [0.52, 0.53, 0.5], "amount": 0.55}),
        surface(3, [0.112, 0.111, 0.109], grain=0.5,
                band={"freq": 0.75, "amp": 0.22, "lo": 0.05, "hi": 0.45, "warp": 0.3},
                grime={"tint": [0.52, 0.53, 0.5], "amount": 0.65}),
        surface(4, [0.074, 0.074, 0.075], grain=0.55, streaks=0.5,
                grime={"tint": [0.45, 0.46, 0.44], "amount": 0.8}),
        surface(5, [0.150, 0.147, 0.137], grain=0.45,
                grime={"tint": [0.5, 0.5, 0.46], "amount": 0.6}),
        surface(6, [0.052, 0.050, 0.048], grain=0.3,
                emissive={"color": [1.25, 1.12, 0.86], "intensity": 2.2,
                          "spacing": 7.5, "level_spacing": 5.6,
                          "chance": 0.6, "glow": 0.3}),
        surface(7, [0.121, 0.119, 0.114], grain=0.5),
    ],
}

with open("levels/brutalist.json", "w") as f:
    json.dump(level, f, indent=1)
    f.write("\n")
print(f"{len(nodes)} nodes, skirt at {skirt}")
