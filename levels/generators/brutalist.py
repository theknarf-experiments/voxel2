"""Generate levels/brutalist.json — a brutalist complex on a deck.

    python3 levels/generators/brutalist.py

The JSON is the artefact and is what ships; this is the SOURCE it was
written in, kept because most of the numbers in it are DERIVED from the
other numbers and getting one wrong is not visible in the file. Editing
the level in the in-game editor (F10) and saving supersedes this.

The composition is four moves, and they are the ones the Quake Brutalist
Jam maps are made of: a raised deck with a court sunk into it, one long
slab and one tower standing on the deck, a bridge flying between them
over the court, and a handful of large flat planes in a colour that is
not grey.

Each mass is a DISTRICT — a `region` scope over the linear axes
`region_xz` lays down — so it carries its own storey height, bay rhythm
and wall pattern, and its fabric is confined to its own rectangle without
anything having to clip it away afterwards. The clip at the end of each
district is gated too, so it bounds that mass in y and cannot reach the
others.
"""
import json

# --- the grid ------------------------------------------------------------
#
# 8 m bays and a 1 m module, which is what the jam's maps measure out at
# (their peaks are at 32, 64, 128 and 256 Quake units — 1, 2, 4 and 8 m).
# EVERY mass face is a whole number of bays from the origin: the pier grid
# is world-space, so a face half a bay off its rhythm gets no pier in its
# window slot and the elevation comes out as open shelving.
BAY = 8.0
# The cross walls deliberately do NOT divide any face's offset, for two
# reasons: a wall plane that lands on an elevation skins the whole thing,
# and one that lands between the piers puts a second, finer rhythm on the
# facade — which is the difference between architecture and graph paper.
CROSS = 25.0
SPINE = 26.0

# The axes every gate is written in. A byte per edge, so an edge lands
# within SPAN/255 = 1.6 m of where it is asked for — and it is set OUTSIDE
# its mass by more than that, deliberately. Inside, and the window cut
# stops partway through the outer wall and the elevation comes out blank;
# outside, and the fabric that overshoots is taken off by the district's
# own clip, which is the last op in every one of them.
ORIGIN = (-200.0, -200.0)
SPAN = (400.0, 400.0)
OUTSET = 2.5


def rect(x0, x1, z0, z1):
    """A gate covering this world rectangle, in the linear axes."""
    return [
        (x0 - ORIGIN[0]) / SPAN[0], (x1 - ORIGIN[0]) / SPAN[0],
        (z0 - ORIGIN[1]) / SPAN[1], (z1 - ORIGIN[1]) / SPAN[1],
    ]


def span(a, b):
    """(centre, half) from two edges."""
    return 0.5 * (a + b), 0.5 * (b - a)


class Mass:
    """A box in world coordinates, and the gate that goes with it."""

    def __init__(self, x, z, y):
        self.x, self.z, self.y = x, z, y

    def box(self, wall=0.0):
        cx, hx = span(self.x[0] + wall, self.x[1] - wall)
        cy, hy = span(self.y[0] + wall, self.y[1] - wall)
        cz, hz = span(self.z[0] + wall, self.z[1] - wall)
        return [cx, cy, cz], [hx, hy, hz]

    def to_deck(self):
        """The same box, opened downward to the deck. What a district's
        clip has to be, so that bounding a mass does not excavate under
        it."""
        return Mass(self.x, self.z, (CLIP_FLOOR, self.y[1]))

    def gate(self, z=None):
        z = z or self.z
        return rect(self.x[0] - OUTSET, self.x[1] + OUTSET,
                    z[0] - OUTSET, z[1] + OUTSET)


# --- the composition -----------------------------------------------------

WALL = 4.0                                   # external wall thickness
DECK = Mass((-184, 168), (-112, 112), (0, 12))
APRON = Mass((-208, 192), (-136, 136), (-4, 6))
COURT = Mass((-40, 72), (-56, 56), (-22, 12))

SLAB = Mass((-168, -56), (-48, 48), (12, 96))
SLAB_CAP = Mass((-168, -56), (-64, 64), (96, 132))   # cantilevered, both ways
SLAB_ALL = Mass(SLAB.x, SLAB_CAP.z, (SLAB.y[0], SLAB_CAP.y[1]))

TOWER = Mass((80, 144), (-48, 16), (12, 204))
BRIDGE = Mass((-56, 80), (-24, 8), (70, 98))
MONOLITH = Mass((-12, 20), (-8, 24), (-20, 64))

SLAB_STOREY = 5.6
TOWER_STOREY = 4.4

# Materials, by the share the jam's own maps use them in: half of every
# surface grey, a fifth of it near-white, a twentieth near-black, and
# about one in ten a single flat colour.
GROUND, PALE, MID, DECKGREY, DARK, INSIDE, PIER = 1, 2, 3, 4, 5, 6, 7
RED, OCHRE, TEAL = 8, 9, 10

nodes = []
ungated = "void"          # last ungated write to the sdf register
gated = []                # last write of each district closed since then
scope = None              # the node list currently being appended to
inner = None              # last write inside the open district


def emit(node):
    (scope if scope is not None else nodes).append(node)
    return node


def sources():
    """What an sdf consumer here has to name.

    A gated write only displaces the writes its own region covers, so
    after two disjoint districts there are three things standing and a
    reader outside them sees all three. The compiler checks this; getting
    it wrong is the level's most common error and it says so by name.
    """
    if scope is not None:
        # Inside a district the chain is its own; the first op in it
        # reaches back to the last ungated write, and no further,
        # because the other districts are disjoint from this one and so
        # their writes are not visible here.
        return inner or ungated
    return ungated if not gated else [ungated] + gated


def chain(name, node):
    """Emit an sdf-consuming node and make it the new head."""
    global ungated, gated, inner
    node["in"] = dict(node.get("in", {}), sdf=sources())
    node["name"] = name
    emit(node)
    if scope is None:
        ungated, gated = name, []
    else:
        inner = name
    return name


def box(name, mode, c, h, mat=None, lod=None):
    n = {"kind": "box", "center": [float(v) for v in c], "half": [float(v) for v in h]}
    if mode != "add":
        n["mode"] = mode
    if mat is not None:
        n["material"] = mat
    if lod is not None:
        n["lod"] = lod
    return chain(name, n)


def mass(name, m, mat, wall=0.0, mode="add"):
    c, h = m.box(wall)
    return box(name, mode, c, h, mat=mat)


# A district's clip reaches DOWN TO THE DECK, not to the underside of the
# mass it bounds. A gated clip intersects everything standing inside its
# rectangle, not merely the fabric declared above it in the same scope —
# so a clip whose box starts at the mass's own base deletes the deck out
# from under it, and the masses stood in trenches with the apron six
# metres down for a floor. Nothing below the deck is visible anyway: it is
# either buried in the deck or refilled by the ground merge at the end.
CLIP_FLOOR = 0.0


def district(name, gate):
    """Open a `region` scope. Its ops are gated to `gate`, which means its
    final clip is gated too — so a district bounds ITSELF and cannot
    reach the mass next door."""
    global scope, inner
    node = {"kind": "region", "name": name, "axes": gate, "nodes": []}
    nodes.append(node)
    scope, inner = node["nodes"], None
    return node


def close():
    global scope, gated, inner
    scope = None
    gated.append(inner)
    inner = None


# --- program -------------------------------------------------------------

nodes.append({"kind": "sdf_void", "name": "void"})
nodes.append({"kind": "height_zero", "name": "sea"})
nodes.append({"kind": "warp_none", "name": "unwarped"})
# A gate means "this block, here" rather than "wherever the noise is".
nodes.append({"kind": "region_xz", "name": "axes",
              "origin": list(ORIGIN), "span": list(SPAN)})

# The plaza the complex stands on: gentle, and deliberately so. Relief
# that reached the deck would read as the buildings sinking into a hill.
nodes.append({"kind": "height_fbm", "name": "ground.1",
              "in": {"height": "sea", "warp": "unwarped"},
              "scale": 0.00055, "amp": 3.6, "octaves": 3})
nodes.append({"kind": "height_fbm", "name": "ground.2",
              "in": {"height": "ground.1", "warp": "unwarped"},
              "offset": [610.0, -220.0],
              "scale": 0.0048, "amp": 1.2, "octaves": 3})
nodes.append({"kind": "height_offset", "name": "ground.3",
              "in": {"height": "ground.2"}, "value": -4.5})

# --- masses, ungated ------------------------------------------------------
mass("deck", DECK, DECKGREY)
mass("mass.slab", SLAB, PALE)
mass("mass.slab.cap", SLAB_CAP, PALE)
mass("mass.tower", TOWER, MID)
# Hollowed. The void stops short of the cantilever's own z, so the join
# between the two halves is solid — a storey of transfer structure, which
# is what actually holds a 16 m overhang up.
mass("void.slab", SLAB, None, wall=WALL, mode="cut")
mass("void.slab.cap", SLAB_CAP, None, wall=WALL, mode="cut")
mass("void.tower", TOWER, None, wall=WALL, mode="cut")

# --- the slab -------------------------------------------------------------
# Long, pale, finely divided: 14 storeys of 3.4 m piers at 8 m centres,
# with the cross walls showing through the glazing on a rhythm of their
# own. The gate covers the CANTILEVER's footprint, so the fabric reaches
# the full width up there and a cut takes it away lower down.
district("slab", SLAB_CAP.gate())
emit({"kind": "lattice_y", "name": "slab.storeys", "spacing": SLAB_STOREY})
chain("slab.windows", {"kind": "slabs_y", "in": {"lattice": "slab.storeys"},
                       "half_thickness": 1.1, "y": 1.9, "cut": True})
chain("slab.plates", {"kind": "slabs_y", "in": {"lattice": "slab.storeys"},
                      "half_thickness": 0.7, "material": INSIDE})
chain("slab.cross", {"kind": "walls", "in": {"lattice": "slab.storeys"},
                     "axis": "x", "spacing": CROSS, "half_thickness": 0.7,
                     "chance": 1.0, "material": INSIDE,
                     "door": {"cell": 9.0, "chance": 0.75,
                              "half": [1.0, 1.35, 2.6], "y": -1.55}})
chain("slab.spine", {"kind": "walls", "in": {"lattice": "slab.storeys"},
                     "axis": "z", "spacing": SPINE, "half_thickness": 0.6,
                     "chance": 1.0, "salt": 617, "material": INSIDE,
                     "door": {"cell": 9.0, "chance": 0.7,
                              "half": [0.9, 1.35, 2.6], "y": -1.55, "salt": 41}})
chain("slab.piers", {"kind": "pillars_xz", "spacing": BAY, "jitter": 0.0,
                     "girth": [1.7, 0.0], "material": PIER})
mass("slab.clip", SLAB_ALL.to_deck(), None, mode="clip")
# The overhang: below the cap the block is 32 m narrower, and the fabric
# the clip left standing out there comes away in one cut a side.
for side in (1, -1):
    c, h = span(SLAB.z[1], SLAB_CAP.z[1] + 4.0)
    box(f"slab.skirt{'+' if side > 0 else '-'}", "cut",
        [0.0, 0.5 * (SLAB.y[0] + SLAB.y[1]), side * c],
        [200.0, 0.5 * (SLAB.y[1] - SLAB.y[0]) + 1.0, h])
# A solid base and a solid crown, so 120 m of elevation begins and ends
# somewhere instead of running off both edges as one texture.
box("slab.base", "add", *Mass(SLAB.x, SLAB.z, (SLAB.y[0], 30.0)).box(), mat=PALE)
box("slab.crown", "add", *Mass(SLAB_CAP.x, SLAB_CAP.z, (124.0, SLAB_CAP.y[1])).box(),
    mat=PALE)
close()

# --- the tower ------------------------------------------------------------
# The vertical, and everything about it is heavier: a tighter storey, a
# 4.4 m pier on the same 8 m bay, and a window that is a slot.
district("tower", TOWER.gate())
emit({"kind": "lattice_y", "name": "tower.storeys", "spacing": TOWER_STOREY})
chain("tower.windows", {"kind": "slabs_y", "in": {"lattice": "tower.storeys"},
                        "half_thickness": 0.85, "y": 1.6, "cut": True})
chain("tower.plates", {"kind": "slabs_y", "in": {"lattice": "tower.storeys"},
                       "half_thickness": 0.55, "material": INSIDE})
chain("tower.cross", {"kind": "walls", "in": {"lattice": "tower.storeys"},
                      "axis": "x", "spacing": CROSS, "half_thickness": 0.7,
                      "chance": 1.0, "material": INSIDE,
                      "door": {"cell": 8.0, "chance": 0.7,
                               "half": [1.0, 1.1, 2.2], "y": -1.3}})
chain("tower.piers", {"kind": "pillars_xz", "spacing": BAY, "jitter": 0.0,
                      "girth": [2.2, 0.0], "material": PIER})
mass("tower.clip", TOWER.to_deck(), None, mode="clip")
box("tower.base", "add", *Mass(TOWER.x, TOWER.z, (TOWER.y[0], 34.0)).box(), mat=MID)
box("tower.crown", "add", *Mass(TOWER.x, TOWER.z, (188.0, TOWER.y[1])).box(), mat=MID)
close()

# --- the colonnade --------------------------------------------------------
# A row of columns along the court's south edge, and the same two nodes a
# district always is: a field, and the box that bounds it.
COLONNADE = Mass((-40, 72), (56, 80), (12, 34))
district("colonnade", COLONNADE.gate())
chain("colonnade.piers", {"kind": "pillars_xz", "spacing": 2.5 * BAY, "jitter": 0.0,
                          "girth": [4.0, 0.0], "material": PIER})
mass("colonnade.clip", COLONNADE.to_deck(), None, mode="clip")
close()

# --- the deck's own things -------------------------------------------------
mass("apron", APRON, DECKGREY)
# One wall and two stones, and no more: every one of them lays a hard
# band across the deck at this sun angle, and three did.
box("wall.long", "add", *Mass((-160, -40), (88, 96), (0, 17)).box(), mat=DECKGREY)
box("stone.a", "add", *Mass((120, 152), (76, 108), (0, 27)).box(), mat=DARK)
box("stone.b", "add", *Mass((84, 100), (92, 108), (0, 15)).box(), mat=DECKGREY)

# --- ground ---------------------------------------------------------------
# Before the court, not after. The plaza is a merge like any other, and a
# court cut above it fills straight back in with earth to the height the
# terrain happens to be at — which is a 4 m dip where a 34 m room was
# drawn.
chain("plaza", {"kind": "height_surface", "in": {"height": "ground.3"},
                "material": GROUND})

# --- carving --------------------------------------------------------------
# The court, sunk 34 m through the deck and the ground under it. Cut
# rather than built: what makes a sunken space read is that the plane
# around it is continuous and has a hole in it.
mass("court", COURT, None, mode="cut")
box("court.floor", "add",
    *Mass(COURT.x, COURT.z, (COURT.y[0] - 3.0, COURT.y[0])).box(), mat=DECKGREY)
# Steps down the court's east end: three landings, each a whole storey.
for i, y in enumerate((4.0, -6.0, -16.0)):
    box(f"court.step{i}", "add",
        *Mass((COURT.x[1] - 16.0 - 12.0 * i, COURT.x[1]), COURT.z,
              (y - 10.0, y)).box(), mat=DECKGREY)
# The bridge: solid, with a band recessed into each flank. Hollowing it
# and cutting through was two 2.5 m shells, and a 2.5 m shell at the LOD
# a 130 m span is seen at is not a wall — it is a rod.
mass("bridge", BRIDGE, MID)
for side, z, d in (("+", BRIDGE.z[1], -1), ("-", BRIDGE.z[0], +1)):
    box(f"bridge.slot{side}", "cut",
        *Mass(BRIDGE.x, sorted((z + 2.5 * -d, z + 3.0 * d)), (80.0, 86.0)).box())
# A shaft standing in the court and passing the bridge — the one thing in
# the composition with no windows and no scale on it at all.
mass("monolith", MONOLITH, DARK)

# --- the flat colour ------------------------------------------------------
# Four planes, no more: the jam's maps are about one surface in ten a
# colour, and the colour only works because everything around it is not.
#
# The slab's east end, whole and unbroken, facing the court and the sun.
box("plane.red", "add", *Mass((-56.0, -52.8), (-48, 48), (12, 96)).box(), mat=RED)
# The tower's east face, from the base band to the crown.
box("plane.ochre", "add", *Mass((144.0, 147.2), (-48, 16), (34, 188)).box(), mat=OCHRE)
# The court's north wall, which sees no sun at all and carries on the
# strength of its own colour.
box("plane.teal", "add",
    *Mass(COURT.x, (COURT.z[0] - 3.2, COURT.z[0]), COURT.y).box(), mat=TEAL)
box("plane.dark", "add", *Mass((-168, -56), (-64, -61.4), (96, 124)).box(), mat=DARK)

nodes.append({
    "kind": "population", "name": "blocks",
    "per_tile": 4, "chance": 0.28,
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


GRIME = {"tint": [0.52, 0.53, 0.5], "amount": 0.6}

level = {
    "environment": {
        # Low, and from BEHIND the long elevation rather than onto it.
        # A punched wall lit head-on is a flat pattern; the same wall in
        # its own shade, with the sun raking the ends and throwing the
        # tower across the deck toward you, is a building. The elevation
        # gives up its brightness and gets back the shadows.
        "sun_direction": [0.86, 0.34, -0.38],
    },
    "lod": {
        "max_level": 10,
        "top_radius": 2,
        "top_y": [-1, 0],
        # Far higher than any other shipped level, and affordable for a
        # reason: the plaza is a heightfield and every mass is bounded by
        # a `box` clip, so `eval_range` decides most of the world as air
        # or as rock without a density pass. What it buys is the ARRIS on
        # a 2.2 m window reveal, which reads as an opening or as a dent
        # depending on whether its edge is one voxel or four.
        #
        # A `detail` volume is the obvious tool and the wrong one here:
        # the landmark IS the level, so a box round it is 400 m across
        # and two levels over that wanted the whole slab budget.
        "split_k": 4.5,
        "merge_k": 5.3,
        "gated_finest_first": False,
    },
    "nodes": nodes,
    "materials": [
        surface(GROUND, [0.060, 0.058, 0.053], grain=0.4,
                grime={"tint": [0.5, 0.49, 0.45], "amount": 0.5}),
        surface(PALE, [0.216, 0.214, 0.205], grain=0.45,
                band={"freq": 0.75, "amp": 0.22, "lo": 0.05, "hi": 0.45, "warp": 0.3},
                grime=GRIME),
        surface(MID, [0.132, 0.131, 0.128], grain=0.5, grime=GRIME),
        surface(DECKGREY, [0.078, 0.077, 0.073], grain=0.45, grime=GRIME),
        surface(DARK, [0.038, 0.038, 0.040], grain=0.55, streaks=0.5,
                grime={"tint": [0.45, 0.46, 0.44], "amount": 0.8}),
        surface(INSIDE, [0.048, 0.046, 0.044], grain=0.3,
                emissive={"color": [1.25, 1.12, 0.86], "intensity": 2.2,
                          "spacing": 8.0, "level_spacing": SLAB_STOREY,
                          "chance": 0.6, "glow": 0.3}),
        surface(PIER, [0.180, 0.177, 0.169], grain=0.5, grime=GRIME),
        surface(RED, [0.232, 0.062, 0.045], grain=0.5, grime=GRIME),
        surface(OCHRE, [0.310, 0.196, 0.052], grain=0.5, grime=GRIME),
        surface(TEAL, [0.048, 0.168, 0.160], grain=0.5, grime=GRIME),
    ],
}

with open("levels/brutalist.json", "w") as f:
    json.dump(level, f, indent=1)
    f.write("\n")


def count(ns):
    return sum(1 + count(n.get("nodes", [])) for n in ns)


print(f"{count(nodes)} nodes, {len(level['materials'])} materials")
