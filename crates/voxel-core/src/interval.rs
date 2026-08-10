//! Interval arithmetic: what a value CAN be over a region, rather than
//! what it is at a point.
//!
//! The generator is a program over a register file, evaluated per sample.
//! Evaluated on intervals instead, the same program answers a different
//! and much cheaper question: what can its SDF be anywhere in this box?
//! An answer that is entirely positive means all air, entirely negative
//! means all solid, and either way there is no surface — nothing to mesh,
//! nothing to draw, and no reason to have spent a 38³ density pass and a
//! GPU round trip discovering it.
//!
//! **Every operation here must be conservative.** The result has to
//! contain every value the exact operation could produce over the inputs.
//! Too wide only costs work that would have been done anyway; too narrow
//! deletes world. The tests are written that way round: they sample the
//! exact operation and assert containment, rather than checking bounds
//! against hand-computed numbers.

/// A closed range of values, `lo <= hi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lo: f32,
    pub hi: f32,
}

impl Interval {
    /// The two ends in either order.
    pub fn new(a: f32, b: f32) -> Self {
        Self {
            lo: a.min(b),
            hi: a.max(b),
        }
    }

    /// A value known exactly.
    pub fn point(v: f32) -> Self {
        Self { lo: v, hi: v }
    }

    /// `[-mag, mag]` — the bound of something that oscillates about zero,
    /// which is most of what a noise field is.
    pub fn symmetric(mag: f32) -> Self {
        let mag = mag.abs();
        Self { lo: -mag, hi: mag }
    }

    /// Everything. What an op that cannot be bounded contributes.
    pub const UNBOUNDED: Self = Self {
        lo: f32::NEG_INFINITY,
        hi: f32::INFINITY,
    };

    pub fn contains(self, v: f32) -> bool {
        v >= self.lo && v <= self.hi
    }

    /// Strictly above zero everywhere: for an SDF, all air.
    pub fn is_positive(self) -> bool {
        self.lo > 0.0
    }

    /// Strictly below zero everywhere: for an SDF, all solid.
    pub fn is_negative(self) -> bool {
        self.hi < 0.0
    }

    /// Could be zero, so an SDF could have a surface in it.
    pub fn straddles_zero(self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn width(self) -> f32 {
        self.hi - self.lo
    }

    /// The smallest interval containing both.
    pub fn union(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Grow by `pad` on each side.
    pub fn widen(self, pad: f32) -> Self {
        Self {
            lo: self.lo - pad.abs(),
            hi: self.hi + pad.abs(),
        }
    }

    /// Bound of `min(a, b)` taken pointwise — NOT the smaller interval.
    /// Two SDFs unioned take the pointwise minimum, and either can be the
    /// smaller at any given point.
    pub fn min(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.min(other.hi),
        }
    }

    /// Bound of `max(a, b)` taken pointwise. A CSG cut is a max.
    pub fn max(self, other: Self) -> Self {
        Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Bound of `|x|`. Zero is the minimum only if the interval reaches
    /// it — `|[2, 5]|` is `[2, 5]`, not `[0, 5]`.
    pub fn abs(self) -> Self {
        if self.straddles_zero() {
            Self {
                lo: 0.0,
                hi: self.lo.abs().max(self.hi.abs()),
            }
        } else {
            Self::new(self.lo.abs(), self.hi.abs())
        }
    }

    /// Bound of `clamp(x, lo, hi)`.
    pub fn clamp(self, lo: f32, hi: f32) -> Self {
        Self {
            lo: self.lo.clamp(lo, hi),
            hi: self.hi.clamp(lo, hi),
        }
    }
}

impl std::ops::Add for Interval {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }
}

impl std::ops::Add<f32> for Interval {
    type Output = Self;
    fn add(self, v: f32) -> Self {
        Self {
            lo: self.lo + v,
            hi: self.hi + v,
        }
    }
}

impl std::ops::Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

impl std::ops::Sub for Interval {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl std::ops::Sub<f32> for Interval {
    type Output = Self;
    fn sub(self, v: f32) -> Self {
        self + (-v)
    }
}

/// Scaling by a constant flips the ends when the constant is negative.
impl std::ops::Mul<f32> for Interval {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self::new(self.lo * k, self.hi * k)
    }
}

/// The product of two ranges is bounded by the extremes of the four
/// corner products — `[-2, 1] * [-3, 1]` reaches 6, which neither
/// endpoint pairing alone would find.
impl std::ops::Mul for Interval {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let a = self.lo * other.lo;
        let b = self.lo * other.hi;
        let c = self.hi * other.lo;
        let d = self.hi * other.hi;
        Self {
            lo: a.min(b).min(c).min(d),
            hi: a.max(b).max(c).max(d),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::Rng;

    /// Sample the exact operation over the inputs and assert the interval
    /// contains every result. Conservativeness is the whole contract:
    /// this is the test that matters, and it is written so a wrong bound
    /// fails rather than a differently-derived one.
    fn assert_bounds(
        rng: &mut Rng,
        exact: impl Fn(f32, f32) -> f32,
        bound: impl Fn(Interval, Interval) -> Interval,
    ) {
        for _ in 0..2_000 {
            let span = |rng: &mut Rng| {
                let a = (rng.next_f32() - 0.5) * 40.0;
                let b = (rng.next_f32() - 0.5) * 40.0;
                Interval::new(a, b)
            };
            let (x, y) = (span(rng), span(rng));
            let got = bound(x, y);
            for _ in 0..24 {
                let px = x.lo + (x.hi - x.lo) * rng.next_f32();
                let py = y.lo + (y.hi - y.lo) * rng.next_f32();
                let v = exact(px, py);
                assert!(
                    got.contains(v),
                    "{v} from ({px}, {py}) escapes {got:?} for {x:?}, {y:?}"
                );
            }
            // The ends are attainable, so they must be inside too.
            for (px, py) in [(x.lo, y.lo), (x.lo, y.hi), (x.hi, y.lo), (x.hi, y.hi)] {
                assert!(got.contains(exact(px, py)), "corner escapes {got:?}");
            }
        }
    }

    #[test]
    fn arithmetic_bounds_what_it_claims() {
        let mut rng = Rng::new(0x1E7E);
        assert_bounds(&mut rng, |a, b| a + b, |a, b| a + b);
        assert_bounds(&mut rng, |a, b| a - b, |a, b| a - b);
        assert_bounds(&mut rng, |a, b| a * b, |a, b| a * b);
        assert_bounds(&mut rng, |a, b| a.min(b), Interval::min);
        assert_bounds(&mut rng, |a, b| a.max(b), Interval::max);
        assert_bounds(&mut rng, |a, _| a.abs(), |a, _| a.abs());
        assert_bounds(&mut rng, |a, _| -a, |a, _| -a);
        assert_bounds(
            &mut rng,
            |a, _| a.clamp(-3.0, 7.0),
            |a, _| a.clamp(-3.0, 7.0),
        );
    }

    /// Scaling by a negative constant swaps the ends; getting this wrong
    /// produces an inverted interval that contains nothing.
    #[test]
    fn scaling_by_a_negative_keeps_the_ends_ordered() {
        let i = Interval::new(2.0, 5.0) * -3.0;
        assert_eq!(i, Interval::new(-15.0, -6.0));
        assert!(i.lo <= i.hi);
        assert!(i.contains(-9.0));
    }

    /// The product of two straddling ranges reaches further than any
    /// single endpoint pairing suggests.
    #[test]
    fn a_product_finds_the_corner_that_matters() {
        let p = Interval::new(-2.0, 1.0) * Interval::new(-3.0, 1.0);
        assert_eq!(p, Interval::new(-3.0, 6.0));
    }

    /// `|x|` of a range that does not reach zero does not reach zero.
    #[test]
    fn abs_keeps_its_distance_from_zero() {
        assert_eq!(Interval::new(2.0, 5.0).abs(), Interval::new(2.0, 5.0));
        assert_eq!(Interval::new(-5.0, -2.0).abs(), Interval::new(2.0, 5.0));
        assert_eq!(Interval::new(-5.0, 2.0).abs(), Interval::new(0.0, 5.0));
    }

    /// The three answers an SDF bound gives, which are the whole point.
    #[test]
    fn an_sdf_bound_classifies_a_region() {
        assert!(Interval::new(3.0, 9.0).is_positive()); // all air
        assert!(Interval::new(-9.0, -3.0).is_negative()); // all solid
        assert!(Interval::new(-1.0, 4.0).straddles_zero()); // a surface
        assert!(!Interval::new(3.0, 9.0).straddles_zero());
        // Touching zero counts as a surface: the boundary is the surface.
        assert!(Interval::new(0.0, 4.0).straddles_zero());
    }

    /// An op nobody has bounded contributes "anywhere", which must not
    /// classify as either air or solid.
    #[test]
    fn unbounded_classifies_as_nothing() {
        assert!(!Interval::UNBOUNDED.is_positive());
        assert!(!Interval::UNBOUNDED.is_negative());
        assert!(Interval::UNBOUNDED.straddles_zero());
    }
}
