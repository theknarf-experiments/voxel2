//! GPU voxel generation: compute passes that evaluate density functions and
//! CSG op lists into per-chunk density buffers (the density arena), plus the
//! classification pass and its async readback.
