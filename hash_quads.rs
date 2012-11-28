use cmp::{Eq};
use to_bytes::{IterBytes,Cb};

use hashlife::Quads;

struct HashQuads { quads: Quads }

// Need to make quads hashable so we can hashcons them. This requires wrapping
// them in another type to declare Eq, IterBytes for it.
impl HashQuads: Eq {
  pure fn eq(other: &HashQuads) -> bool {
    do self.quads.alli |i,c| unsafe {
      ptr::addr_of(*c) == ptr::addr_of(other.quads[i])
    }
  }
  pure fn ne(other: &HashQuads) -> bool { !self.eq(other) }
}

impl HashQuads: IterBytes {
  pure fn iter_bytes(lsb0: bool, f: to_bytes::Cb) {
    for self.quads.each |c| unsafe {
      ptr::to_uint(*c).iter_bytes(lsb0, f)
    }
  }
}

// shim until https://github.com/mozilla/rust/pull/4052 lands in master.
impl<A: IterBytes, B: IterBytes> (A,B): IterBytes {
  #[inline(always)]
  pure fn iter_bytes(lsb0: bool, f: to_bytes::Cb) {
    match self {
      (ref a, ref b) => {
        a.iter_bytes(lsb0, f);
        b.iter_bytes(lsb0, f);
      }
    }
  }
}
