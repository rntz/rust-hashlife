// Core imports
use cmp::{Eq,Ord};
use dvec::{DVec};
use option::{Option};
use to_bytes::{IterBytes};

// Std imports
extern mod std;
use map = std::map;
use map::HashMap;

// shim.
impl<A: IterBytes, B: IterBytes> (A,B): IterBytes {
  #[inline(always)]
  pure fn iter_bytes(lsb0: bool, f: to_bytes::Cb) {
    let &(ref a, ref b) = &self;
    a.iter_bytes(lsb0, f);
    b.iter_bytes(lsb0, f);
  }
}

// TODO: garbage collection of cells in the cache
// TODO?: use type-level natural numbers to enforce cell rank invariants

// Note: When representing a grid in a unsigned integer, states are stored
// top-to-bottom, left-to-right. When there are excess bits (ie. using a u8 to
// represent a 2x2 grid, the low bits are significant and the high bits
// ignored). so for example 0b_abcd_u8 (where a,b,c,d are bits) represents the
// 2x2 grid:
//
//     |a b|
//     |c d|
//
// while 0b_abcd_efgh_ijkl_mnop_u16 represents the 4x4 grid:
//
//     |a b c d|
//     |e f g h|
//     |i j k l|
//     |m n o p|
//
// This means that putting together four 2x2 results into a 4x4 cell requires
// some bit-swizzling.

// When representing a grid by a list of quadrants or other things, they are
// listed from top-to-bottom, left-to-right. So a cell with quadrants [a,b,c,d]
// looks like the following:
//
//     |a b|
//     |c d|

type Cell = @mut Cell_;
// TODO: make branches into structs with the appropriate fields mutable
enum Cell_ {
  // quads, result
  Node(Quads, Option<Cell>),
  // 4x4 grid: bits, result
  Four(u16, Option<u8>)
}
// would like these to be unique, or even better, a 4-tuple, but unique results
// in ownership/copy/aliasing issues and 4-tuple is a PITA.
type Quads = [Cell * 4];

struct HashQuads { quads: Quads }

enum Result { Cell(Cell), Two(u8) }

struct Env_ {
  // we have different cons-caches for each size of cell. this makes gc nicer.
  // the base-level cache.
  cache4x4 : HashMap<u16, Cell>,
  // higher-level caches.
  caches : DVec<HashMap<HashQuads, Cell>>,
}
type Env = &Env_;


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


// Misc. utilities
fn quadmap<T:Copy, U:Copy>(arr: [T * 4], f: fn&(T) -> U) -> [U * 4] {
  [f(arr[0]), f(arr[1]), f(arr[2]), f(arr[3])]
}

pure fn rank_to_size(rank: uint) -> uint { 4u << rank }

impl Cell {
  // note: this is a slow function, call only for debugging/assert purposes
  pure fn rank(&const self) -> uint {
    match *self {
      @Four(_,_) => 0u,
      @Node(quads, result) => {
        let rank = quads[0].rank();
        assert do quads.all |x| { x.rank() == rank };
        assert match result {
          Some(r) => rank == r.rank(),
          None => true
        };
        rank+1
      }
    }
  }

  pure fn size(&const self) -> uint { rank_to_size(self.rank()) }
}


// Macrocell ops
fn result(env: Env, rank: uint, c: Cell) -> Result {
  assert c.rank() == rank;
  match c {
    // Return memorized results if present.
    @Node(_, Some(r)) => Cell(r),
    @Four(_, Some(r)) => Two(r),

    // Compute via life algo.
    @Four(copy bits, None) => {
      // TODO: hand-optimize this?
      let sw = next_state_hack(bits),
      se = next_state_hack(bits >> 1),
      nw = next_state_hack(bits >> 4),
      ne = next_state_hack(bits >> 5);
      let res = (nw << 3) + (ne << 2) + (se << 1) + sw;
      *c = Four(bits, Some(res));
      Two(res)
    }

    // The interesting case. Uses the algorithm as described by Bill Gosper in
    // "Exploiting regularities in large cellular spaces", Physica 10D, 1984.

    // We don't actually need to copy quads, but I'm not sure how it's possible
    // to convince rust of this.
    @Node(copy quads, None) => {
      // Compute overlapping nonads from quads, then compute their results.
      let nonad_results = do quads_to_nonads(env, rank-1, &quads).map |x| {
        result(env, rank-1, *x)
      };

      const quad_is: [[uint*4]*4] =
        [[0,1,3,4], [1,2,4,5],
         [3,4,6,7], [4,5,7,8]];

      // Merge their results into overlapping quads for halfway future point,
      // then compute results for those quads & merge them.
      let res = merge(env, rank-1, do quadmap(quad_is) |is| {
        result(env, rank-1, merge(env, rank-2, do quadmap(is) |i| {
          nonad_results[i]
        }))
      });

      *c = Node(quads, Some(res));
      Cell(res)
    }
  }
}

// Given nw, ne, sw, se quadrants, compute north, south, east, west, and center
// cells and combine these in proper order (see comment at beginning offile)
// into a vector of overlapping nonads.
//
// `rank' is the rank of each quadrant, and also the rank of the output
// quadrants.
fn quads_to_nonads(env: Env, rank: uint, quads: &Quads) -> [Cell * 9] {
  assert do quads.all |x| { x.rank() == rank };

  // Split the quads into sixteenths (which are Results, not Cells).
  let qquads = do quads.flat_map |x| { split(*x) };

  // indices of north, west, center, east, south quads
  // should be const, but I can't write it's type :/
  //  0  1  2  3
  //  4  5  6  7
  //  8  9 10 11
  // 12 13 14 15
  const qquad_is: [[uint * 4] * 5] =
    [                [ 1, 2, 5, 6]
     ,[ 4, 5, 8, 9], [ 5, 6, 9,10], [ 6, 7,10,11]
     ,               [ 9,10,13,14]               ];

  // Determine north, west, center, east, south quads
  let nwces = do qquad_is.map |is| {
    merge(env, rank-1, do quadmap(*is) |i| { qquads[i] })
  };

  // Return overlapping nonads
  [quads[0], nwces[0], quads[1],
   nwces[1], nwces[2], nwces[3],
   quads[2], nwces[4], quads[3]]
}

// Merges four results into a cell. This and its helper makeFour are the only
// function that performs hash-consing. `rank' is the rank of each quad; the
// rank of the resulting cell is `rank+1'.
fn merge(env: Env, rank: uint, results: [Result * 4]) -> Cell {
  let r = (results[0], results[1], results[2], results[3]);
  if (rank == 0) {
    let (nw,ne,sw,se) = match r {
      (Two(a), Two(b), Two(c), Two(d)) => (a,b,c,d),
      _ => fail ~"invariant violation"
    };

    // Reconstruct the right bitboard from its corners.
    const hi: u8 = 0b1100_u8;
    const lo: u8 = 0b11_u8;
    let bits: u16 =
      ((nw & hi) as u16 << 12) | ((ne & hi) as u16 << 10)
      | ((nw & lo) as u16 << 10) | ((ne & lo) as u16 <<  8)
      | ((sw & hi) as u16 <<  4) | ((se & hi) as u16 <<  2)
      | ((sw & lo) as u16 <<  2) | ((se & lo) as u16);

    // Look up bitboard in cache.
    return makeFour(env, bits);
  }

  // Try looking it up in the cache
  let cache: HashMap<HashQuads, Cell>;
  let quads = match r {
    (Cell(a), Cell(b), Cell(c), Cell(d)) => [a,b,c,d],
    _ => fail ~"invariant violation"
  };
  let hquads = HashQuads { quads: quads };

  if (rank <= env.caches.len()) {
    cache = env.caches.index(rank-1)
  } else {
    // rank is larger than highest rank in cache. if rank isn't the size of the
    // cache, that means the cells we're merging aren't in cache either -- an
    // invariant violation!
    assert rank-1 == env.caches.len();
    cache = HashMap();
    env.caches.push(cache);
  }

  match cache.find(hquads) {
    Some(c) => c,
    None => {
      // Construct a cell & put it in the cache.
      let c = @mut Node(quads, None);
      cache.insert(hquads, c);
      c
    }
  }
}

fn makeFour(env: Env, bits: u16) -> Cell {
  return match env.cache4x4.find(bits) {
    Some(c) => c,
    None => @mut Four(bits, None),
  };
}

// Does the inverse of merge.
fn split(c : Cell) -> ~[Result] {
  match c {
    @Node(quads, _) => do quads.map |x| { Cell(*x) },
    @Four(bits, _) => {
      const mask : u16 = 0b0111_0111_0111_u16;
      ~[        Two((bits & mask) as u8), Two(((bits >> 1u) & mask) as u8),
        Two(((bits >> 4u) & mask) as u8), Two(((bits >> 5u) & mask) as u8)]
    }
  }
}

// Some nice bit-ops.

// Determines the next state of the middle lower-right cell in a four-by-four
// grid, ie the cell marked "x" below:
//
//     |_ _ _ _|
//     |_ _ _ _|
//     |_ _ x _|
//     |_ _ _ _|

pure fn next_state_hack(grid : u16) -> u8 {
  let nalive = bitset_hack(grid) + bitset_hack(grid >> 4u) +
    bitset_hack(grid >> 8u);
  match (grid & 0b10_0000_u16 > 0u16, nalive) {
    (true, 3u16 .. 4u16) | (false, 3u16) => { 1u8 }
    _ => { 0u8 }
  }
}

// Returns the number of bits set among the low 3 bits.
pure fn bitset_hack(grid : u16) -> u16 {
  (grid & 1u16) + ((grid & 2u16) >> 1u) + ((grid & 4u16) >> 2u)
}


// Creating cells
fn cell_from_map<T:map::Map<(uint,uint), ()>>
  (env: Env, rank: uint, map: T) -> Cell
{
  do cell_from_fn(env, rank) |i,j| { map.contains_key((i,j)) }
}

fn cell_from_fn(env: Env, rank: uint, state: fn&(uint,uint) -> bool) -> Cell {
  cell_from_fn_offset(env, rank, state, 0u, 0u)
}

fn cell_from_fn_offset
  (env: Env, rank: uint, state: fn&(uint,uint) -> bool, xoff: uint, yoff:uint)
  -> Cell
{
  if (rank > 0) {
    let size = rank_to_size(rank);
    let f: fn&(uint,uint) -> Result = |i,j| {
      Cell(cell_from_fn_offset(env, rank-1, state, xoff+i, yoff+j))
    };
    merge(env, rank, [f(0,0), f(size,0),
                      f(0,size), f(size,size)])
  } else {
    let mut bits: u16 = 0;
    let mut i: uint = 0;
    while (i < 16) {
      let bit = if state(xoff + i / 4, yoff + i % 4) { 0 } else { 1 };
      bits = (bits << 1) | bit;
    }
    makeFour(env, bits)
  }
}


// Parsing plaintext into cells.
//
// Parses the pattern into the smallest cell that will contain it, roughly
// centered in that cell.

// Main program
fn main() {
  let env: Env = &Env_ {
    cache4x4: HashMap(),
    caches: DVec(),
  };

  *env;
  io::print("fux");
}
