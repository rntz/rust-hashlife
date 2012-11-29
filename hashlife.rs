// Core imports
use cmp::{Eq,Ord};
use dvec::{DVec};
use option::{Option};
use either::{Either,Left,Right};

// Std imports
use std::map::{Map,HashMap};

// hlife imports
use hash_quads::HashQuads;

// TODO: garbage collection of cells in the cache
// TODO?: use type-level natural numbers to enforce cell rank invariants

// ---------- NOTE ON REPRESENTATIONS ----------
//
// We represent 4x4 cells using u16. The number 0b_abcd_efgh_ijkl_mnop_u16
// (where a,b,c,...,p are bits 0 or 1) represents the 4x4 grid:
//
//     |a b c d|
//     |e f g h|
//     |i j k l|
//     |m n o p|
//
// We represent 2x2 results using u8. The number 0b_00ab_00cd_u8 represents the
// 2x2 grid:
//
//     |a b|
//     |c d|
//
// Note the two 0-bits of padding between the bits a,b and c,d. This makes
// combining 2x2 results into 4x4 cells easier.

// To represent larger cells, we divide them into quadrants of sub-cells, and
// list them from top-to-bottom, left-to-right. So a cell with quadrants
// [a,b,c,d] looks like the following:
//
//     |a b|
//     |c d|

pub type Cell = @mut Cell_;
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

pub enum Result { Cell(Cell), Two(u8) }

struct World {
  // we have different cons-caches for each size of cell. this makes gc nicer.
  // the base-level cache.
  cache4x4 : HashMap<u16, Cell>,
  // higher-level caches.
  caches : DVec<HashMap<HashQuads, Cell>>,
}


// Misc. utilities
pure fn quadmap<T, U>(arr: &[T * 4], f: fn&(&T) -> U) -> [U * 4] {
  [f(&arr[0]), f(&arr[1]), f(&arr[2]), f(&arr[3])]
}

pure fn rank_to_size(rank: uint) -> uint { 4u << rank }

// finds the smallest rank capable of containing a size x size square
pure fn size_to_rank(size: uint) -> uint {
  let mut rank = 0u;
  while rank_to_size(rank) < size { rank += 1; }
  rank
}

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
impl World {
  fn result(rank: uint, c: Cell) -> Result {
    assert c.rank() == rank;
    match c {
      // Return memorized results if present.
      @Node(_, Some(r)) => Cell(r),
      @Four(_, Some(r)) => Two(r),

      // Compute via life algo.
      @Four(copy bits, None) => {
        // TODO?: hand-optimize this
        let se = next_state_hack(bits),
        sw = next_state_hack(bits >> 1),
        ne = next_state_hack(bits >> 4),
        nw = next_state_hack(bits >> 5);
        let res = (nw << 5) | (ne << 4) | (sw << 1) | se;
        assert res & 0b_0011_0011 == res;
        *c = Four(bits, Some(res));
        Two(res)
      }

      // The interesting case. Uses the algorithm as described by Bill Gosper in
      // "Exploiting regularities in large cellular spaces", Physica 10D, 1984.

      // We don't actually need to copy quads, but I'm not sure how it's possible
      // to convince rust of this.
      @Node(copy quads, None) => {
        // Compute overlapping nonads from quads, then compute their results.
        let nonad_results = do self.quads_to_nonads(rank-1, &quads).map |x| {
          self.result(rank-1, *x)
        };

        const quad_is: [[uint*4]*4] =
          [[0,1,3,4], [1,2,4,5],
           [3,4,6,7], [4,5,7,8]];

        // Merge their results into overlapping quads for halfway future point,
        // then compute results for those quads & merge them.
        let res = self.merge(rank-1, do quadmap(&quad_is) |is| {
          self.result(rank-1, self.merge(rank-1, do quadmap(is) |i| {
              nonad_results[*i]
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
  fn quads_to_nonads(rank: uint, quads: &Quads) -> [Cell * 9] {
    assert do quads.all |x| { x.rank() == rank };

    // Split the quads into sixteenths (which are Results, not Cells).
    let qquads = do quadmap(quads) |x| { split(*x) };

    // Indices of north, west, center, east, south quads
    //
    //  0 1 | 0 1
    //  2 3 | 2 3
    // -----+----
    //  0 1 | 0 1
    //  2 3 | 2 3
    //
    const qquad_is: [[(uint,uint) * 4] * 5] =
      [[(0,1), (1,0), (0,3), (1,2)],  //north
       [(0,2), (0,3), (2,0), (2,1)],  //west
       [(0,3), (1,2), (2,1), (3,0)],  //center
       [(1,2), (1,3), (3,0), (3,1)],  //east
       [(2,1), (3,0), (2,3), (3,2)]]; //south

    // Determine north, west, center, east, south quads
    let nwces = do qquad_is.map |is| {
      self.merge(rank, do quadmap(is) |xy| {
        let (i,j) = *xy;
        qquads[i][j]
      })
    };
    assert do nwces.all |x| { x.rank() == rank };

    // Return overlapping nonads
    [quads[0], nwces[0], quads[1],
     nwces[1], nwces[2], nwces[3],
     quads[2], nwces[4], quads[3]]
  }

  // Merges four results into a cell. This and its helper makeFour are the only
  // functions that performs hash-consing. `rank' must be the rank of the cell
  // that results from merging the results.
  fn merge(rank: uint, results: [Result * 4]) -> Cell {
    let quads = match case_results(rank, &results) {
      Right(bits) => { return self.makeFour(bits); }
      Left(move quads) => move quads
    };
    assert rank > 0;            // if rank == 0 we would already have returned

    // Try looking it up in the cache
    let cache: HashMap<HashQuads, Cell>;
    let hquads = HashQuads { quads: quads };

    if (rank <= self.caches.len()) {
      cache = self.caches.index(rank-1)
    } else {
      // rank is larger than highest rank in cache. if rank isn't the size of the
      // cache, that means the cells we're merging aren't in cache either -- an
      // invariant violation!
      assert rank-1 == self.caches.len();
      cache = HashMap();
      self.caches.push(cache);
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

  fn makeFour(bits: u16) -> Cell {
    return match self.cache4x4.find(bits) {
      Some(c) => c,
      None => @mut Four(bits, None),
    };
  }
}

// `rank' is the rank of a cell which would produce results of these sizes
fn case_results(rank: uint, results: &[Result * 4]) -> Either<[Cell * 4], u16> {
  let r = (results[0], results[1], results[2], results[3]);
  if (rank > 0) {
    match r {
      (Cell(a), Cell(b), Cell(c), Cell(d)) => Left([a,b,c,d]),
      _ => fail fmt!("invariant violation, rank=%u", rank),
    }
  } else {
    match r {
      (Two(nw), Two(ne), Two(sw), Two(se)) => {
        // Reconstruct the bitboard from its corners.
        Right(  (nw as u16) << 10 | (ne as u16) << 8
              | (sw as u16) << 2 | (se as u16))
      }
      _ => fail fmt!("invariant violation, rank=%u", rank),
    }
  }
}

// Does the inverse of World.merge.
fn split(c : Cell) -> [Result * 4] {
  match c {
    @Node(quads, _) => do quadmap(&quads) |x| { Cell(*x) },
    @Four(bits, _) => {
      const mask : u16 = 0b_0011_0011_u16;
      [Two(((bits >> 10) & mask) as u8), Two(((bits >>  8) & mask) as u8),
       Two(((bits >>  2) & mask) as u8), Two( (bits        & mask) as u8)]
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
  // NB. `nalive' counts living cells in the neighborhood, INCLUDING the cell
  // we're calculating the next state of
  let nalive = bitset_hack(grid) + bitset_hack(grid >> 4u) +
    bitset_hack(grid >> 8u);
  match (grid & 0b_0010_0000_u16 > 0u16, nalive) {
    (true, 3..4) | (false, 3) => { 1u8 }
    _ => { 0u8 }
  }
}

// Returns the number of bits set among the low 3 bits.
pure fn bitset_hack(grid : u16) -> u16 {
  (grid & 1) + ((grid >> 1) & 1) + ((grid >> 2) & 1)
}
