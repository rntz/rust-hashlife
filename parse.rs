use std::map::{Map,Set};
use hashlife::{rank_to_size};

// Parsing plaintext into cells.
//
// Parses the pattern into the smallest cell that will contain it, roughly
// centered in that cell.


// Creating cells
fn cell_from_map(w: &World, rank: uint, map: Map<(uint,uint), ()>) -> Cell
{
  do cell_from_fn(w, rank) |i,j| { map.contains_key((i,j)) }
}

fn cell_from_fn(w: &World, rank: uint, state: fn&(uint,uint) -> bool) -> Cell {
  cell_from_fn_offset(w, rank, state, 0u, 0u)
}

fn cell_from_fn_offset
  (w: &World, rank: uint, state: fn&(uint,uint) -> bool, xoff: uint, yoff:uint)
  -> Cell
{
  if (rank > 0) {
    let size = rank_to_size(rank);
    let f: fn&(uint,uint) -> Result = |i,j| {
      Cell(cell_from_fn_offset(w, rank-1, state, xoff+i, yoff+j))
    };
    w.merge(rank, [f(0,0), f(size,0),
                   f(0,size), f(size,size)])
  } else {
    let mut bits: u16 = 0;
    let mut i: uint = 0;
    while (i < 16) {
      let bit = if state(xoff + i / 4, yoff + i % 4) { 0 } else { 1 };
      bits = (bits << 1) | bit;
    }
    w.makeFour(bits)
  }
}
