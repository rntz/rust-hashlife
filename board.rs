use std::map::{Map,HashMap,Set};

use hashlife::{rank_to_size,size_to_rank};

struct Board {
  live_cells: Set<(uint,uint)>,
  rows: uint,
  cols: uint
}

// Converting boards to cells

// Roughly centers the board in a new cell of the smallest rank capable of
// containing it.
fn board_to_cell(world: &World, board: &Board) -> Cell {
  let rank = size_to_rank(uint::max(board.rows, board.cols));
  let size = rank_to_size(rank);

  let xoff = (size - board.cols) / 2,
      yoff = (size - board.rows) / 2;
  do cell_from_fn(world, rank, xoff, yoff) |i,j| {
    board.live_cells.contains_key((i,j))
  }
}

fn cell_from_set(world: &World, rank: uint, set: Set<(uint,uint)>) -> Cell {
  do cell_from_fn(world, rank, 0, 0) |i,j| {
    set.contains_key((i,j))
  }
}

fn cell_from_fn
  (w: &World, rank: uint, xoff: uint, yoff:uint, state: fn&(uint,uint) -> bool)
  -> Cell
{
  if (rank > 0) {
    let size = rank_to_size(rank);
    let f: fn&(uint,uint) -> Result = |i,j| {
      Cell(cell_from_fn(w, rank-1, xoff+i, yoff+j, state))
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
